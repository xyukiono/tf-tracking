import os
import glob
import numpy as np
import tensorflow as tf
from imageio import imread, imsave
from cf_utils import get_exemplar_images
from utils.misc import get_center
from utils.io_utils import read_text

class SiameseVIDDataset(object):
    def __init__(self, context_amount=0.5, template_image_size=127, query_image_size=255, max_seq_length=500, max_motion=0.5, loc_thresh=16, num_threads=8):
        self.context_amount = context_amount
        self.z_image_size = template_image_size
        self.x_image_size = query_image_size
        self.num_threads = num_threads
        self.max_seq_length = max_seq_length
        self.max_motion = max_motion
        self.loc_thresh = loc_thresh

    def get_dataset(self, root_dir, phase='train', batch_size=16, shuffle=True, num_epoch=None, seed=None):
        if phase == 'train':
            data_dir = os.path.join(root_dir, 'Data/VID/train/')
            ann_dir = os.path.join(root_dir, 'tfann/train')
        elif phase == 'val':
            data_dir = os.path.join(root_dir, 'Data/VID/val/')
            ann_dir = os.path.join(root_dir, 'tfann/val')
        else:
            raise ValueError('Unknown phase: {}'.format(config.phase))

        sub_dirs = [x.path for x in os.scandir(ann_dir) if x.is_dir()]
        tf_anns = []
        filenames = []
        imsizes = []
        bboxes = []
        seq_inds = []
        seq_lengths = []
        curr_id = 0

        for t, sub_dir in enumerate(sub_dirs):
            ann_files = [x.path for x in os.scandir(sub_dir) if x.name.endswith('.npz')]
            print('#{}/{} Load {} annotations...'.format(t, len(sub_dirs), len(ann_files)))
            for ann_file in ann_files:
                # print(ann_file)
                ann_data = np.load(ann_file)
                N = len(ann_data['filename'])
                _filenames = ann_data['filename']
                _imsizes = ann_data['imsize']
                _bboxes = ann_data['bbox']
                if self.max_seq_length > 0 and N > self.max_seq_length:
                    indices = np.arange(N)
                    np.random.shuffle(indices)
                    indices = np.sort(indices[:self.max_seq_length])
                    _filenames = _filenames[indices]
                    _imsizes = _imsizes[indices]
                    _bboxes = _bboxes[indices]
                    N = self.max_seq_length

                # conver box-format: from [left,top,right,bottom] to [cy, cx, height, width]
                _bboxes = _bboxes.astype(np.float32)
                _x1s = _bboxes[:,0]
                _y1s = _bboxes[:,1]
                _x2s = _bboxes[:,2]
                _y2s = _bboxes[:,3]
                _new_bboxes = np.zeros_like(_bboxes)
                _new_bboxes[:,0] = (_y1s + _y2s) * 0.5
                _new_bboxes[:,1] = (_x1s + _x2s) * 0.5
                _new_bboxes[:,2] = _x2s - _x1s
                _new_bboxes[:,3] = _y2s - _y1s
                # print(_bboxes[0], '-->', _new_bboxes[0])


                filenames.append(_filenames)
                imsizes.append(_imsizes)
                bboxes.append(_new_bboxes)
                seq_inds.append(np.ones(N, dtype=np.int32)*curr_id)
                seq_lengths.append(N)
                curr_id += 1
        filenames = np.concatenate(filenames, axis=0)
        imsizes = np.concatenate(imsizes, axis=0)
        bboxes = np.concatenate(bboxes, axis=0)
        seq_inds = np.concatenate(seq_inds, axis=0)
        seq_lengths = np.array(seq_lengths, dtype=np.int32)
        seq_offsets = np.concatenate([np.zeros(1), np.cumsum(seq_lengths)]).astype(np.int32)

        assert len(filenames) == len(imsizes) == len(bboxes) == len(seq_inds)
        assert len(seq_lengths) == curr_id

        self.data_root_dir = tf.convert_to_tensor(data_dir) # data_dir must end with '/'
        self.filenames = tf.convert_to_tensor(filenames)
        self.imsizes = tf.convert_to_tensor(imsizes)
        self.bboxes = tf.convert_to_tensor(bboxes)
        self.seq_inds = tf.convert_to_tensor(seq_inds)

        self.seq_lengths = tf.convert_to_tensor(seq_lengths)
        self.seq_offsets = tf.convert_to_tensor(seq_offsets)
        self.num_seqs = curr_id
        self.num_examples = len(filenames)
        print('#SEQ={} #frames={}, min-len={}, max-len={}'.format(len(seq_lengths), self.num_examples, seq_lengths.min(), seq_lengths.max()))

        dataset = tf.data.Dataset.range(self.num_examples)
        if shuffle:
            dataset = dataset.shuffle(self.num_examples, seed=seed)
        dataset = dataset.repeat(count=num_epoch)
        dataset = dataset.map(self.parser, num_parallel_calls=self.num_threads)
        dataset = dataset.batch(batch_size)

        return dataset

    def parser(self, tgt_id):
        tgt_id = tf.cast(tgt_id, tf.int32) # tf.int64->tf.int32
        seq_id = self.seq_inds[tgt_id]
        length = self.seq_lengths[seq_id]

        ref_id = self.seq_offsets[seq_id] + tf.random_uniform((), 0, length, dtype=tf.int32) # low <= val < high
        image_z = self.decode_image(self.data_root_dir+self.filenames[tgt_id])
        image_x = self.decode_image(self.data_root_dir+self.filenames[ref_id])

        box_z = self.bboxes[tgt_id]
        box_x = self.bboxes[ref_id]

        scale_factor = 1.0
        # scale_factor = tf.random_uniform((), 1/scale_step, scale_step)

        patch_z, _, _ = self.build_search_image(image_z, box_z, 1.0)
        _cy, _cx, _height, _width = tf.unstack(box_x)
        _max_length = tf.maximum(_height, _width)
        motion_x = _max_length * self.max_motion * tf.random_uniform((), -1.0, 1.0)
        motion_y = _max_length * self.max_motion * tf.random_uniform((), -1.0, 1.0)

        box_x_perturb = tf.stack([_cy+motion_y, _cx+motion_x, _height, _width])

        patch_x, _, crop_box = self.build_search_image(image_x, box_x_perturb, scale_factor)

        if self.z_image_size < self.x_image_size:
            patch_z = patch_z[None]
            patch_z = get_exemplar_images(patch_z, [self.z_image_size, self.z_image_size])
            patch_z = patch_z[0]


        patch_z.set_shape([self.z_image_size, self.z_image_size, 3])
        patch_x.set_shape([self.x_image_size, self.x_image_size, 3])

        # Ground truth
        x_size = self.x_image_size
        im_height, im_width = tf.unstack(tf.shape(image_x))[:2]
        y1, x1, y2, x2 = tf.unstack(crop_box)
        y_ratio = float(x_size) / ((y2-y1)*tf.to_float(im_height))
        x_ratio = float(x_size) / ((x2-x1)*tf.to_float(im_width))
        loc_patch_center = x_size * 0.5
        loc_x = loc_patch_center - motion_x * x_ratio
        loc_y = loc_patch_center - motion_y * y_ratio
        response = self.build_response(patch_x, loc_x, loc_y)
        response.set_shape([self.x_image_size, self.x_image_size, 1])
        return patch_z, patch_x, response

    def build_search_image(self, image, bbox, scale_factor):
        context_amount = self.context_amount
        size_z = self.z_image_size
        size_x = self.x_image_size

        # image: [H,W,3]
        # bbox: [4], cy,cx,height,width
        frame_sz = tf.shape(image)
        target_yx = bbox[0:2] #y,x
        target_size = bbox[2:4] # height, width
        avg_chan = tf.reduce_mean(image, axis=(0, 1), name='avg_chan')

        # Compute base values
        base_z_size = target_size
        base_z_context_size = base_z_size + context_amount * tf.reduce_sum(base_z_size) # w+2p, h+2p
        base_s_z = tf.sqrt(tf.reduce_prod(base_z_context_size))  # Canonical size
        base_scale_z = tf.div(tf.to_float(size_z), base_s_z) # s = sqrt(A**2/((w+2p)(h+2p))
        d_search = (size_x - size_z) / 2.0
        base_pad = tf.div(d_search, base_scale_z)
        base_s_x = base_s_z + 2 * base_pad
        base_scale_x = tf.div(tf.to_float(size_x), base_s_x)

        s_x = scale_factor * base_s_x
        frame_sz_1 = tf.to_float(frame_sz[0:2] - 1)
        topleft = tf.div(target_yx - get_center(s_x), frame_sz_1)
        bottomright = tf.div(target_yx + get_center(s_x), frame_sz_1)
        crop_box = tf.concat([topleft, bottomright], axis=0)
        scale_x = base_scale_x / scale_factor

        image_minus_avg = tf.expand_dims(image - avg_chan, 0)
        image_cropped = tf.image.crop_and_resize(image_minus_avg, crop_box[None],
                                                 box_ind=tf.zeros((1), tf.int32),
                                                 crop_size=[size_x, size_x])
        search_image = image_cropped + avg_chan
        search_image = search_image[0] # [1,H,W,3] --> [H,W,3]
        return search_image, scale_x, crop_box

    def build_template(self, search_image):
        size_z = self.z_image_size

        # Exemplar image lies at the center of the search image in the first frame
        exemplar_images = get_exemplar_images(search_image[None], [size_z, size_z])
        return exemplar_images[0]

    def decode_image(self, filename):
        image = tf.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3, dct_method="INTEGER_ACCURATE")
        image = tf.cast(image, tf.float32)
        return image

    def build_response(self, patch, mu_x, mu_y):
        height, width = tf.unstack(tf.shape(patch))[:2]
        pos_x, pos_y = tf.meshgrid(tf.range(width), tf.range(height))
        pos_x = tf.cast(pos_x, tf.float32)
        pos_y = tf.cast(pos_y, tf.float32)
        distance = tf.sqrt((pos_x-mu_x)**2+(pos_y-mu_y)**2)
        response = 2 * tf.to_float(tf.less(distance, self.loc_thresh)) - 1
        return response[...,None] # [H,W,1]


class CFVIDDataset(object):
    def __init__(self, context_amount=0.5, template_image_size=127, query_image_size=255, max_seq_length=500, max_motion=0.5, loc_thresh=16, num_threads=8):
        self.context_amount = context_amount
        self.z_image_size = template_image_size
        self.x_image_size = query_image_size
        self.num_threads = num_threads
        self.max_seq_length = max_seq_length
        self.max_motion = max_motion
        self.loc_thresh = loc_thresh

    def get_dataset(self, root_dir, phase='train', batch_size=16, shuffle=True, num_epoch=None, seed=None):
        if phase == 'train':
            data_dir = os.path.join(root_dir, 'Data/VID/train/')
            ann_dir = os.path.join(root_dir, 'tfann/train')
        elif phase == 'val':
            data_dir = os.path.join(root_dir, 'Data/VID/val/')
            ann_dir = os.path.join(root_dir, 'tfann/val')
        else:
            raise ValueError('Unknown phase: {}'.format(config.phase))

        sub_dirs = [x.path for x in os.scandir(ann_dir) if x.is_dir()]
        tf_anns = []
        filenames = []
        imsizes = []
        bboxes = []
        seq_inds = []
        seq_lengths = []
        curr_id = 0

        for t, sub_dir in enumerate(sub_dirs):
            ann_files = [x.path for x in os.scandir(sub_dir) if x.name.endswith('.npz')]
            print('#{}/{} Load {} annotations...'.format(t, len(sub_dirs), len(ann_files)))
            for ann_file in ann_files:
                # print(ann_file)
                ann_data = np.load(ann_file)
                N = len(ann_data['filename'])
                _filenames = ann_data['filename']
                _imsizes = ann_data['imsize']
                _bboxes = ann_data['bbox']
                if self.max_seq_length > 0 and N > self.max_seq_length:
                    indices = np.arange(N)
                    np.random.shuffle(indices)
                    indices = np.sort(indices[:self.max_seq_length])
                    _filenames = _filenames[indices]
                    _imsizes = _imsizes[indices]
                    _bboxes = _bboxes[indices]
                    N = self.max_seq_length

                # conver box-format: from [left,top,right,bottom] to [cy, cx, height, width]
                _bboxes = _bboxes.astype(np.float32)
                _x1s = _bboxes[:,0]
                _y1s = _bboxes[:,1]
                _x2s = _bboxes[:,2]
                _y2s = _bboxes[:,3]
                _new_bboxes = np.zeros_like(_bboxes)
                _new_bboxes[:,0] = (_y1s + _y2s) * 0.5
                _new_bboxes[:,1] = (_x1s + _x2s) * 0.5
                _new_bboxes[:,2] = _x2s - _x1s
                _new_bboxes[:,3] = _y2s - _y1s
                # print(_bboxes[0], '-->', _new_bboxes[0])


                filenames.append(_filenames)
                imsizes.append(_imsizes)
                bboxes.append(_new_bboxes)
                seq_inds.append(np.ones(N, dtype=np.int32)*curr_id)
                seq_lengths.append(N)
                curr_id += 1
        filenames = np.concatenate(filenames, axis=0)
        imsizes = np.concatenate(imsizes, axis=0)
        bboxes = np.concatenate(bboxes, axis=0)
        seq_inds = np.concatenate(seq_inds, axis=0)
        seq_lengths = np.array(seq_lengths, dtype=np.int32)
        seq_offsets = np.concatenate([np.zeros(1), np.cumsum(seq_lengths)]).astype(np.int32)

        assert len(filenames) == len(imsizes) == len(bboxes) == len(seq_inds)
        assert len(seq_lengths) == curr_id

        self.data_root_dir = tf.convert_to_tensor(data_dir) # data_dir must end with '/'
        self.filenames = tf.convert_to_tensor(filenames)
        self.imsizes = tf.convert_to_tensor(imsizes)
        self.bboxes = tf.convert_to_tensor(bboxes)
        self.seq_inds = tf.convert_to_tensor(seq_inds)

        self.seq_lengths = tf.convert_to_tensor(seq_lengths)
        self.seq_offsets = tf.convert_to_tensor(seq_offsets)
        self.num_seqs = curr_id
        self.num_examples = len(filenames)
        print('#SEQ={} #frames={}, min-len={}, max-len={}'.format(len(seq_lengths), self.num_examples, seq_lengths.min(), seq_lengths.max()))

        dataset = tf.data.Dataset.range(self.num_examples)
        if shuffle:
            dataset = dataset.shuffle(self.num_examples, seed=seed)
        dataset = dataset.repeat(count=num_epoch)
        dataset = dataset.map(self.parser, num_parallel_calls=self.num_threads)
        dataset = dataset.batch(batch_size)

        return dataset

    def parser(self, tgt_id):
        tgt_id = tf.cast(tgt_id, tf.int32) # tf.int64->tf.int32
        seq_id = self.seq_inds[tgt_id]
        length = self.seq_lengths[seq_id]

        ref_id = self.seq_offsets[seq_id] + tf.random_uniform((), 0, length, dtype=tf.int32) # low <= val < high
        image_z = self.decode_image(self.data_root_dir+self.filenames[tgt_id])
        image_x = self.decode_image(self.data_root_dir+self.filenames[ref_id])

        box_z = self.bboxes[tgt_id]
        box_x = self.bboxes[ref_id]

        scale_factor = 1.0
        # scale_factor = tf.random_uniform((), 1/scale_step, scale_step)

        patch_z, _, _ = self.build_search_image(image_z, box_z, 1.0)
        _cy, _cx, _height, _width = tf.unstack(box_x)
        _max_length = tf.maximum(_height, _width)
        motion_x = _max_length * self.max_motion * tf.random_uniform((), -1.0, 1.0)
        motion_y = _max_length * self.max_motion * tf.random_uniform((), -1.0, 1.0)

        box_x_perturb = tf.stack([_cy+motion_y, _cx+motion_x, _height, _width])

        patch_x, _, crop_box = self.build_search_image(image_x, box_x_perturb, scale_factor)

        if self.z_image_size < self.x_image_size:
            patch_z = patch_z[None]
            patch_z = get_exemplar_images(patch_z, [self.z_image_size, self.z_image_size])
            patch_z = patch_z[0]


        patch_z.set_shape([self.z_image_size, self.z_image_size, 3])
        patch_x.set_shape([self.x_image_size, self.x_image_size, 3])

        # Ground truth
        x_size = self.x_image_size
        im_height, im_width = tf.unstack(tf.shape(image_x))[:2]
        y1, x1, y2, x2 = tf.unstack(crop_box)
        y_ratio = float(x_size) / ((y2-y1)*tf.to_float(im_height))
        x_ratio = float(x_size) / ((x2-x1)*tf.to_float(im_width))
        loc_patch_center = x_size * 0.5
        loc_x = loc_patch_center - motion_x * x_ratio
        loc_y = loc_patch_center - motion_y * y_ratio
        response = self.build_gauss_response(patch_x, loc_x, loc_y)
        response.set_shape([self.x_image_size, self.x_image_size, 1])
        return patch_z, patch_x, response

    def build_search_image(self, image, bbox, scale_factor):
        context_amount = self.context_amount
        size_z = self.z_image_size
        size_x = self.x_image_size

        # image: [H,W,3]
        # bbox: [4], cy,cx,height,width
        frame_sz = tf.shape(image)
        target_yx = bbox[0:2] #y,x
        target_size = bbox[2:4] # height, width
        avg_chan = tf.reduce_mean(image, axis=(0, 1), name='avg_chan')

        # Compute base values
        base_z_size = target_size
        base_z_context_size = base_z_size + context_amount * tf.reduce_sum(base_z_size) # w+2p, h+2p
        base_s_z = tf.sqrt(tf.reduce_prod(base_z_context_size))  # Canonical size
        base_scale_z = tf.div(tf.to_float(size_z), base_s_z) # s = sqrt(A**2/((w+2p)(h+2p))
        d_search = (size_x - size_z) / 2.0
        base_pad = tf.div(d_search, base_scale_z)
        base_s_x = base_s_z + 2 * base_pad
        base_scale_x = tf.div(tf.to_float(size_x), base_s_x)

        s_x = scale_factor * base_s_x
        frame_sz_1 = tf.to_float(frame_sz[0:2] - 1)
        topleft = tf.div(target_yx - get_center(s_x), frame_sz_1)
        bottomright = tf.div(target_yx + get_center(s_x), frame_sz_1)
        crop_box = tf.concat([topleft, bottomright], axis=0)
        scale_x = base_scale_x / scale_factor

        image_minus_avg = tf.expand_dims(image - avg_chan, 0)
        image_cropped = tf.image.crop_and_resize(image_minus_avg, crop_box[None],
                                                 box_ind=tf.zeros((1), tf.int32),
                                                 crop_size=[size_x, size_x])
        search_image = image_cropped + avg_chan
        search_image = search_image[0] # [1,H,W,3] --> [H,W,3]
        return search_image, scale_x, crop_box

    def build_template(self, search_image):
        size_z = self.z_image_size

        # Exemplar image lies at the center of the search image in the first frame
        exemplar_images = get_exemplar_images(search_image[None], [size_z, size_z])
        return exemplar_images[0]

    def decode_image(self, filename):
        image = tf.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3, dct_method="INTEGER_ACCURATE")
        image = tf.cast(image, tf.float32)
        return image

    def build_gauss_response(self, patch, mu_x, mu_y, sigma=7.0):
        height, width = tf.unstack(tf.shape(patch))[:2]
        pos_x, pos_y = tf.meshgrid(tf.range(width), tf.range(height))
        pos_x = tf.cast(pos_x, tf.float32)
        pos_y = tf.cast(pos_y, tf.float32)
        psf = tf.exp(-((pos_x-mu_x)**2/(2*sigma**2) + (pos_y-mu_y)**2/(sigma**2))) # not multiple by 2
        return psf[...,None] # [H,W,1]

class ImageNet(object):
    # Data augmentation code are come from
    # https://github.com/tensorflow/models/blob/master/official/resnet/imagenet_preprocessing.py
    
    def __init__(self, num_threads=8):
        self.num_threads = num_threads

        _R_MEAN = 123.68
        _G_MEAN = 116.78
        _B_MEAN = 103.94
        self.RGB_MEAN = [_R_MEAN, _G_MEAN, _B_MEAN]
        self.NUM_CHANNELS = 3
        self.DEFAULT_IMAGE_SIZE = 224
        self.NUM_CLASSES = 1001 # background(0) + objects(1,...,1000)
        self._RESIZE_MIN = 256
        
    def get_dataset(self, root_dir, phase='train', batch_size=16, is_training=True, one_hot=True, shuffle=True, subtract_mean=True, num_epoch=None, seed=None):
        ann_dir = os.path.join(root_dir, 'annotations', phase)
        data_dir = os.path.join(root_dir, 'ILSVRC2015/Data/CLS-LOC', phase) + '/' # data_dir must end with '/'
        
        ann_files = [x.path for x in os.scandir(ann_dir) if x.name.endswith('npz')]

        filenames = []
        imsizes = []
        class_inds = []
        bboxes = []

        for afile in ann_files:
            adata = np.load(afile)
            filenames.append(adata['filename'])
            imsizes.append(adata['imsize'])
            class_inds.append(adata['class_id'])
            # convert from unnormalized [x1,y1,x2,y2] to normalized [y1,x1,y2,x2]
            _bboxes = adata['bbox'].astype(np.float32)
            _imsizes = adata['imsize'].astype(np.float32) # H,W,C
            _new_bboxes = np.zeros_like(_bboxes)
            _new_bboxes[:,0] = _bboxes[:,1] / _imsizes[:,0] # y1
            _new_bboxes[:,1] = _bboxes[:,0] / _imsizes[:,1] # x1
            _new_bboxes[:,2] = _bboxes[:,3] / _imsizes[:,0] # y2
            _new_bboxes[:,2] = _bboxes[:,2] / _imsizes[:,1] # x2
            bboxes.append(_new_bboxes)

        filenames = np.concatenate(filenames, axis=0)
        imsizes = np.concatenate(imsizes, axis=0)
        class_inds = np.concatenate(class_inds, axis=0)
        bboxes = np.concatenate(bboxes, axis=0)

        self.num_examples = len(filenames)
        self.data_dir = tf.convert_to_tensor(data_dir)
        self.filenames = tf.convert_to_tensor(filenames)
        self.imsizes = tf.convert_to_tensor(imsizes)
        self.bboxes = tf.convert_to_tensor(bboxes)
        self.class_inds = tf.convert_to_tensor(class_inds)
        
        print('[{}] #Examples={}'.format(phase, self.num_examples))
        
        dataset = tf.data.Dataset.range(self.num_examples)
        if shuffle:
            dataset = dataset.shuffle(self.num_examples, seed=seed)
        dataset = dataset.repeat(count=num_epoch)
        dataset = dataset.map(
                    lambda x: self.parser(x, is_training, one_hot, subtract_mean), num_parallel_calls=self.num_threads)
        dataset = dataset.batch(batch_size)
        return dataset
    
    def parser(self, tgt_id, is_training, one_hot, subtract_mean):
        filename = self.data_dir + self.filenames[tgt_id]
        bbox = self.bboxes[tgt_id][None,None] # [batch,num_boxes,coords]
        class_label = self.class_inds[tgt_id]
        if one_hot:
            print(class_label)
            class_label = tf.one_hot(class_label, self.NUM_CLASSES)
            print(class_label)
        image_buffer = tf.read_file(filename)
        
        image = self.preprocess(image_buffer, bbox, 
                                output_height=self.DEFAULT_IMAGE_SIZE,
                                output_width=self.DEFAULT_IMAGE_SIZE,
                                num_channels=self.NUM_CHANNELS,
                                is_training=is_training
                               )
        if subtract_mean:
            image = self._mean_image_subtraction(image, self.RGB_MEAN, self.NUM_CHANNELS)
        return image, class_label
        
    def preprocess(self, image_buffer, bbox, output_height, output_width, num_channels, is_training=False):
        if is_training:
            # For training, we want to randomize some of the distortions.
            image = self._decode_crop_and_flip(image_buffer, bbox, num_channels)
            image = self._resize_image(image, output_height, output_width)
        else:
            # For validation, we want to decode, resize, then just crop the middle.
            image = tf.image.decode_jpeg(image_buffer, channels=num_channels)
            image = self._aspect_preserving_resize(image, self._RESIZE_MIN)
            image = self._central_crop(image, output_height, output_width)
            
        image.set_shape([output_height, output_width, num_channels])
        
        return image
    
    def _decode_crop_and_flip(self, image_buffer, bbox, num_channels):
        # A large fraction of image datasets contain a human-annotated bounding box
        # delineating the region of the image containing the object of interest.  We
        # choose to create a new bounding box for the object which is a randomly
        # distorted version of the human-annotated bounding box that obeys an
        # allowed range of aspect ratios, sizes and overlap with the human-annotated
        # bounding box. If no box is supplied, then we assume the bounding box is
        # the entire image.
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
          tf.image.extract_jpeg_shape(image_buffer),
          bounding_boxes=bbox,
          min_object_covered=0.1,
          aspect_ratio_range=[0.75, 1.33],
          area_range=[0.05, 1.0],
          max_attempts=100,
          use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, _ = sample_distorted_bounding_box

        # Reassemble the bounding box in the format the crop op requires.
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        crop_window = tf.stack([offset_y, offset_x, target_height, target_width])

        # Use the fused decode and crop op here, which is faster than each in series.
        cropped = tf.image.decode_and_crop_jpeg(
          image_buffer, crop_window, channels=num_channels)

        # Flip to add a little more random distortion in.
        cropped = tf.image.random_flip_left_right(cropped)
        return cropped

    def _central_crop(self, image, crop_height, crop_width):
        shape = tf.shape(image)
        height, width = shape[0], shape[1]

        amount_to_be_cropped_h = (height - crop_height)
        crop_top = amount_to_be_cropped_h // 2
        amount_to_be_cropped_w = (width - crop_width)
        crop_left = amount_to_be_cropped_w // 2
        return tf.slice(
              image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])
        
    def _smallest_size_at_least(self, height, width, resize_min):
        resize_min = tf.cast(resize_min, tf.float32)
        
        # Convert to floats to make subsequent calculations go smoothly.
        height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)
        
        smaller_dim = tf.minimum(height, width)
        scale_ratio = resize_min / smaller_dim

        # Convert back to ints to make heights and widths that TF ops will accept.
        new_height = tf.cast(height * scale_ratio, tf.int32)
        new_width = tf.cast(width * scale_ratio, tf.int32)

        return new_height, new_width        
    
    def _resize_image(self, image, height, width):
        return tf.image.resize_images(
            image, [height, width], method=tf.image.ResizeMethod.BILINEAR,
            align_corners=False)
    
    def _aspect_preserving_resize(self, image, resize_min):
        shape = tf.shape(image)
        height, width = shape[0], shape[1]
        new_height, new_width = self._smallest_size_at_least(height, width, resize_min)
        return self._resize_image(image, new_height, new_width)
    
    def _mean_image_subtraction(self, image, means, num_channels):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        if len(means) != num_channels:
            raise ValueError('len(means) must match the number of channels')

        # We have a 1-D tensor of means; convert to 3-D.
        means = tf.expand_dims(tf.expand_dims(means, 0), 0)

        return image - means
