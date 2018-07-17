from __future__ import print_function
import os
import sys
import math
from imageio import imread
import numpy as np
import pickle
import cv2
from tqdm import tqdm
import xml.etree.ElementTree as ET
import tensorflow as tf

from cf_utils import get_gauss_filter_weight, get_area
from utils.io_utils import save_pickle


class TrackRect(object):
    def __init__(self, rotated_rect, bg_ratio=2):
        if rotated_rect.size == 4:
            # x1,y1,w,h
            x1,y1, w, h = rotated_rect
            x2 = x1 + w
            y2 = y1 + h
            rotated_rect = np.array([x1,y1,x2,y1,x2,y2,x1,y2])
        cx = rotated_rect[0::2].mean()
        cy = rotated_rect[1::2].mean()
        x1 = rotated_rect[0::2].min()
        x2 = rotated_rect[0::2].max()
        y1 = rotated_rect[1::2].min()
        y2 = rotated_rect[1::2].max()
        p1 = rotated_rect[0:2]
        p2 = rotated_rect[2:4]
        p3 = rotated_rect[4:6]
        A1 = np.linalg.norm(p1-p2) * np.linalg.norm(p2-p3)
        A2 = (x2-x1) * (y2-y1)
        s = math.sqrt(A1/A2)
        fg_w = s * (x2-x1) + 1
        fg_h = s * (y2-y1) + 1
        bg_w = fg_w * bg_ratio
        bg_h = fg_h * bg_ratio
        hfg_w = fg_w / 2
        hfg_h = fg_h / 2
        hbg_w = bg_w / 2
        hbg_h = bg_h / 2
        self.rotated_rect = rotated_rect
        self.out_rect = np.round(np.array([x1,y1,x2,y2])).astype(np.int32)
        self.fg_rect = np.round(np.array([cx-hfg_w,cy-hfg_h, cx+hfg_w,cy+hfg_h])).astype(np.int32)
        self.bg_rect = np.round(np.array([cx-hbg_w,cy-hbg_h, cx+hbg_w,cy+hbg_h])).astype(np.int32)
        self.rotated_area = A1
        self.out_area = A2
        self.fg_area = (self.fg_rect[2]-self.fg_rect[0]) * (self.fg_rect[3]-self.fg_rect[1])
        self.bg_area = (self.bg_rect[2]-self.bg_rect[0]) * (self.bg_rect[3]-self.bg_rect[1])
        self.pos = np.round(np.array([cx, cy])).astype(np.int32)
        self.fg_width = self.fg_rect[2]-self.fg_rect[0]
        self.fg_height = self.fg_rect[3]-self.fg_rect[1]
        self.bg_width = self.bg_rect[2]-self.bg_rect[0]
        self.bg_height = self.bg_rect[3]-self.bg_rect[1]

def load_vid_annotation(xml_file):
    xmltree = ET.parse(xml_file)
    folder = xmltree.find('folder').text
    filename = xmltree.find('filename').text
    im_width = int(xmltree.find('size').find('width').text)
    im_height = int(xmltree.find('size').find('height').text)
    
    objects = xmltree.findall("object")
    if objects is None:
        print('{} doesnot have any objects'.format(xml_file) )
        return None
    bboxes = []
    for object_iter in objects:
        bbox = object_iter.find('bndbox')
        xmax = int(bbox.find('xmax').text)
        xmin = int(bbox.find('xmin').text)
        ymax = int(bbox.find('ymax').text)
        ymin = int(bbox.find('ymin').text)        
        width = xmax-xmin
        height = ymax-ymin
        bboxes.append(np.array([xmin,ymin,width,height]))
        # bboxes.append([xmin,ymin,xmax,ymax])
    output = {}
    output['filename'] = os.path.join(folder, filename+'.JPEG')
    output['width'] = im_width
    output['height'] = im_height
    output['bboxes'] = bboxes
    
    return output


def dump_vid_annotations(config):
    root_dir = config.vid_dir
    if config.phase == 'train':
        data_dir = os.path.join(root_dir, 'Data/VID/train')
        ann_dirs = sorted([x.path for x in os.scandir(os.path.join(root_dir, 'Annotations/VID/train')) if x.is_dir()])
        # data_dirs = sorted([x.path for x in os.scandir(os.path.join(root_dir, 'Data/VID/train')) if x.is_dir()])
        # ann_dirs = [x.replace('Data', 'Annotations') for x in data_dirs]
    elif config.phase == 'val':
        data_dir = os.path.join(root_dir, 'Data/VID/val')
        ann_dirs = [os.path.join(root_dir, 'Annotations/VID/val')]
        # data_dirs = [os.path.join(root_dir, 'Data/VID/val')]
        # ann_dirs = [x.replace('Data', 'Annotations') for x in data_dirs]
    else:
        raise ValueError('Unknown phase: {}'.format(config.phase))

    num_set = len(ann_dirs)

    sub_dirs = []
    for subdir in ann_dirs:
        _sub_dirs = sorted([x.name for x in os.scandir(subdir) if x.is_dir()])
        sub_dirs.append(_sub_dirs)
        print('{} has {} sequences'.format(subdir, len(_sub_dirs)))

    for t, ann_dir in enumerate(ann_dirs):
        print('#{} {} subdirs={}'.format(t, ann_dir, len(sub_dirs[t])))
        out_dir = os.path.join(config.out_dir, 'annotation', config.dataset, config.phase, os.path.basename(ann_dir))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        for i, sub_dir in enumerate(sub_dirs[t]):
            ann_paths = sorted([x.path for x in os.scandir(os.path.join(ann_dir, sub_dir)) if x.name.endswith('xml')])
            num_frames = len(ann_paths)
            annotations = [load_vid_annotation(x) for x in ann_paths]

            filenames = []
            imsizes = []
            bboxes = []

            # with tf.python_io.TFRecordWriter(out_path) as writer:
            for ann in annotations:
                # ignore invali label
                if ann is None:
                    continue
                if len(ann['bboxes']) == 0:
                    continue
                box_w = ann['bboxes'][0][2]
                box_h = ann['bboxes'][0][3]
                if box_w <= 0 or box_h <= 0:
                    continue
                trect = TrackRect(ann['bboxes'][0])
                fg_rect = trect.fg_rect
                filename = ann['filename']
                width = ann['width']
                height = ann['height']
                if get_area(fg_rect) / float(width*height) > config.box_ratio: # ignore too big bbox
                    continue
                filenames.append(filename)
                imsizes.append((height, width))
                bboxes.append(fg_rect)
                # example = tf.train.Example(features=tf.train.Features(feature={
                #     'bbox': tf.train.Feature(int64_list=tf.train.Int64List(value=list(fg_rect))),
                #     'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                #     'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                #     'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode(encoding='utf-8')])),
                # }))
                # writer.write(example.SerializeToString())

            assert len(filenames) == len(imsizes) == len(bboxes)
            if len(filenames) < 50:
                print('Skip due to too few frames ({}) in {}'.format(len(filenames), sub_dir))
                continue
            out_path = os.path.join(out_dir, '{}.npz'.format(sub_dir))
            np.savez(out_path, filename=filenames, imsize=imsizes, bbox=bboxes)

def load_synsets(filename, add_background=True):
    label_names = np.loadtxt(filename, str, delimiter='\t')
    synsets = {}

    if add_background:
        synsets['n00000000'] = (0, 'background') # unused on classification tasks
        class_id = 1
    else:
        class_id = 0
    
    for ln in label_names:
        idx = ln.find(' ')
        label = ln[:idx]
        name = ln[idx+1:]
        synsets[label] = (class_id, name)
        class_id += 1

    return synsets

def load_imagenet_annotation(xml_file, synsets):
    xmltree = ET.parse(xml_file)
    output = {}
    class_label = xmltree.find('folder').text
    filename = xmltree.find('filename').text
    im_width = int(xmltree.find('size').find('width').text)
    im_height = int(xmltree.find('size').find('height').text)
    im_depth = int(xmltree.find('size').find('depth').text)
    
    objects = xmltree.findall('object')
    if objects is None:
        print('{} doesnot have any objects'.format(xml_file) )
        return None    
    bboxes = []
    for object_iter in objects:
        name = object_iter.find('name').text
        if class_label != 'val' and name != class_label:
            continue
        bbox = object_iter.find('bndbox')
        xmax = int(bbox.find('xmax').text)
        xmin = int(bbox.find('xmin').text)
        ymax = int(bbox.find('ymax').text)
        ymin = int(bbox.find('ymin').text)
        width = xmax-xmin
        height = ymax-ymin
        if width > 0 and height > 0:
            bboxes.append(np.array([xmin,ymin,width,height], dtype=np.int32))

    if len(bboxes) == 0:
        return None

    if class_label == 'val':
        class_label = name

    class_id, class_name = synsets[class_label]
    output['class_label'] = class_label
    output['class_id'] = class_id
    output['class_name'] = class_name
    output['filename'] = '{}/{}.JPEG'.format(class_label, filename)
    output['width'] = im_width
    output['height'] = im_height
    output['depth'] = im_depth
    output['bboxes'] = bboxes
    return output

def dump_imagenet_annotations(config):
    root_dir = config.imagenet_dir
    synsets = load_synsets(os.path.join(root_dir, 'synset_words.txt'))

    if config.phase == 'train':
        ann_dirs = sorted([x.path for x in os.scandir(os.path.join(root_dir, 'ILSVRC2015/Annotations/CLS-LOC/train'))])
    elif config.phase == 'val':
        ann_dirs = [os.path.join(root_dir, 'ILSVRC2015/Annotations/CLS-LOC/val')]

    out_dir = os.path.join(config.out_dir, config.phase)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    skip_image_list = [
        'n01784675/n01784675_9426.JPEG',
        'n03781244/n03781244_9319.JPEG',
    ]

    for adir in ann_dirs:
        xmlpaths = sorted([x.path for x in os.scandir(adir) if x.name.endswith('xml')])
        annotations = [load_imagenet_annotation(x, synsets) for x in xmlpaths]   
        xml_file = xmlpaths[0]     

        filenames = []
        imsizes = []
        bboxes = []
        class_labels = []
        class_ids = []
        class_names = []

        for ann in annotations:
            # ignore invali label
            if ann is None:
                continue
            if ann['filename'] in skip_image_list:
                print('Skip because {} cannot read !'.format(ann['filename']))
                continue
            trect = TrackRect(ann['bboxes'][0])
            fg_rect = trect.fg_rect # x1,y1,x2,y2
            bboxes.append(fg_rect)
            filenames.append(ann['filename'])
            imsizes.append((ann['height'], ann['width'], ann['depth']))
            class_labels.append(ann['class_label'])
            class_ids.append(ann['class_id'])
            class_names.append(ann['class_name'])
        assert len(filenames) == len(imsizes) == len(bboxes) == len(class_labels) == len(class_ids) == len(class_names)

        out_path = os.path.join(out_dir, '{}.npz'.format(os.path.basename(adir)))
        np.savez(out_path, filename=filenames, imsize=imsizes, bbox=bboxes,
                class_label=class_labels, class_id=class_ids, class_name=class_names)
        print('Save {} ({} annotations in {} samples)'.format(out_path, len(filenames), len(annotations)))

def load_imagenet_check(config):
    # Check each image in imagenet can be readable
    root_dir = config.imagenet_dir
    phase = config.phase
    ann_dir = os.path.join(root_dir, 'annotations', phase)
    data_dir = os.path.join(root_dir, 'ILSVRC2015/Data/CLS-LOC', phase) + '/' # data_dir must end with '/'
    
    ann_files = [x.path for x in os.scandir(ann_dir) if x.name.endswith('npz')]

    fail_list =[]

    for t, afile in enumerate(ann_files):
        if t < config.start or config.end < t:
            continue
        adata = np.load(afile)
        filenames = adata['filename']
        print('{}/{} {} {}files #fails={}'.format(t+1, len(ann_files), afile, len(filenames), len(fail_list)))
        for n, fname in enumerate(filenames):
            filepath = data_dir + fname
            try:
                image = imread(filepath)
                if image.shape[0] * image.shape[1] == 0:
                    print('INVALID IMAGE {}, {}'.format(filepath, image.shape))
                    fail_list.append(filepath)
            except:
                print('FAIL TO READ IMAGE {}'.format(filepath))
                fail_list.append(filepath) 

    for file in fail_list:
        print(file)

if __name__ == '__main__':

    from utils.argparse_utils import *
    parser = get_parser()

    parser.add_argument('--dataset', type=str, default='imagenet',
                        help='dataset name (vid|imagenet)')
    parser.add_argument('--phase', type=str, default='train',
                        help='phase (train|val)')
    parser.add_argument('--vid_dir', type=str, default='/cvlabdata1/home/ono/datasets/VID/ILSVRC2015',
                        help='path to VOT dataset')
    parser.add_argument('--imagenet_dir', type=str, default='/cvlabdata1/home/ono/datasets/imagenet',
                        help='path to VOT dataset')
    parser.add_argument('--out_dir', type=str, default='/cvlabdata1/home/ono/datasets/imagenet/annotations',
                        help='output directory')
    parser.add_argument('--box_ratio', type=float, default=0.2,
                        help='box_area / img_area')
    parser.add_argument('--start', type=int, default=0,
                        help='box_area / img_area')
    parser.add_argument('--end', type=float, default=1000,
                        help='box_area / img_area')

    config, unparsed = get_config(parser)

    if len(unparsed) > 0:
        raise ValueError('Warning: miss identify argument ?? unparsed={}\n'.format(unparsed))


    if config.dataset == 'vid':
        dump_vid_annotations(config)
    elif config.dataset == 'imagenet':
        dump_imagenet_annotations(config)
    else:
        load_imagenet_check(config)
