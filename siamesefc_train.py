from __future__ import print_function
import os
import sys
import numpy as np
import tensorflow as tf
import importlib
from datetime import datetime
import time
import glob
import cv2
from tqdm import tqdm
import pickle

LOCAL_PATH = './'
if LOCAL_PATH not in sys.path:
    sys.path.append(LOCAL_PATH)

from datasets import SiameseVIDDataset
from utils.tf_layer_utils import *
from utils.tf_train_utils import get_optimizer, get_piecewise_lr, get_activation_fn

from models import *
from cf_utils import *

MY_UTILS_PATH = '../dnnutils/'
if MY_UTILS_PATH not in sys.path:
    sys.path.append(MY_UTILS_PATH)
import tfvisualizer as tv
from utils.io_utils import read_text
SAVE_MODEL = True

def patch_eval_one_epoch(sess, ops, ev_params, name='valid'):
    num_examples = ev_params['num_examples']
    batch_size = ev_params['batch_size']
    summary_writer = ev_params['summary_writer']
    num_iter = num_examples // batch_size

    sess.run(ev_params['ev_init_op'])

    metrics = np.zeros(1, dtype=np.float32)

    for i in range(num_iter):
        feed_dict = {
            ops['is_training']: False,
            ops['handle']: ev_params['handle'],
        }

        fetch_dict = {
            'loss': ops['loss'],
        }

        outs = sess.run(fetch_dict, feed_dict=feed_dict)

        metrics += np.array([outs['loss']])

    #------ END OF ALL SAMPLES
    step = sess.run(ops['step'])
    metrics /= num_iter

    loss_ev, = metrics

    print('')
    print('[{}] iter={} Loss: {:g} '.format(
                name, step, 
                loss_ev))

    tag_list = ['loss_ev']
    summaries = []
    for _tag in tag_list:
        summaries.append( tf.Summary.Value(tag=_tag, simple_value=eval(_tag)) )
    summary_writer.add_summary(tf.Summary(value=summaries), global_step=step)


def build_network(config, next_batch, is_training):

    max_outputs = 5
    axis123 = [1,2,3]
    if config.backbone == 'alexnet':
        backbone = alexnet
        backbone_ckpt = os.path.join(config.ckpt_dir, 'alexnet')
    elif config.backbone == 'vgg16':
        backbone = vgg_16
        backbone_ckpt = os.path.join(config.ckpt_dir, 'vgg_16/vgg_16.ckpt')
    elif config.backbone == 'resnet50':
        backbone = resnet_v2_50
        backbone_ckpt = os.path.join(config.ckpt_dir, 'resnet_50/resnet_v2_50.ckpt')
    elif config.backbone == 'mobilenet':
        # it doesn't works on tensorflow v1.4
        depth_multiplier = 1.4
        backbone = mobilenet_v2
        backbone_ckpt = os.path.join(config.ckpt_dir, 'mobilenet_v2/mobilenet_v2_1.4_224.ckpt')
    else:
        raise ValueError('Unknown backbone: {}'.format(config.backbone))

    template_src, query_src, query_res = next_batch
    batch_size = tf.shape(template_src)[0]

    if config.ignore_pretrain:
        # Not load pretrained model
        template_img = template_src
        query_img = query_src
    else:
        template_img = template_src - IMAGENET_RGB_MEAN
        query_img = query_src - IMAGENET_RGB_MEAN

    # Get CNN response of query and template
    feats_X, endpoints_X = backbone(query_img, is_training=is_training, reuse=False)
    feats_Z, endpoints_Z = backbone(template_img, is_training=is_training, reuse=True)
    var_list = endpoints_X['var_list']
    # for k, v in endpoints_X.items():
    #     if isinstance(v, tf.Tensor):
    #         print(k, v.shape)

    # feats_X = endpoints_X[config.feat_layer]
    # feats_Z = endpoints_Z[config.feat_layer]
    print('FEAT-SIZE [Q] {}, [T] {}'.format(feats_X.get_shape().as_list(), feats_Z.get_shape().as_list()))
    with tf.variable_scope('detection'):
        def _translation_match(x, z):
            # x, z = [H,W,C]
            x = tf.expand_dims(x, 0) # [1=batch,H,W,C]
            z = tf.expand_dims(z, -1) # [h,w,C,1=out_channels]
            return tf.nn.conv2d(x,z, strides=[1, 1, 1, 1], padding='VALID', name='translation_match') # [1,H',W',1]
        outputs = tf.map_fn(
            lambda x: _translation_match(x[0], x[1]),
            (feats_X, feats_Z), dtype=feats_X.dtype) # [B,1,H,W,1]
        outputs = tf.squeeze(outputs,[1]) # [B,H,W,1]

        # response = spatial_softmax(outputs) * 2 - 1.0
        bias = tf.get_variable('biases', [1],
                             dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0, dtype=tf.float32),
                             trainable=False)
        var_list.append(bias)
        response = config.adjust_response_config_scale * outputs + bias # need scale in order to avoid exp-divergence
        tf.summary.histogram('bias', bias)
    tf.summary.histogram('response', response)

    res_height, res_width = response.get_shape().as_list()[1:3]
    query_res_down = tf.image.resize_images(query_res, (res_height, res_width), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True)

    is_pos = tf.to_float(tf.equal(query_res_down, 1))
    is_neg = tf.to_float(tf.equal(query_res_down, -1))
    num_pos = tf.reduce_sum(is_pos, axis=axis123)
    num_neg = tf.reduce_sum(is_neg, axis=axis123)
    logistic_loss = tf.log(1+tf.exp(-response * query_res_down))
    loss_pos = tf.reduce_sum(is_pos * logistic_loss, axis=axis123) / tf.maximum(num_pos, 1.0)
    loss_neg = tf.reduce_sum(is_neg * logistic_loss, axis=axis123) / tf.maximum(num_neg, 1.0)
    loss = tf.reduce_mean(loss_pos+loss_neg)

    if config.weight_decay > 0:
        print('Add weight decay loss with lambda={} (#trainable-params={})'.format(config.weight_decay, len(var_list)))
        wd_loss = tf.add_n([tf.nn.l2_loss(v) for v in var_list])
        tf.summary.scalar('logistic_loss', loss)
        tf.summary.scalar('weight_decay_loss', wd_loss)
        loss = loss + config.weight_decay * wd_loss

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('num_pixel', tf.reduce_mean(num_pos+num_neg))

    # Visualization
    c_red = tf.constant([1,0,0], dtype=tf.float32)
    c_green = tf.constant([0,1,0], dtype=tf.float32)
    c_blue = tf.constant([0,0,1], dtype=tf.float32)

    img_height, img_width = query_src.get_shape().as_list()[1:3]
    response_up_norm = tf.image.resize_images(normalize_01(response), (img_height, img_width)) # normalized by 0~1
    query_res_norm = (query_res + 1.0) * 0.5
    template_canvas = tf.image.resize_images(template_src, (img_height, img_width))
    predict_canvas = (1.0-response_up_norm) * query_src + response_up_norm * c_green * 255
    gt_canvas = (1.0-query_res_norm) * query_src + query_res_norm * c_green * 255
    canvas = tf.cast(tf.concat([template_canvas, predict_canvas, gt_canvas], axis=2), tf.uint8)
    canvas = tf.image.resize_images(canvas, (img_height//2, img_width*3//2))

    tf.summary.image('TEMP-PRED-GT', canvas, max_outputs=max_outputs)



    endpoints = {
        'query_src': query_src,
        'template_src': template_src,
        'query_res': query_res,
        'query_res_down': query_res_down,
        'feats_X': feats_X,
        'feats_Z': feats_Z,
        'num_pos': tf.reduce_mean(num_pos),
        'num_neg': tf.reduce_mean(num_neg),
        'loss': loss,
        'var_list': var_list,
        'backbone_name': backbone.name,
        'backbone_ckpt': backbone_ckpt,
    }


    return loss, endpoints


def main(config):
    tf.reset_default_graph() # for sure
    log_dir = config.log_dir
    learning_rate = config.lr
    va_batch_size = 1
    print('Setup dataset')
    tr_provider = SiameseVIDDataset(template_image_size=config.template_image_size, query_image_size=config.query_image_size, 
                        max_seq_length=config.max_length, num_threads=config.num_threads)
    va_provider = SiameseVIDDataset(template_image_size=config.template_image_size, query_image_size=config.query_image_size, 
                        max_seq_length=config.max_length, num_threads=config.num_threads)
    tr_dataset = tr_provider.get_dataset(config.vid_dir, phase='train', batch_size=config.batch_size, shuffle=True)
    va_dataset = va_provider.get_dataset(config.vid_dir, phase='val', batch_size=va_batch_size, shuffle=True, seed=1234)
    tr_num_examples = tr_provider.num_examples
    va_num_examples = min(va_provider.num_examples, 1000)
    print('#examples = {}, {}'.format(tr_num_examples, va_num_examples))

    handle = tf.placeholder(tf.string, shape=[])

    dataset_iter = tf.data.Iterator.from_string_handle(handle, tr_dataset.output_types, tr_dataset.output_shapes) # create mock of iterator
    next_batch = list(dataset_iter.get_next()) #tuple --> list to make it possible to modify each elements

    tr_iter = tr_dataset.make_one_shot_iterator() # infinite loop
    va_iter = va_dataset.make_initializable_iterator() # require initialization in every epoch

    is_training = tf.placeholder(tf.bool, name='is_training')
    global_step = tf.Variable(0, name='global_step', trainable=False)

    print('Build network')
    loss, endpoints = build_network(config, next_batch, is_training)

    if config.lr_decay:
        max_epoch = 50
        boundaries = list((np.arange(max_epoch, dtype=np.int32)+1) * 5000)
        lr_values = list(np.logspace(-2, -5, max_epoch))
        learning_rate = get_piecewise_lr(global_step, boundaries, lr_values, show_summary=True)
        print('Enable adaptive learning. LR will decrease {} when #iter={}'.format(lr_values, boundaries))        


    minimize_op = get_optimizer(config.optim_method, global_step, learning_rate, loss, endpoints['var_list'], show_var_and_grad=config.show_histogram)
    print('Done.')


    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True # almost the same as tf.InteractiveSession
    sess = tf.Session(config=tfconfig)

    summary = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    tr_handle = sess.run(tr_iter.string_handle())
    va_handle = sess.run(va_iter.string_handle())

    if config.clear_logs and tf.gfile.Exists(log_dir):
        print('Clear all files in {}'.format(log_dir))
        try:
            tf.gfile.DeleteRecursively(log_dir) 
        except:
            print('Fail to delete {}. You probably have to kill tensorboard process.'.format(log_dir))

    # load pretrained detector model
    if not config.ignore_pretrain:
        pretrained_model = endpoints['backbone_ckpt']
        if os.path.isdir(pretrained_model):
            checkpoint = tf.train.latest_checkpoint(pretrained_model)
        else:
            checkpoint = pretrained_model

        if checkpoint is not None:
            # use global_vars instead of var_list in order to get not only trainable but also batch-norm params
            if config.resume_bn:
                global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                pretrained_vars = [] # Resume only detector variables
                for var in global_vars:
                    if var.name.startswith(endpoints['backbone_name']): # not include Optimization/...
                        # print('[PT] {}'.format(var.name))
                        pretrained_vars.append(var)
                    else:
                        pass
                        # print('[SKIP] {}'.format(var.name))
            else:
                pretrained_vars = endpoints['var_list']

            if len(pretrained_vars) == 0:
                raise ValueError('Cannot find any variables to resume')
            print('Resume pretrained variables...'.format(len(pretrained_vars)))
            for i, var in enumerate(pretrained_vars):
                print('#{} {} [{}]'.format(i, var.name, var.shape))

            saver = tf.train.Saver(pretrained_vars)
            saver.restore(sess, checkpoint)
            saver = None
            print('Load pretrained model from {}'.format(checkpoint))
        else:
            raise ValueError('Cannot open checkpoint: {}'.format(checkpoint))
    else:
        print('Skip loading pretrained model...')

    best_saver = tf.train.Saver(max_to_keep=10, save_relative_paths=True)
    latest_saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)

    latest_checkpoint = tf.train.latest_checkpoint(log_dir)
    best_score_filename = os.path.join(log_dir, 'valid', 'best_score.txt')
    best_score = 0 # larger is better
    if latest_checkpoint is not None:
        from parse import parse
        print('Resume the previous model...')
        latest_saver.restore(sess, latest_checkpoint)
        curr_step = sess.run(global_step)
        if os.path.exists(best_score_filename):
            with open(best_score_filename, 'r') as f:
                dump_res = f.read()
            dump_res = parse('{step:d} {best_score:g}\n', dump_res)
            best_score = dump_res['best_score']
            print('Previous best score = {} @ #step={}'.format(best_score, curr_step))

    train_writer = tf.summary.FileWriter(
        os.path.join(log_dir, 'train'), graph=sess.graph
    )
    valid_writer = tf.summary.FileWriter(
        os.path.join(log_dir, 'valid'), graph=sess.graph
    )    

    if SAVE_MODEL:
        latest_saver.export_meta_graph(os.path.join(log_dir, "models.meta"))
    # Save config
    with open(os.path.join(log_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)    

    ops = {
        'is_training': is_training,
        'handle': handle,
        'step': global_step,
        'summary': summary,
        'minimize_op': minimize_op,
    }
    for k, v in endpoints.items():
        if isinstance(v, tf.Tensor):
            ops[k] = v

    #----------------------
    # Start Training
    #----------------------
    save_summary_interval = 1000
    save_model_interval = 50000
    valid_interval = 5000

    va_params = {
        'batch_size': va_batch_size,
        'num_examples': va_num_examples,
        'summary_writer': valid_writer,
        'handle': va_handle,
        'ev_init_op': va_iter.initializer,
    }

    def check_counter(counter, interval):
        return (interval > 0 and counter % interval == 0)

    start_itr = sess.run(ops['step'])

    for _ in range(start_itr, config.max_itr):

        feed_dict = {
            ops['is_training']: True,
            ops['handle']: tr_handle,
        }        

        step, _ = sess.run([ops['step'], ops['minimize_op']], feed_dict=feed_dict)

        if check_counter(step, save_summary_interval):
            feed_dict = {
                ops['is_training']: False,
                ops['handle']: tr_handle,
            }
            fetch_dict = {
                'loss': ops['loss'],
                'summary': ops['summary'],
            }
            start_time = time.time()
            outs = sess.run(fetch_dict, feed_dict=feed_dict)
            elapsed_time = time.time() - start_time
            train_writer.add_summary(outs['summary'], step) # save summary
            summaries = [tf.Summary.Value(tag='sec/step', simple_value=elapsed_time)]
            train_writer.add_summary(tf.Summary(value=summaries), global_step=step)
            train_writer.flush()

            print('[Train] {}step Loss: {:g} ({:.1f}sec)'.format(
                        step,
                        outs['loss'],
                        elapsed_time))
            if SAVE_MODEL and latest_saver is not None:
                latest_saver.save(sess, os.path.join(log_dir, 'models-latest'), global_step=step, write_meta_graph=False)


        if SAVE_MODEL and best_saver is not None and check_counter(step, save_model_interval):
            # print('#{}step Save latest model'.format(step))
            best_saver.save(sess, os.path.join(log_dir, 'models-best'), global_step=step, write_meta_graph=False)

        if check_counter(step, valid_interval):
            patch_eval_one_epoch(sess, ops, va_params)

if __name__ == '__main__':

    from utils.argparse_utils import *
    parser = get_parser()

    general_arg = add_argument_group('General', parser)
    general_arg.add_argument('--num_threads', type=int, default=8,
                            help='the number of threads (for dataset)')

    train_arg = add_argument_group('Train', parser)
    train_arg.add_argument('--log_dir', type=str, default='logs/siamese',
                            help='where to save')
    train_arg.add_argument('--clear_logs', action='store_const',
                            const=True, default=False,
                            help='clear logs if it exists')
    train_arg.add_argument('--show_histogram', action='store_const',
                            const=True, default=False,
                            help='show variable / gradient histograms on tensorboard (consume a lot of disk space)')
    train_arg.add_argument('--max_itr', type=int, default=50000,
                            help='max epoch')
    train_arg.add_argument('--batch_size', type=int, default=8,
                            help='batch size')
    train_arg.add_argument('--ignore_pretrain', type=str2bool, default=True,
                            help='ignore loading pretrained model')
    train_arg.add_argument('--optim_method', type=str, default='adam',
                            help='adam, momentum, ftrl, rmsprop')
    train_arg.add_argument('--lr', type=float, default=1e-3,
                            help='learning rate')
    train_arg.add_argument('--lr_decay', type=str2bool, default=False,
                            help='apply lr decay')
    train_arg.add_argument('--weight_decay', type=float, default=0,
                            help='weight decay (not apply if 0, otherwise we suggest 5e-4)')

    dataset_arg = add_argument_group('Dataset', parser)
    dataset_arg.add_argument('--vid_dir', type=str, default='/cvlabdata1/home/ono/datasets/VID/ILSVRC2015',
                            help='validation TFRecords directory')
    dataset_arg.add_argument('--template_image_size', type=int, default=127,
                            help='template_image_size')
    dataset_arg.add_argument('--query_image_size', type=int, default=255,
                            help='query_image_size')
    dataset_arg.add_argument('--max_length', type=int, default=500,
                            help='max_length')


    net_arg = add_argument_group('Network', parser)
    net_arg.add_argument('--ckpt_dir', type=str, default='/cvlabdata1/home/ono/datasets/CFCF/pretrained/ckpt/',
                            help='backbone CNN')
    net_arg.add_argument('--backbone', type=str, default='alexnet',
                            help='backbone CNN (alexnet|vgg16|resnet50|mobilenet)')
    net_arg.add_argument('--adjust_response_config_scale', type=float, default=0.001,
                            help='')
    # net_arg.add_argument('--adjust_response_config_scale', type=float, default=1.000,
    #                         help='')
    # vgg16
    #   - [56, 56, 256] --> vgg_16/conv3/conv3_3
    #   - [28, 28, 512] --> vgg_16/conv4/conv4_3
    #   - [14, 14, 512] --> vgg_16/conv5/conv5_3
    # resnet50
    #   - [56, 56, 256] --> resnet_v2_50/block1/unit_2/bottleneck_v2
    #   - [28, 28, 256] --> resnet_v2_50/block1
    #   - [14, 14, 512] --> resnet_v2_50/block2
    #   - [7, 7, 1024] --> resnet_v2_50/block3
    #   - [7, 7, 2048] --> resnet_v2_50/block4
    # mobilenet
    #   - [56, 56, 24] --> layer_4/output
    #   - [28, 28, 32] --> layer_5/output
    #   - [14, 14, 64] --> layer_8/output
    #   - [14, 14, 96] --> layer_12/output
    #   - [7, 7, 160] --> layer_15/output
    #   - [7, 7, 320] --> layer_18/output
    #   - [7, 7, 1280] --> layer_19
    net_arg.add_argument('--feat_layer', type=str, default='vgg_16/conv4/conv4_3',
                            help='feature maps layer in backbone')

    config, unparsed = get_config(parser)

    if len(unparsed) > 0:
        raise ValueError('Miss finding argument: unparsed={}\n'.format(unparsed))

    main(config)