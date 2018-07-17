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
from imageio import imread, imsave

LOCAL_PATH = './'
if LOCAL_PATH not in sys.path:
    sys.path.append(LOCAL_PATH)

# from datasets import CFDatasets, VOTDataset
from utils.io_utils import read_text
from utils.misc import *
import utils.tfvisualizer as tv
from models import *
from cf_utils import *
from inference import inference_wrapper, inference_cfcf
from inference.tracker import Tracker

def main(config):
    tf.reset_default_graph()
    log_dir = config.log_dir
    if config.net_type == 'siamese':
        model = inference_wrapper.InferenceWrapper(config)
    elif config.net_type == 'cfcf':
        model = inference_cfcf.InferenceCFCF(config)
    else:
        raise ValueError('Unknown net_type: ', config.net_type)

    var_list = model.build_graph_from_config()

    if config.clear_logs and tf.gfile.Exists(log_dir):
        print('Clear all files in {}'.format(log_dir))
        try:
            tf.gfile.DeleteRecursively(log_dir) 
        except:
            print('Fail to delete {}. You probably have to kill tensorboard process.'.format(log_dir))

    seq_names = read_text(config.seq_text, dtype=np.str)
    video_dirs = [os.path.join(config.root_dir, x) for x in seq_names]

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True # almost the same as tf.InteractiveSession
    sess = tf.Session(config=tfconfig)
    sess.run(tf.global_variables_initializer())
    # restore_fn(sess)

    if osp.isdir(config.model):
        checkpoint = tf.train.latest_checkpoint(config.model)
    else:
        checkpoint = config.model

    saver = tf.train.Saver(var_list)
    saver.restore(sess, checkpoint)

    tracker = Tracker(model, config=config)

    for video_dir in video_dirs:
        if not os.path.isdir(video_dir):
            continue

        video_name = os.path.basename(video_dir)
        video_log_dir = os.path.join(log_dir, video_name)
        mkdir_p(video_log_dir)

        filenames = sort_nicely(glob.glob(video_dir + '/img/*.jpg'))
        first_line = open(video_dir + '/groundtruth_rect.txt').readline()
        bb = [int(v) for v in first_line.strip().split(',')]
        init_bb = Rectangle(bb[0] - 1, bb[1] - 1, bb[2], bb[3])  # 0-index in python

        trajectory = tracker.track(sess, init_bb, filenames, video_log_dir)
        with open(osp.join(video_log_dir, 'track_rect.txt'), 'w') as f:
            for region in trajectory:
                rect_str = '{},{},{},{}\n'.format(region.x + 1, region.y + 1,
                                                region.width, region.height)
                f.write(rect_str)

if __name__ == '__main__':

    from utils.argparse_utils import *
    parser = get_parser()

    general_arg = add_argument_group('General', parser)
    general_arg.add_argument('--num_threads', type=int, default=8,
                            help='the number of threads (for dataset)')

    train_arg = add_argument_group('Train', parser)
    
    ## Siamese FC setting
    train_arg.add_argument('--net_type', type=str, default='siamese',
                            help='Network type: cfcf|siamese|cfnet')
    train_arg.add_argument('--model', type=str, default='/cvlabdata1/home/ono/datasets/CFCF/pretrained/ckpt/alexnet/',
                            help='where to save')
    # train_arg.add_argument('--model', type=str, default='/cvlabdata2/home/ono/results/practice/CFNet/180710-siamesefc-basic/bs-8/momentum-lr-1e-3/wd-0',
    #                         help='where to save')
    # train_arg.add_argument('--model', type=str, default='./learned_models/siamese/180710',
    #                         help='where to save')
    train_arg.add_argument('--z_image_size', type=int, default=127,
                            help='')    
    train_arg.add_argument('--backbone', type=str, default='alexnet',
                            help='backbone CNN (alexnet|vgg16|resnet50|mobilenet)')
    train_arg.add_argument('--log_dir', type=str, default='logs_track/src_siamese',
                            help='output directry')

    ## CFCF setting
    # train_arg.add_argument('--net_type', type=str, default='cfcf',
    #                         help='Network type: cfcf|siamese|cfnet')
    # # train_arg.add_argument('--model', type=str, default='/cvlabdata2/home/ono/results/practice/CFNet/180711-cf-basic/bs-8/adam-lr-1e-4/wd-0/balancedl2/vgg16/vgg_16/conv4/conv4_3/reg-0.01',
    # #                         help='where to save')
    # train_arg.add_argument('--model', type=str, default='./learned_models/cfcf/180711',
    #                         help='where to save')
    # train_arg.add_argument('--z_image_size', type=int, default=255,
    #                         help='')
    # train_arg.add_argument('--backbone', type=str, default='vgg16',
    #                         help='backbone CNN (alexnet|vgg16|resnet50|mobilenet)')
    # train_arg.add_argument('--feat_layer', type=str, default='vgg_16/conv4/conv4_3',
    #                         help='feature layer')
    # train_arg.add_argument('--reglambda', type=float, default=0.01,
    #                         help='lambda for regularization')
    # train_arg.add_argument('--log_dir', type=str, default='logs_track/cfcf_noupd',
    #                         help='output directry')
    
    ## Others
    train_arg.add_argument('--clear_logs', action='store_const',
                            const=True, default=False,
                            help='clear logs if it exists')

    dataset_arg = add_argument_group('Dataset', parser)
    dataset_arg.add_argument('--root_dir', type=str, default='/cvlabdata2/home/ono/Datasets/OTB2015',
                            help='dataset root directory')
    dataset_arg.add_argument('--seq_text', type=str, default='./datasets/otb2015.txt',
                            help='sequence text')
    dataset_arg.add_argument('--patch_size', type=int, default=101,
                            help='input patch size')

    track_arg = add_argument_group('Track', parser)
    track_arg.add_argument('--mmr_thresh', type=float, default=10.0,
                            help='threshold to update filter')
    track_arg.add_argument('--update_rate', type=float, default=1.0,
                            help='moving average rate: new_template = update_rate*old_template+(1-update_rate)*curr_template')
    # track_arg.add_argument('--update_rate', type=float, default=0.995,
    #                         help='moving average rate: new_template = update_rate*old_template+(1-update_rate)*curr_template')
    track_arg.add_argument('--num_scales', type=int, default=3,
                            help='the number of scales')
    track_arg.add_argument('--scale_step', type=float, default=1.0375,
                            help='scale step')
    track_arg.add_argument('--scale_penalty', type=float, default=0.9745,
                            help='scale penalty')

    cf_arg = add_argument_group('CFNet', parser)

    cf_arg.add_argument('--x_image_size', type=int, default=255,
                            help='')
    cf_arg.add_argument('--adjust_response_config_scale', type=float, default=0.001,
                            help='')
    cf_arg.add_argument('--adjust_response_config_train_bias', type=str2bool, default=True,
                            help='')
    cf_arg.add_argument('--upsample_method', type=str, default='bicubic',
                            help='')
    cf_arg.add_argument('--upsample_factor', type=int, default=16,
                            help='')
    cf_arg.add_argument('--window_influence', type=float, default=0.176,
                            help='')
    cf_arg.add_argument('--scale_damp', type=float, default=0.59,
                            help='')
    cf_arg.add_argument('--log_level', type=int, default=1,
                            help='')
    cf_arg.add_argument('--embed_stride', type=int, default=8,
                            help='')

    tmp_config, unparsed = get_config(parser)
    config = tmp_config


    main(config)