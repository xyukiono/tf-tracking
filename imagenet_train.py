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

from datasets import ImageNet
from utils.tf_layer_utils import *
from utils.tf_train_utils import get_optimizer, get_piecewise_lr, get_activation_fn
import utils.tfvisualizer as tv
from utils.io_utils import read_text

MODEL_PATH = './models'
if MODEL_PATH not in sys.path:
    sys.path.append(MODEL_PATH)

SAVE_MODEL = True

def eval_one_epoch(sess, ops, ev_params, name='valid'):
    num_examples = ev_params['num_examples']
    batch_size = ev_params['batch_size']
    summary_writer = ev_params['summary_writer']
    num_iter = num_examples // batch_size

    sess.run(ev_params['ev_init_op'])

    metrics = np.zeros(3, dtype=np.float32)

    for i in range(num_iter):
        feed_dict = {
            ops['is_training']: False,
            ops['handle']: ev_params['handle'],
        }

        fetch_dict = {
            'loss': ops['loss'],
            'top1': ops['top1'],
            'top5': ops['top5'],
        }
        try:
            outs = sess.run(fetch_dict, feed_dict=feed_dict)
            metrics += np.array([outs['loss'], outs['top1'], outs['top5']])
        except:
            print('[TEST] Error happens but keep training...')


    #------ END OF ALL SAMPLES
    step = sess.run(ops['step'])
    metrics /= num_iter

    loss_ev, top1_ev, top5_ev = metrics

    print('')
    print('[{}] iter={} Loss: {:g}, Top1: {:g} Top5: {:g}'.format(
                name, step, 
                loss_ev, top1_ev, top5_ev))

    tag_list = ['loss_ev', 'top1_ev', 'top5_ev']
    summaries = []
    for _tag in tag_list:
        summaries.append( tf.Summary.Value(tag=_tag, simple_value=eval(_tag)) )
    summary_writer.add_summary(tf.Summary(value=summaries), global_step=step)


def build_network(config, next_batch, is_training, num_classes=1001):

    max_outputs = 5
    axis123 = [1,2,3]
    
    images, labels = next_batch

    MODEL = importlib.import_module(config.model)

    if config.model == 'custom_alexnet':
        logits, endpoints = MODEL.get_model(images, is_training, num_classes=num_classes)
    elif config.model == 'custom_vgg':
        logits, endpoints = MODEL.get_model(images, is_training, num_classes=num_classes,
                            num_pooling=4)
    var_list = endpoints['var_list']

    # Loss
    xentropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(xentropy)

    if config.weight_decay > 0:
            print('Add weight decay loss with lambda={}'.format(config.weight_decay))
            wd_loss = tf.add_n([tf.nn.l2_loss(v) for v in var_list])
            tf.summary.scalar('xentropy_loss', loss)
            tf.summary.scalar('weight decay loss', wd_loss)
            loss = loss + config.weight_decay * wd_loss
    tf.summary.scalar('loss', loss)

    # Eval
    pred_labels = tf.argmax(logits, axis=1)
    gt_labels = tf.argmax(labels, axis=1)
    top1 = tf.reduce_mean(tf.cast(tf.equal(pred_labels, gt_labels), tf.float32))
    top5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=logits, targets=gt_labels, k=5), tf.float32))
    tf.summary.scalar('top1', top1)
    tf.summary.scalar('top5', top5)

    if True:
        # visualize conv1
        conv1_weights = var_list[0]
        in_ch, out_ch = conv1_weights.get_shape().as_list()[2:4]
        assert in_ch == 3
        conv1_weights = tf.transpose(conv1_weights, [3,0,1,2]) # [H,W,C,B]-->[B,H,W,C]
        conv_tiles = tv.convert_tile_image(conv1_weights)
        conv_tiles = tf.clip_by_value(conv_tiles, -1.0, 1.0)
        tf.summary.image('conv1_weights', conv_tiles)

    endpoints['loss'] = loss
    endpoints['top1'] = top1
    endpoints['top5'] = top5
    endpoints['pred_labels'] = pred_labels
    endpoints['gt_labels'] = gt_labels


    return loss, endpoints    

def main(config):
    tf.reset_default_graph() # for sure

    log_dir = config.log_dir
    learning_rate = config.lr
    va_batch_size = 10

    print('Setup dataset')

    tr_provider = ImageNet(num_threads=config.num_threads)
    va_provider = ImageNet(num_threads=config.num_threads)
    tr_dataset = tr_provider.get_dataset(config.imagenet_dir, phase='train', batch_size=config.batch_size, 
                                is_training=True, shuffle=True)
    va_dataset = va_provider.get_dataset(config.imagenet_dir, phase='val', batch_size=va_batch_size, 
                                is_training=False, shuffle=True, seed=1234)
    tr_num_examples = tr_provider.num_examples
    va_num_examples = min(va_provider.num_examples, 10000)
    print('#examples = {}, {}'.format(tr_num_examples, va_num_examples))

    handle = tf.placeholder(tf.string, shape=[])

    dataset_iter = tf.data.Iterator.from_string_handle(handle, tr_dataset.output_types, tr_dataset.output_shapes) # create mock of iterator
    next_batch = list(dataset_iter.get_next()) #tuple --> list to make it possible to modify each elements

    tr_iter = tr_dataset.make_one_shot_iterator() # infinite loop
    va_iter = va_dataset.make_initializable_iterator() # require initialization in every epoch

    is_training = tf.placeholder(tf.bool, name='is_training')
    global_step = tf.Variable(0, name='global_step', trainable=False)

    print('Build network')
    loss, endpoints = build_network(config, next_batch, is_training, num_classes=tr_provider.NUM_CLASSES)

    if config.lr_decay:
        # copy from official/resnet
        batch_denom = 256
        initial_learning_rate = 0.1 * config.batch_size / batch_denom
        batches_per_epoch = tr_num_examples / config.batch_size
        boundary_epochs = [30, 60, 80, 90]
        decay_rates=[1, 0.1, 0.01, 0.001, 1e-4]
        boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
        lr_values = [initial_learning_rate * decay for decay in decay_rates]
        learning_rate = get_piecewise_lr(global_step, boundaries, lr_values, show_summary=True)

        # max_epoch = 50
        # boundaries = list((np.arange(max_epoch, dtype=np.int32)+1) * 5000)
        # lr_values = list(np.logspace(-1, -5, max_epoch))
        # learning_rate = get_piecewise_lr(global_step, boundaries, lr_values, show_summary=True)
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
    save_model_interval = 5000
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

        try:
            step, _ = sess.run([ops['step'], ops['minimize_op']], feed_dict=feed_dict)
        except:
            print('Error happens but keep training...')

        if check_counter(step, save_summary_interval):
            feed_dict = {
                ops['is_training']: False,
                ops['handle']: tr_handle,
            }
            fetch_dict = {
                'loss': ops['loss'],
                'top1': ops['top1'],
                'top5': ops['top5'],
                'summary': ops['summary'],
            }
            try:
                outs = sess.run(fetch_dict, feed_dict=feed_dict)
                start_time = time.time()
                outs = sess.run(fetch_dict, feed_dict=feed_dict)
                elapsed_time = time.time() - start_time
                train_writer.add_summary(outs['summary'], step) # save summary
                summaries = [tf.Summary.Value(tag='sec/step', simple_value=elapsed_time)]
                train_writer.add_summary(tf.Summary(value=summaries), global_step=step)
                train_writer.flush()

                print('[Train] {}step Loss: {:g}, Top1: {:g}, Top5: {:g} ({:.1f}sec)'.format(
                            step,
                            outs['loss'], outs['top1'], outs['top5'],
                            elapsed_time))
            except:
                print('Error happens but keep training...')

            if SAVE_MODEL and latest_saver is not None:
                latest_saver.save(sess, os.path.join(log_dir, 'models-latest'), global_step=step, write_meta_graph=False)

        # if SAVE_MODEL and best_saver is not None and check_counter(step, save_model_interval):
        #     # print('#{}step Save latest model'.format(step))
        #     best_saver.save(sess, os.path.join(log_dir, 'models-best'), global_step=step, write_meta_graph=False)

        if check_counter(step, valid_interval):
            eval_one_epoch(sess, ops, va_params)

if __name__ == '__main__':

    from utils.argparse_utils import *
    parser = get_parser()

    general_arg = add_argument_group('General', parser)
    general_arg.add_argument('--num_threads', type=int, default=8,
                            help='the number of threads (for dataset)')

    train_arg = add_argument_group('Train', parser)
    train_arg.add_argument('--log_dir', type=str, default='logs/imagenet',
                            help='where to save')
    train_arg.add_argument('--clear_logs', action='store_const',
                            const=True, default=False,
                            help='clear logs if it exists')
    train_arg.add_argument('--show_histogram', action='store_const',
                            const=True, default=False,
                            help='show variable / gradient histograms on tensorboard (consume a lot of disk space)')
    train_arg.add_argument('--max_itr', type=int, default=1000000,
                            help='max epoch')
    train_arg.add_argument('--batch_size', type=int, default=32,
                            help='batch size')
    train_arg.add_argument('--optim_method', type=str, default='adam',
                            help='adam, momentum, ftrl, rmsprop')
    train_arg.add_argument('--lr', type=float, default=1e-3,
                            help='learning rate')
    train_arg.add_argument('--lr_decay', type=str2bool, default=False,
                            help='apply lr decay')
    train_arg.add_argument('--weight_decay', type=float, default=0,
                            help='weight decay (not apply if 0, otherwise we suggest 5e-4)')

    dataset_arg = add_argument_group('Dataset', parser)
    dataset_arg.add_argument('--imagenet_dir', type=str, default='/cvlabdata1/home/ono/datasets/imagenet',
                            help='imagenet root directory')

    net_arg = add_argument_group('Network', parser)
    net_arg.add_argument('--model', type=str, default='custom_alexnet',
                            help='model name(custom_alexnet|custom_vgg)')

    config, unparsed = get_config(parser)

    if len(unparsed) > 0:
        raise ValueError('Miss finding argument: unparsed={}\n'.format(unparsed))

    main(config)