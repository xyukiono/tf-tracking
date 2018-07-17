from __future__ import print_function
import os
import sys
import numpy as np
import tensorflow as tf

from models import *
from cf_utils import *

MY_UTILS_PATH = '../dnnutils/'
if MY_UTILS_PATH not in sys.path:
    sys.path.append(MY_UTILS_PATH)

def build_network(config, feats_X, feats_Y, GX, GY):
    # feats_X : [B,H,W,C] feature maps of query
    # feats_Y : [B,H,W,C] feature maps of template
    # GX : [B,H,W,1] correlation filter of query (Ground truth)
    # GY : [B,H,W,1] correlation filter of template (always center)
    reglambda = config.reglambda
    batch_size = tf.shape(feats_X)[0]
    height = tf.shape(feats_X)[1]
    width = tf.shape(feats_X)[2]
    channels = feats_X.get_shape().as_list()[-1] # must be fixed

    #-------------------
    #  Forward
    #-------------------

    desired = GX # desired output (ground truth)

    FY = batch_fft2d(feats_Y)
    FX = batch_fft2d(feats_X)
    FGY = batch_fft2d(GY) # centerized

    FH1 = (tf.conj(FGY) * FY) / (tf.reduce_sum(FY * tf.conj(FY), axis=-1, keep_dims=True) + reglambda)
    # h1 = tf.real(batch_ifft2d(FH1))
    estimated = tf.reduce_sum(tf.real(batch_ifft2d(tf.conj(FH1) * FX)), axis=-1, keep_dims=True)

    loss_diff = estimated - desired
    loss = tf.reduce_mean(loss_diff * loss_diff)    

    #-------------------
    # Analytical gradients
    # You can also obtain each gradients (both analytical and numerical ways) by using tf.test.compute_gradient
    #-------------------
    gt_delLX, gt_delLY = tf.gradients(loss, [feats_X, feats_Y])

    #-------------------
    #  custom gradients
    #  We don't need to implement by ourselves in practical training because TF-auto differentiation takes care all of them
    #  You can also use "Defun" of tensorflow if you want to implement custom gradient from scratch.
    #-------------------
    FE = batch_fft2d(loss_diff / tf.cast(batch_size*height*width, tf.float32))
    delLH = tf.real(batch_ifft2d(FE * FX))
    delLX = tf.real(batch_ifft2d(FE * FH1))

    DY = tf.reduce_sum(FY * tf.conj(FY), axis=-1, keep_dims=True) + reglambda
    DY2 = tf.square(DY)
    K1_pre = tf.conj(FGY) / DY
    FA = batch_fft2d(delLH)    

    def inner_loop_body(l, k, grad):
        if k != l:
            K1 = 0
        else:
            K1 = K1_pre
        K2 = tf.conj(FGY) * (FY[...,l] * tf.conj(FY[...,k]))[...,None] / DY2
        K3 = tf.conj(FGY) * (FY[...,l] * FY[...,k])[...,None] / DY2
        FA_l = FA[...,l][...,None] # [B,H,W,1]
        grad = grad + (K1 - K2) * FA_l - K3 * tf.conj(FA_l)
        return l+1, k, grad

    num_parallel = 1
    back_prop = False
    delLY = []

    for k in range(channels):
        init_state = [0, k, tf.zeros_like(K1_pre)]
        condition = lambda l, _, _2: l < channels

        l, _, grad = tf.while_loop(condition, inner_loop_body, init_state,
                                       parallel_iterations=num_parallel,
                                       back_prop=back_prop)
        grad = batch_ifft2d(grad)
        delLY.append(grad)

    delLY = tf.concat(delLY, axis=-1)
    delLY = tf.real(delLY)

    endpoints = {
        'feats_X': feats_X,
        'feats_Y': feats_Y,
        'GX': GX,
        'GY': GY,
        'loss': loss,
        'delLX': delLX,
        'delLY': delLY,
        'delLH': delLH,
        'gt_delLX': gt_delLX,
        'gt_delLY': gt_delLY,
    }

    return endpoints

def main(config):
    tf.reset_default_graph() # for sure

    # set arbitrary tensor size
    batch_size = 4
    height = 24
    width = 16
    channels = 32

    feats_X = tf.placeholder(tf.float32, [batch_size, height, width, channels])
    feats_Y = tf.placeholder(tf.float32, [batch_size, height, width, channels])
    GX = tf.placeholder(tf.float32, [batch_size, height, width, 1])
    GY = tf.placeholder(tf.float32, [batch_size, height, width, 1])

    endpoints = build_network(config, feats_X, feats_Y, GX, GY)

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True # almost the same as tf.InteractiveSession
    sess = tf.Session(config=tfconfig)

    for itr in range(config.N):

        feed_dict = {
            feats_X: np.random.random([batch_size, height, width, channels]),
            feats_Y: np.random.random([batch_size, height, width, channels]),
            GX: np.random.random([batch_size, height, width, 1]),
            GY: np.random.random([batch_size, height, width, 1]),
        }

        fetch_dict = {
            'delLX': endpoints['delLX'],
            'delLY': endpoints['delLY'],
            'gt_delLX': endpoints['gt_delLX'],
            'gt_delLY': endpoints['gt_delLY'],
        }
        outs = sess.run(fetch_dict, feed_dict=feed_dict)

        Ex = np.max(np.abs(outs['delLX']-outs['gt_delLX']))
        Ey = np.max(np.abs(outs['delLY']-outs['gt_delLY']))

        print('#{}/{} Ex={}, Ey={}'.format(itr+1, config.N, Ex, Ey))


if __name__ == '__main__':
    from utils.argparse_utils import *
    parser = get_parser()

    parser.add_argument('--N', type=int, default=10,
                        help='the number of iteration')
    parser.add_argument('--reglambda', type=float, default=0.01,
                            help='lambda for regularization')
    config, unparsed = get_config(parser)

    if len(unparsed) > 0:
        raise ValueError('Warning: miss identify argument ?? unparsed={}\n'.format(unparsed))

    main(config)