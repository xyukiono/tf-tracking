# -*- coding: utf-8 -*-

import tensorflow as tf
from utils.tf_layer_utils import *

def get_model(inputs, 
              is_training,
              num_classes=None,
              activation_fn=tf.nn.relu,
              use_xavier=True, perform_bn=True, 
              reuse=False):
    print('===== Custom AlexNet =====')
    print('Input shape: {}'.format(inputs.shape))

    with tf.variable_scope('AlexNet', reuse=reuse) as net_sc:
        curr_in = inputs

        # conv1
        curr_in = conv2d(curr_in, 96, kernel_size=11, 
                         scope='conv1',
                         stride=2,
                         use_xavier=use_xavier,
                         padding='VALID',
                         use_bias=False
                        )
        curr_in = batch_norm_act(curr_in, activation_fn=activation_fn,
                                 perform_bn=perform_bn, 
                                 is_training=is_training,
                                 bnname='bn1'
                                )
        curr_in = max_pool2d(curr_in, 3, stride=2, name='pool1')

        # conv2
        curr_in = conv2d(curr_in, 128, kernel_size=5, 
                         scope='conv2',
                         use_xavier=use_xavier,
                         padding='VALID',
                         use_bias=False
                        )
        curr_in = batch_norm_act(curr_in, activation_fn=activation_fn,
                                 perform_bn=perform_bn, 
                                 is_training=is_training,
                                 bnname='bn2'
                                )
        curr_in = max_pool2d(curr_in, 3, stride=2, name='pool2')

        # conv3
        curr_in = conv2d(curr_in, 384, kernel_size=5, 
                         scope='conv3',
                         use_xavier=use_xavier,
                         padding='VALID',
                         use_bias=False
                        )
        curr_in = batch_norm_act(curr_in, activation_fn=activation_fn,
                                 perform_bn=perform_bn, 
                                 is_training=is_training,
                                 bnname='bn3'
                                )

        # conv4
        curr_in = conv2d(curr_in, 192, kernel_size=3, 
                         scope='conv4',
                         use_xavier=use_xavier,
                         padding='VALID',
                         use_bias=False
                        )
        curr_in = batch_norm_act(curr_in, activation_fn=activation_fn,
                                 perform_bn=perform_bn, 
                                 is_training=is_training,
                                 bnname='bn4'
                                )        

        # conv5
        curr_in = conv2d(curr_in, 128, kernel_size=3, 
                         scope='conv5',
                         use_xavier=use_xavier,
                         padding='VALID',
                         use_bias=False
                        )
        curr_in = batch_norm_act(curr_in, activation_fn=activation_fn,
                                 perform_bn=perform_bn, 
                                 is_training=is_training,
                                 bnname='bn5'
                                )
        if num_classes is not None:
            curr_in = max_pool2d(curr_in, 2, stride=2, name='pool3')
            curr_in = conv2d(curr_in, 128, kernel_size=3, 
                             scope='conv6',
                             use_xavier=use_xavier,
                             padding='VALID',
                             use_bias=False
                            )
            curr_in = batch_norm_act(curr_in, activation_fn=activation_fn,
                                     perform_bn=perform_bn, 
                                     is_training=is_training,
                                     bnname='bn6'
                                    )            
            print('Final conv shape: {}'.format(curr_in.shape))
            curr_in = tf.layers.flatten(curr_in)
            # FC1
            curr_in = fully_connected(curr_in, 4096, 'fc1', 
                                      use_xavier=use_xavier
                                     )
            curr_in = batch_norm_act(curr_in, activation_fn=activation_fn,
                                     perform_bn=perform_bn, 
                                     is_training=is_training,
                                     bnname='bn_fc1'
                                    )
            # FC2
            curr_in = fully_connected(curr_in, 4096, 'fc2', 
                                      use_xavier=use_xavier
                                     )
            curr_in = batch_norm_act(curr_in, activation_fn=activation_fn,
                                     perform_bn=perform_bn, 
                                     is_training=is_training,
                                     bnname='bn_fc2'
                                    )
            curr_in = fully_connected(curr_in, num_classes, 'fc3',
                                      use_xavier=use_xavier
                                     )

        logits = curr_in
        print('Output shape: {}'.format(logits.shape))

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, net_sc.name)
        endpoints = {}
        endpoints['logits'] = logits
        endpoints['var_list'] = var_list
        return logits, endpoints
