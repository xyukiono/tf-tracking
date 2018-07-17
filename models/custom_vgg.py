# -*- coding: utf-8 -*-

import tensorflow as tf
from utils.tf_layer_utils import *

def get_model(inputs, 
              is_training,
              num_classes,
              num_pooling=3,
              num_layer_per_block=2,
              init_out_channels=16,
              activation_fn=tf.nn.relu,
              use_xavier=True, perform_bn=True, 
              bn_decay=None, bn_affine=True, use_bias=True,
              reuse=False, data_format='channels_last'):
    # inputs is NHWC format
    out_channels_list = [init_out_channels*(2**i) for i in range(num_pooling)]
    print('======= VGG Style Network =======')
    print('Input shape: {}'.format(inputs.shape))
    
    set_data_format(data_format)
    
    with tf.variable_scope('VGG', reuse=reuse) as net_sc:
        curr_in = inputs

        if not is_NHWC(data_format):
            curr_in = tf.transpose(curr_in, [0, 3, 1, 2]) # [N,H,W,C] --> [N,C,H,W]
            
        for n in range(num_pooling):
            out_channels = out_channels_list[n]
            with tf.variable_scope('ConvBlock{}'.format(n+1)):
                for i in range(num_layer_per_block):
                    curr_in = conv2d(curr_in, out_channels, kernel_size=3,
                                     scope='conv{}-{}'.format(n+1,i+1),
                                     use_xavier=use_xavier,
                                     use_bias=use_bias
                                    )
                    curr_in = batch_norm_act(curr_in, activation_fn=activation_fn,
                                             perform_bn=perform_bn, 
                                             is_training=is_training,
                                             bn_decay=bn_decay,
                                             bn_affine=bn_affine, 
                                             bnname='bn{}-{}'.format(n+1,i+1)
                                            )
            # -------
            curr_in = max_pool2d(curr_in, 2, stride=2, name='pool{}'.format(n+1))
            print('#{} conv-block shape: {}'.format(n+1, curr_in.shape))
        # Finalize
        
        curr_in = tf.layers.flatten(curr_in)
        curr_in = fully_connected(curr_in, 2048, 'fc1', 
                                  use_xavier=use_xavier
                                 )
        curr_in = batch_norm_act(curr_in, activation_fn=activation_fn,
                                 perform_bn=perform_bn, 
                                 is_training=is_training,
                                 bn_decay=bn_decay,
                                 bn_affine=bn_affine, 
                                 bnname='bn_fc1'
                                )
        logits = fully_connected(curr_in, num_classes, 'logits',
                                 use_xavier=use_xavier,
                                )
        print('logits shape: {}'.format(logits.shape))

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, net_sc.name)
        endpoints = {}
        endpoints['logits'] = logits
        endpoints['var_list'] = var_list
        return logits, endpoints
