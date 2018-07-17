import tensorflow as tf
from .mobilenet.mobilenet_v2 import mobilenet, training_scope


def mobilenet_v2(inputs, is_training, depth_multiplier=1.4, reuse=None, scope='MobilenetV2'):

    with tf.contrib.slim.arg_scope(training_scope(is_training=is_training)):
        # depth_multiplier=1.4 if you load a checkpoint from MobileNet_v2_1.4_224
        net, endpoints = mobilenet(inputs, num_classes=None, depth_multiplier=depth_multiplier, scope=scope, reuse=reuse)
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        endpoints['var_list'] = var_list
        return net, endpoints

mobilenet_v2.default_image_size = 224
mobilenet_v2.name = 'MobilenetV2'
