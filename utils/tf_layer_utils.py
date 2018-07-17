# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

# data_format = 'channels_first', 'NCHW' or 'channels_last', 'NHWC'
_DATA_FORMAT = 'channels_last'

_SHOW_VAR_SUMMARY = False
_SHOW_GRAD_SUMMARY = False

def set_data_format(data_format):
    global _DATA_FORMAT
    _DATA_FORMAT = data_format

def set_summary_visibility(variables=True, gradients=True):
    global _SHOW_VAR_SUMMARY
    global _SHOW_GRAD_SUMMARY
    _SHOW_VAR_SUMMARY = variables
    _SHOW_GRAD_SUMMARY = gradients

def is_NHWC(data_format):
    data_format = data_format.lower()
    if data_format == 'channels_last' or data_format == 'nhwc':
        return True
    elif data_format == 'channels_first' or data_format == 'nchw':
        return False
    else:
        raise ValueError('Unknown data_format: {}\n'.format(data_format))

def get_shape_as_list(x):
    return [_s if _s is not None else - 1 for _s in x.get_shape().as_list()]

def _get_variable(name, shape, initializer, dtype=tf.float32):
    if shape is None:
        return None
    else:
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        return var

def _get_W_b(wshape, bshape, use_xavier=True, dtype=tf.float32):
    """Helper to create an initialized Variable with weight decay.
    
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        shape: list of ints
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
        use_xavier: bool, whether to use xavier initializer

    Returns:
        Variable Tensor
    """

    if use_xavier:
        initializer = tf.contrib.layers.xavier_initializer()
    else:
        initializer = tf.variance_scaling_initializer(mode='fan_in')

    if isinstance(bshape, int):
        bshape = [bshape] 

    W = _get_variable('weights', wshape, initializer, dtype=dtype)
    b = _get_variable('biases', bshape, initializer=tf.zeros_initializer(), dtype=dtype)

    if _SHOW_VAR_SUMMARY:
        if W is not None:
            tf.summary.histogram('weights', W)
        if b is not None:
            tf.summary.histogram('biases', b)

    return W, b

def batch_norm_template(inputs, is_training, scope, moments_dims, bn_decay=None, affine=True):
    """ Batch normalization on convolutional maps and beyond...
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

    Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
    Return:
      normed:        batch-normalized maps
    """
    with tf.variable_scope(scope) as sc:
        if len(moments_dims) == 1:
            num_channels = inputs.get_shape()[-1].value
        elif len(moments_dims) == 3:
            if 1 in moments_dims:
                # NHWC order
                num_channels = inputs.get_shape()[-1].value
            else:
                # NCHW order
                num_channels = inputs.get_shape()[1].value
        else:
            raise ValueError('custom_batch_norm_act suppose len(moments_dim) is either 1 or 3: moments_dim={}\n'.format(moments_dim))

        beta = _get_variable('beta', [num_channels], initializer=tf.zeros_initializer, dtype=tf.float32)
        gamma = _get_variable('gamma', [num_channels], initializer=tf.ones_initializer, dtype=tf.float32)

        batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
        decay = bn_decay if bn_decay is not None else 0.9
        ema = tf.train.ExponentialMovingAverage(decay=decay)
        # Operator that maintains moving averages of variables.
        ema_apply_op = tf.cond(is_training,
                               lambda: ema.apply([batch_mean, batch_var]),
                               lambda: tf.no_op())
    
        # Update moving average and return current batch's avg and var.
        def mean_var_with_update():
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
    
        # ema.average returns the Variable holding the average of var.
        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))

        if _SHOW_VAR_SUMMARY:
            tf.summary.histogram('beta', beta)
            tf.summary.histogram('gamma', gamma)
            tf.summary.histogram('mean', mean)
            tf.summary.histogram('var', var)

        normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)

        return normed

# def batch_norm_for_fc(inputs, is_training, bn_decay, scope='bn', affine=True):
#     return batch_norm_template(inputs, is_training, scope, [0,], bn_decay, affine=affine)
# def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope='bn', affine=True):
#     return batch_norm_template(inputs, is_training, scope, [0,1,2], bn_decay, affine=affine)

def custom_batch_norm_act(inputs, activation_fn=tf.nn.relu, 
                      perform_bn=False, is_training=None, 
                      bn_decay=None, bn_affine=True, 
                      bnname='bn',
                      actname=None,
                      data_format=None):
    # [Warning] scope has to be unique
    # [Warning] can't use under reuse=True
    if data_format is None:
        data_format = _DATA_FORMAT
    if perform_bn:
        inputs_dim = inputs.get_shape().ndims
        if inputs_dim == 4:
            # batch norm for conv
            moments_dims = [0,1,2] if is_NHWC(data_format) else [0,2,3]
        elif inputs_dim == 2:
            # batch norm for fully-connected
            moments_dims = [0,]
        else:
            raise ValueError('custom_batch_norm_act suppose inputs dim is either 2 or 4: inputs_dim={}\n'.format(inputs_dim))

        inputs = batch_norm_template(inputs, is_training, bnname, moments_dims, bn_decay=bn_decay, affine=bn_affine)

    if activation_fn is not None:
        inputs = activation_fn(inputs, name=actname)

    return inputs

def tf_batch_norm_act(inputs, activation_fn=tf.nn.relu, 
                      perform_bn=False, is_training=None, 
                      bn_decay=None, bn_affine=True,
                      bnname=None,
                      actname=None,
                      data_format=None):
    # You have to use UPDATE_OPS update at optimizer gradient descenet
    # NHWC
    # use default one

    # scope is invalid argument

    if data_format is None:
        data_format = _DATA_FORMAT
    if perform_bn:
        # _BATCH_NORM_DECAY = 0.997 # bn_decay
        _BATCH_NORM_DECAY = bn_decay if bn_decay is not None else 0.9
        _BATCH_NORM_EPSILON = 1e-5

        if is_NHWC(data_format):
            axis = -1
        else:
            axis = 1

        inputs = tf.layers.batch_normalization(
                    inputs, axis=axis, 
                    momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
                    scale=bn_affine, training=is_training, fused=True, name=bnname
        )
    if activation_fn is not None:
        inputs = activation_fn(inputs, name=actname)
    return inputs

# User can select batch_norm_act implementation
# batch_norm_act = custom_batch_norm_act
batch_norm_act = tf_batch_norm_act

def dropout(inputs,
            is_training,
            scope='drop',
            keep_prob=0.5,
            noise_shape=None):
    """ Dropout layer.

    Args:
        inputs: tensor
        is_training: boolean tf.Variable
        scope: string
        keep_prob: float in [0,1]
        noise_shape: list of ints

    Returns:
        tensor variable
    """
    with tf.variable_scope(scope) as sc:
        outputs = tf.cond(is_training,
                        lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                        lambda: inputs)
        return outputs

def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=1,
           padding='SAME',
           use_xavier=True,
           use_bias=True,
           data_format=None):
    """ 2D convolution with non-linear operation.

    Args:
        inputs: 4-D tensor variable BxHxWxC
        num_output_channels: int
        kernel_size: kernel size (schalar)
        scope: string
        stride: stride size (schalar)
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

    Returns:
        Variable tensor
    """
    if data_format is None:
        data_format = _DATA_FORMAT    
        
    with tf.variable_scope(scope) as sc:

        if is_NHWC(data_format):
            num_in_channels = inputs.get_shape()[-1].value
            strides = [1, stride, stride, 1]
            data_format = 'NHWC'
        else:
            num_in_channels = inputs.get_shape()[1].value
            strides = [1, 1, stride, stride]
            data_format = 'NCHW'

        wshape = [kernel_size, kernel_size,
                        num_in_channels, num_output_channels] # [kernel_h, kernel_w, in_channels, out_channels] whatever data_format
        bshape = num_output_channels if use_bias else None

        W, b = _get_W_b(wshape, bshape, use_xavier=use_xavier)

        outputs = tf.nn.conv2d(inputs, W, 
                                strides,
                                padding=padding,
                                data_format=data_format)

        if b is not None:
            outputs = tf.nn.bias_add(outputs, b,
                                     data_format=data_format
                                    )

        return outputs

def fixed_padding(inputs, kernel_size, data_format):
    """Pads the input along the spatial dimensions independently of input size.
    Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').
    Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if is_NHWC(data_format):
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
    else:
        padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                [pad_beg, pad_end], [pad_beg, pad_end]])
    return padded_inputs

def conv2d_fixed_padding(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=1,
           use_xavier=True,
           use_bias=True,
           data_format=None):
    """Strided 2-D convolution with explicit padding."""
    # The padding is consistent and is based only on `kernel_size`, not on the
    # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
    # ex) In case kernel_size=3, stride=[2,2] , it is impossible to keep same size as input neither padding='VALID', 'SAME'
    if data_format is None:
        data_format = _DATA_FORMAT
    if stride > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)

    padding= 'SAME' if stride == 1 else 'VALID'
    return conv2d(inputs, num_output_channels, 
           kernel_size, scope, 
           stride=stride, 
           padding=padding,
           use_xavier=use_xavier,
           use_bias=use_bias,
           data_format=data_format
        )

def conv2d_transpose(inputs,
                   num_output_channels,
                   kernel_size,
                   scope,
                   stride=2,
                   output_shape=None,
                   padding='SAME',
                   use_xavier=True,
                   use_bias=True,
                   data_format=None):
    if data_format is None:
        data_format = _DATA_FORMAT    
        
    with tf.variable_scope(scope) as sc:
        in_sh = tf.shape(inputs)
        if is_NHWC(data_format):
            num_in_channels = inputs.get_shape()[-1].value
            strides = [1, stride, stride, 1]
            data_format = 'NHWC'
            if output_shape is None:
                output_shape = tf.stack([in_sh[0], in_sh[1]*stride, in_sh[2]*stride, num_output_channels])
            data_format = 'NHWC'
        else:
            num_in_channels = inputs.get_shape()[1].value
            strides = [1, 1, stride, stride]
            if output_shape is None:
                output_shape = tf.stack([in_sh[0], num_output_channels, in_sh[2]*stride, in_sh[3]*stride])
            data_format = 'NCHW'

        wshape = [kernel_size, kernel_size,
                        num_output_channels, num_in_channels] # [kernel_h, kernel_w, out_channels, in_channels] whatever data_format
        bshape = num_output_channels if use_bias else None

        W, b = _get_W_b(wshape, bshape, use_xavier=use_xavier)

        outputs = tf.nn.conv2d_transpose(inputs, W, output_shape,
                                strides=strides,
                                padding=padding,
                                data_format=data_format)

        if b is not None:
            outputs = tf.nn.bias_add(outputs, b,
                                     data_format=data_format
                                    )

        return outputs

def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    use_bias=True):
    """ Fully connected layer with non-linear operation.

    Args:
        inputs: 2-D tensor BxN
        num_outputs: int

    Returns:
        Variable tensor of size B x num_outputs.
    """


    with tf.variable_scope(scope) as sc:
        num_input_units = inputs.get_shape()[-1].value

        wshape = [num_input_units, num_outputs]
        bshape = num_outputs if use_bias else None

        W, b = _get_W_b(wshape, bshape, 
                        use_xavier=use_xavier)
        outputs = tf.matmul(inputs, W)

        if b is not None:
            outputs = tf.nn.bias_add(outputs, b)

        return outputs

def fully_connected_custom(inputs,
                    num_outputs,
                    scope,
                    W_initializer,
                    b_initializer
                    ):
    """ Fully connected layer with non-linear operation.

    Args:
        inputs: 2-D tensor BxN
        num_outputs: int

    Returns:
        Variable tensor of size B x num_outputs.
    """


    with tf.variable_scope(scope) as sc:
        num_input_units = inputs.get_shape()[-1].value

        if hasattr(W_initializer, 'shape'):
            wshape = None # you don't have to specify shape if initializer already has it
        else:
            wshape = [num_input_units, num_outputs]
        if hasattr(b_initializer, 'shape'):
            bshape = None
        else:
            bshape = [num_outputs]

        W = tf.get_variable('weights', wshape, initializer=W_initializer, dtype=tf.float32)
        b = tf.get_variable('biases', bshape, initializer=b_initializer, dtype=tf.float32)

        outputs = tf.matmul(inputs, W)
        outputs = tf.nn.bias_add(outputs, b)

        return outputs

def max_pool2d(inputs,
               kernel_size,
               stride=2,
               padding='VALID',
               name=None,
               data_format=None):
    """ 2D max pooling.

    Args:
    inputs: 4-D tensor B,H,W,C or B,C,H,W
    kernel_size: int scalar
    stride: int scalar

    Returns:
    Variable tensor
    """
    if data_format is None:
        data_format = _DATA_FORMAT
    if is_NHWC(data_format):
        ksize = [1, kernel_size, kernel_size, 1]
        strides = [1, stride, stride, 1]
        data_format = 'NHWC' # max_pool doesn't allow channels_firsst nor channels_last
    else:
        ksize = [1, 1, kernel_size, kernel_size]
        strides = [1, 1, stride, stride]
        data_format = 'NCHW'
    outputs = tf.nn.max_pool(inputs,
                             ksize=ksize,
                             strides=strides,
                             padding=padding,
                             name=name,
                             data_format=data_format)
    return outputs

def avg_pool2d(inputs,
               kernel_size,
               stride=2,
               padding='VALID',
               name=None,
               data_format=None):
    """ 2D avg pooling.

    Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints

    Returns:
    Variable tensor
    """
    if data_format is None:
        data_format = _DATA_FORMAT
    if is_NHWC(data_format):
        ksize = [1, kernel_size, kernel_size, 1]
        strides = [1, stride, stride, 1]
        data_format = 'NHWC' # max_pool doesn't allow channels_firsst nor channels_last
    else:
        ksize = [1, 1, kernel_size, kernel_size]
        strides = [1, 1, stride, stride]
        data_format = 'NCHW'

    outputs = tf.nn.avg_pool(inputs,
                             ksize=ksize,
                             strides=strides,
                             padding=padding,
                             name=name,
                             data_format=data_format)
    return outputs

def global_avg_pool2d(inputs, name=None, data_format=None):
    # compress H,W axis
    # inputs [B,H,W,C] --> outputs [B,C]
    assert inputs.get_shape().ndims == 4
    if data_format is None:
        data_format = _DATA_FORMAT
    if is_NHWC(data_format):
        return tf.reduce_mean(inputs, [1,2], name=name) 
    else:
        return tf.reduce_mean(inputs, [2,3], name=name)

def global_max_pool2d(inputs, name=None, data_format=None):
    assert inputs.get_shape().ndims == 4
    if data_format is None:
        data_format = _DATA_FORMAT
    if is_NHWC(data_format):
        return tf.reduce_max(inputs, [1,2], name=name) 
    else:
        return tf.reduce_max(inputs, [2,3], name=name)

def leaky_relu(x, leak=0.2, name="lrelu"):
    print('LEAK ', leak)
    return tf.maximum(x, leak*x)

def crop_and_concat(x1,x2, data_format=None):
    if data_format is None:
        data_format = _DATA_FORMAT    
    # x1.shape >= x2.shape
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    if is_NHWC(data_format):
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)
    else:
        offsets = [0, 0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2]
        size = [-1, -1, x2_shape[1], x2_shape[2]]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 1)

