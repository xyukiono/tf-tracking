#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import logging
import os
import os.path as osp

import numpy as np
import tensorflow as tf
from utils.misc import *
from models import *
from cf_utils import *

class InferenceCFCF():

    def __init__(self, config):
        self.config = config
        self.image = None
        self.target_bbox_feed = None
        self.search_images = None
        self.embeds = None
        self.templates = None
        self.init = None
        self.model_config = None
        self.track_config = None
        self.response_up = None

        if config.backbone == 'vgg16':
            self.backbone = vgg.vgg_16
        else:
            raise ValueError('Unknown backbone: {}'.format(config.backbone))
        # if config.backbone == 'alexnet':
        #     self.backbone = alexnet
        # else:
        #     raise ValueError('Unknown backbone: {}'.format(config.backbone))

    def build_graph_from_config(self):
        self.build_model()
        ema = tf.train.ExponentialMovingAverage(0)
        variables_to_restore = ema.variables_to_restore(moving_avg_variables=[]) # get variables.moving_average_variables() + variables.trainable_variables() 

        # Filter out State variables
        variables_to_restore_filterd = {}
        for key, value in variables_to_restore.items():
            print(key)
            if key.split('/')[1] != 'State':
                variables_to_restore_filterd[key] = value

        return variables_to_restore_filterd

    def build_model(self):
        self.build_inputs()
        self.build_search_images()
        self.build_template()
        self.build_detection()
        self.build_upsample()
        self.dumb_op = tf.no_op('dumb_operation')
        self.summary_op = tf.no_op('summary_operation')
        self.summary_writer = None
        self.summary_count = 0

    def build_summary(self, summary_writer):
        if isinstance(self.summary_op, tf.Operation) and self.summary_op.type == 'NoOp':
            self.summary_op = tf.summary.merge_all()
        self.summary_writer = summary_writer
        self.summary_count = 0

    def build_inputs(self):
        filename = tf.placeholder(tf.string, [], name='filename')
        image_file = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_file, channels=3, dct_method="INTEGER_ACCURATE")
        image = tf.to_float(image)
        self.image = image
        self.target_bbox_feed = tf.placeholder(dtype=tf.float32,
                                               shape=[4],
                                               name='target_bbox_feed')  # center's y, x, height, width
    def build_search_images(self):
        """Crop search images from the input image based on the last target position

        1. The input image is scaled such that the area of target&context takes up to (scale_factor * z_image_size) ^ 2
        2. Crop an image patch as large as x_image_size centered at the target center.
        3. If the cropped image region is beyond the boundary of the input image, mean values are padded.
        """
        config = self.config

        size_z = config.z_image_size # 255
        size_x = config.x_image_size # 255
        context_amount = 0.5

        num_scales = config.num_scales # 3
        scales = np.arange(num_scales) - get_center(num_scales) # [-1,0,1]
        assert np.sum(scales) == 0, 'scales should be symmetric'
        search_factors = [config.scale_step ** x for x in scales] # scale_step=1.0375, [0.9638554216867469, 1.0, 1.0375]

        frame_sz = tf.shape(self.image)
        target_yx = self.target_bbox_feed[0:2] #y,x
        target_size = self.target_bbox_feed[2:4] # height, width
        avg_chan = tf.reduce_mean(self.image, axis=(0, 1), name='avg_chan')

        # Compute base values
        base_z_size = target_size
        base_z_context_size = base_z_size + context_amount * tf.reduce_sum(base_z_size)
        base_s_z = tf.sqrt(tf.reduce_prod(base_z_context_size))  # Canonical size
        base_scale_z = tf.div(tf.to_float(size_z), base_s_z)
        d_search = (size_x - size_z) / 2.0
        base_pad = tf.div(d_search, base_scale_z)
        base_s_x = base_s_z + 2 * base_pad
        base_scale_x = tf.div(tf.to_float(size_x), base_s_x)

        boxes = []
        for factor in search_factors:
            s_x = factor * base_s_x
            frame_sz_1 = tf.to_float(frame_sz[0:2] - 1)
            topleft = tf.div(target_yx - get_center(s_x), frame_sz_1)
            bottomright = tf.div(target_yx + get_center(s_x), frame_sz_1)
            box = tf.concat([topleft, bottomright], axis=0)
            boxes.append(box)
        boxes = tf.stack(boxes)

        scale_xs = []
        for factor in search_factors:
            scale_x = base_scale_x / factor
            scale_xs.append(scale_x)
        self.scale_xs = tf.stack(scale_xs)

        # Note we use different padding values for each image
        # while the original implementation uses only the average value
        # of the first image for all images.
        image_minus_avg = tf.expand_dims(self.image - avg_chan, 0)
        image_cropped = tf.image.crop_and_resize(image_minus_avg, boxes,
                                                 box_ind=tf.zeros((num_scales), tf.int32),
                                                 crop_size=[size_x, size_x])
        self.search_images = image_cropped + avg_chan


    def get_hanning_tensor(self, height, width):
        hann2t, hann1t = np.ogrid[0:height, 0:width]
        hann1t = 0.5 * (1 - np.cos(2*np.pi*hann1t/(width-1)))
        hann2t = 0.5 * (1 - np.cos(2*np.pi*hann2t/(height-1)))
        hann2d = hann2t * hann1t
        hann4d = hann2d[None,...,None]

        hann4d_tensor = tf.convert_to_tensor(hann4d, dtype=tf.float32)
        return hann4d_tensor # [1,H,W,1]

    def get_image_embedding(self, images, reuse=None):
        imgnet_mean = tf.convert_to_tensor([123.68, 116.78, 103.94])
        print('Subtract ImageNet mean')
        images = images - imgnet_mean

        _, endpoints = self.backbone(images, is_training=False, reuse=reuse)

        embed = endpoints[self.config.feat_layer]

        emb_height, emb_width = embed.get_shape().as_list()[1:3]
        cyclic_window = self.get_hanning_tensor(emb_height, emb_width)
        embed = embed * cyclic_window # apply cosine window
        return embed

    def build_template(self):
        config = self.config
        num_scales = config.num_scales

        if config.z_image_size < config.x_image_size:
            # Exemplar image lies at the center of the search image in the first frame
            exemplar_images = get_exemplar_images(self.search_images, [config.z_image_size,
                                                                       config.z_image_size])
        else:
            exemplar_images = self.search_images

        tf.summary.image('template_images', exemplar_images)
        feat_maps = self.get_image_embedding(exemplar_images)

        center_scale = int(get_center(num_scales))
        center_feat_maps = tf.identity(feat_maps[center_scale])
        feat_maps = tf.stack([center_feat_maps for _ in range(num_scales)])

        # Correlation Filter
        im_size, _ = exemplar_images.get_shape().as_list()[1:3]
        feat_size, _ = feat_maps.get_shape().as_list()[1:3]
        gauss_response = get_template_correlation_response(im_size=im_size, out_size=[feat_size, feat_size])
        GZ = tf.convert_to_tensor(gauss_response[None,...,None]) # [1,H,W,1]
        GZ = tf.tile(GZ, [num_scales,1,1,1]) # [B,H,W,1]

        FZ = batch_fft2d(feat_maps)
        FGZ = batch_fft2d(GZ) # centerized
        # template in frequency domain
        templates = (tf.conj(FGZ) * FZ) / (tf.reduce_sum(FZ * tf.conj(FZ), axis=-1, keep_dims=True) + config.reglambda)
        self.templates_out = templates
        self.templates_feed = tf.placeholder(tf.complex64, templates.get_shape().as_list(), 
                                            name='templates_feed')

        with tf.variable_scope('target_template'):
            # Store template in Variable such that we don't have to feed this template every time.
            with tf.variable_scope('State'):
                state = tf.get_variable('exemplar',
                                        initializer=tf.zeros(templates.get_shape().as_list(), dtype=templates.dtype),
                                        trainable=False)
                with tf.control_dependencies([templates]):
                    self.init = tf.assign(state, templates, validate_shape=True) # if you run 'init', template value will be hold
            self.templates = state
            self.update_op = tf.assign(state, config.update_rate*self.templates+(1.0-config.update_rate)*self.templates_feed)

    def build_detection(self):
        config = self.config
        tf.summary.image('search_images', self.search_images)
        feat_maps = self.get_image_embedding(self.search_images, reuse=True)

        # Apply correlation filter on frequency domain
        FX = batch_fft2d(feat_maps)
        FH = self.templates
        self.response = tf.reduce_sum(tf.real(batch_ifft2d(tf.conj(FH) * FX)), axis=-1, keep_dims=True)
        # MMR
        max_vals = tf.reduce_max(self.response, axis=[1,2,3])
        mean_vals = tf.reduce_mean(self.response, axis=[1,2,3])
        self.MMRs = max_vals / (mean_vals+1e-6)

    def build_upsample(self):
        """Upsample response to obtain finer target position"""
        config = self.config

        with tf.variable_scope('upsample'):
            response = self.response # [num_scales, H,W,1]
            up_method = self.config.upsample_method
            methods = {'bilinear': tf.image.ResizeMethod.BILINEAR,
                     'bicubic': tf.image.ResizeMethod.BICUBIC}
            up_method = methods[up_method]
            response_spatial_size = self.response.get_shape().as_list()[1:3]
            up_size = [s * config.upsample_factor for s in response_spatial_size]
            response_up = tf.image.resize_images(response,
                                               up_size,
                                               method=up_method,
                                               align_corners=True)
            tf.summary.image('response', response)
            response_up = tf.squeeze(response_up, [3])
            tf.summary.histogram('response_up', response_up)
            self.response_up = response_up

    def initialize(self, sess, input_feed):
        image_path, target_bbox = input_feed

        scale_xs, _, summaries = sess.run([self.scale_xs, self.init, self.summary_op],
                                feed_dict={'filename:0': image_path,
                                "target_bbox_feed:0": target_bbox, })
        if self.summary_writer is not None:
            self.summary_writer.add_summary(summaries, self.summary_count)
            self.summary_count += 1
        return scale_xs

    def inference_step(self, sess, input_feed):
        image_path, target_bbox = input_feed
        log_level = self.config.log_level
        image_cropped_op = self.search_images if log_level > 0 else self.dumb_op
        image_cropped, scale_xs, response_output, MMRs, summaries = sess.run(
                fetches=[image_cropped_op, self.scale_xs, self.response_up, self.MMRs, self.summary_op],
                feed_dict={
                    "filename:0": image_path,
                    "target_bbox_feed:0": target_bbox, })

        if self.summary_writer is not None:
            self.summary_writer.add_summary(summaries, self.summary_count)
            self.summary_count += 1

        output = {
          'image_cropped': image_cropped,
          'scale_xs': scale_xs,
          'response': response_output,
          'MMRs': MMRs,}
        return output, None

    def update(self, sess, input_feed):
        image_path, target_bbox = input_feed
        print(image_path)
        templates = sess.run(self.templates_out,
                                feed_dict={'filename:0': image_path,
                                "target_bbox_feed:0": target_bbox, })
        sess.run(self.update_op, feed_dict={'templates_feed:0': templates})
                                


