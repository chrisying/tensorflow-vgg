import os
import tensorflow as tf

import numpy as np
import time
import inspect

from CONSTANTS import *

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19:
    """
    A trainable version VGG19.
    """

    def __init__(self, vgg19_npy_path=None):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}

    def build(self, key_img, search_img, ground_truth):
        """
        load variable from npy to build the VGG

        :param key_img: RGB [1, KEY_FRAME_SIZE, KEY_FRAME_SIZE, 3]
        :param search_img: RGB [batch, SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE, 3]
        """

        #TODO: assert values are [0, 255]

        # Convert RGB to BGR
        key_red, key_green, key_blue = tf.split(axis=3, num_or_size_splits=3, value=key_img)
        key_bgr = tf.concat(axis=3, values=[
            key_blue - VGG_MEAN[0],
            key_green - VGG_MEAN[1],
            key_red - VGG_MEAN[2],
        ])
        assert key_bgr.get_shape().as_list() == [1, KEY_FRAME_SIZE, KEY_FRAME_SIZE, 3]

        search_red, search_green, search_blue = tf.split(axis=3, num_or_size_splits=3, value=search_img)
        search_bgr = tf.concat(axis=3, values=[
            search_blue - VGG_MEAN[0],
            search_green - VGG_MEAN[1],
            search_red - VGG_MEAN[2],
        ])
        assert search_bgr.get_shape().as_list()[1:] == [SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE, 3]

        # Key frame extractor
        self.key_conv1_1 = self.conv_layer(key_bgr, 3, 64, "conv1_1")
        self.key_conv1_2 = self.conv_layer(self.key_conv1_1, 64, 64, "conv1_2")
        self.key_pool1 = self.max_pool(self.key_conv1_2, 'pool1')

        self.key_conv2_1 = self.conv_layer(self.key_pool1, 64, 128, "conv2_1")
        self.key_conv2_2 = self.conv_layer(self.key_conv2_1, 128, 128, "conv2_2")
        self.key_pool2 = self.max_pool(self.key_conv2_2, 'pool2')

        self.key_conv3_1 = self.conv_layer(self.key_pool2, 128, 256, "conv3_1")
        self.key_conv3_2 = self.conv_layer(self.key_conv3_1, 256, 256, "conv3_2")
        self.key_conv3_3 = self.conv_layer(self.key_conv3_2, 256, 256, "conv3_3")
        self.key_conv3_4 = self.conv_layer(self.key_conv3_3, 256, 256, "conv3_4")
        self.key_pool3 = self.max_pool(self.key_conv3_4, 'pool3')

        self.key_conv4_1 = self.conv_layer(self.key_pool3, 256, 512, "conv4_1")
        self.key_conv4_2 = self.conv_layer(self.key_conv4_1, 512, 512, "conv4_2")
        self.key_conv4_3 = self.conv_layer(self.key_conv4_2, 512, 512, "conv4_3")
        self.key_conv4_4 = self.conv_layer(self.key_conv4_3, 512, 512, "conv4_4")
        self.key_pool4 = self.max_pool(self.key_conv4_4, 'pool4')

        self.key_conv5_1 = self.conv_layer(self.key_pool4, 512, 512, "conv5_1")
        self.key_conv5_2 = self.conv_layer(self.key_conv5_1, 512, 512, "conv5_2")
        self.key_conv5_3 = self.conv_layer(self.key_conv5_2, 512, 512, "conv5_3")
        self.key_conv5_4 = self.conv_layer(self.key_conv5_3, 512, 512, "conv5_4")
        self.key_pool5 = self.max_pool(self.key_conv5_4, 'pool5')

        # Search frame extractor
        self.search_conv1_1 = self.conv_layer(search_bgr, 3, 64, "conv1_1")
        self.search_conv1_2 = self.conv_layer(self.search_conv1_1, 64, 64, "conv1_2")
        self.search_pool1 = self.max_pool(self.search_conv1_2, 'pool1')

        self.search_conv2_1 = self.conv_layer(self.search_pool1, 64, 128, "conv2_1")
        self.search_conv2_2 = self.conv_layer(self.search_conv2_1, 128, 128, "conv2_2")
        self.search_pool2 = self.max_pool(self.search_conv2_2, 'pool2')

        self.search_conv3_1 = self.conv_layer(self.search_pool2, 128, 256, "conv3_1")
        self.search_conv3_2 = self.conv_layer(self.search_conv3_1, 256, 256, "conv3_2")
        self.search_conv3_3 = self.conv_layer(self.search_conv3_2, 256, 256, "conv3_3")
        self.search_conv3_4 = self.conv_layer(self.search_conv3_3, 256, 256, "conv3_4")
        self.search_pool3 = self.max_pool(self.search_conv3_4, 'pool3')

        self.search_conv4_1 = self.conv_layer(self.search_pool3, 256, 512, "conv4_1")
        self.search_conv4_2 = self.conv_layer(self.search_conv4_1, 512, 512, "conv4_2")
        self.search_conv4_3 = self.conv_layer(self.search_conv4_2, 512, 512, "conv4_3")
        self.search_conv4_4 = self.conv_layer(self.search_conv4_3, 512, 512, "conv4_4")
        self.search_pool4 = self.max_pool(self.search_conv4_4, 'pool4')

        self.search_conv5_1 = self.conv_layer(self.search_pool4, 512, 512, "conv5_1")
        self.search_conv5_2 = self.conv_layer(self.search_conv5_1, 512, 512, "conv5_2")
        self.search_conv5_3 = self.conv_layer(self.search_conv5_2, 512, 512, "conv5_3")
        self.search_conv5_4 = self.conv_layer(self.search_conv5_3, 512, 512, "conv5_4")
        self.search_pool5 = self.max_pool(self.search_conv5_4, 'pool5')

        # Cross correlation layers
        self.corr1 = self.cross_corr_layer(self.key_pool1, self.search_pool1, "corr1")
        self.corr2 = self.cross_corr_layer(self.key_pool2, self.search_pool2, "corr2")
        self.corr3 = self.cross_corr_layer(self.key_pool3, self.search_pool3, "corr3")
        self.corr4 = self.cross_corr_layer(self.key_pool4, self.search_pool4, "corr4")
        self.corr5 = self.cross_corr_layer(self.key_pool5, self.search_pool5, "corr5")

        # Loss
        # TODO: maybe experiment with learned upsampling via deconvolution
        self.rcorr1 = tf.image.resize_bilinear(self.corr1, (SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE))
        self.rcorr2 = tf.image.resize_bilinear(self.corr2, (SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE))
        self.rcorr3 = tf.image.resize_bilinear(self.corr3, (SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE))
        self.rcorr4 = tf.image.resize_bilinear(self.corr4, (SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE))
        self.rcorr5 = tf.image.resize_bilinear(self.corr5, (SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE))

        self.loss1 = tf.reduce_mean(tf.log(1.0 + tf.exp(-ground_truth * self.rcorr1)))
        self.loss2 = tf.reduce_mean(tf.log(1.0 + tf.exp(-ground_truth * self.rcorr2)))
        self.loss3 = tf.reduce_mean(tf.log(1.0 + tf.exp(-ground_truth * self.rcorr3)))
        self.loss4 = tf.reduce_mean(tf.log(1.0 + tf.exp(-ground_truth * self.rcorr4)))
        self.loss5 = tf.reduce_mean(tf.log(1.0 + tf.exp(-ground_truth * self.rcorr5)))

        # TODO: implement weighted loss via gate
        self.loss = self.loss1 + self.loss2 + self.loss3 + self.loss4 + self.loss5

        self.data_dict = None

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def cross_corr_layer(self, key_pool, search_pool, name):
        with tf.variable_scope(name):
            key_mean, key_var = tf.nn.moments(key_pool, [1,2,3], keep_dims=True)
            key_white = (key_pool - key_mean) / (tf.sqrt(key_var) + 0.0001)
            search_mean, search_var = tf.nn.moments(search_pool, [1,2,3], keep_dims=True)
            search_white = (search_pool - search_mean) / (tf.sqrt(search_var) + 0.0001)

            cross_corr = tf.nn.conv2d(
                    search_white,
                    tf.transpose(key_white, perm=[1,2,3,0]),
                    [1, 1, 1, 1],
                    padding='SAME')

            corr_bias = self.get_var(name, [1], 0, name + "_bias")
            bias = tf.nn.bias_add(cross_corr, corr_bias)

            corr_mean, corr_var = tf.nn.moments(bias, [1,2,3], keep_dims=True)
            corr_norm = (bias - corr_mean) / (tf.sqrt(corr_var) + 0.0001)

            return corr_norm

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        filters = self.get_var(name, [filter_size, filter_size, in_channels, out_channels], 0, name + "_filters")
        biases = self.get_var(name, [out_channels], 1, name + "_biases")

        return filters, biases

    def get_var(self, name, shape, idx, var_name):
        try:
            if self.data_dict is not None and name in self.data_dict:
                init = tf.constant_initializer(self.data_dict[name][idx])
                var = tf.get_variable(var_name, shape=shape, initializer=init)
                print 'Loaded Variable %s in %s from file' % (var_name, tf.get_variable_scope().name)
            else:
                init = tf.truncated_normal_initializer(shape, 0.0, 0.001)
                var = tf.get_variable(var_name, shape=shape, initializer=init)
                print 'Initialized Variable %s in %s' % (var_name, tf.get_variable_scope().name)
        except ValueError:
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                var = tf.get_variable(var_name)
                print 'Reused variable %s in %s' % (var_name, tf.get_variable_scope().name)

        self.var_dict[(name, idx)] = var
        return var

    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in self.var_dict.items():
            var_out = sess.run(var)
            if not data_dict.has_key(name):
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print("file saved", npy_path)
        return npy_path

    def get_var_count(self):
        count = 0
        for v in self.var_dict.values():
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
