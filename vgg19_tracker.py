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

    def __init__(self, vgg19_npy_path=None, trainable=True):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable

    def build(self, key_img, search_img):
        """
        load variable from npy to build the VGG

        :param key_img: RGB [1, KEY_FRAME_SIZE, KEY_FRAME_SIZE, 3]
        :param search_img: RGB [batch, SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE, 3]
        """

        #TODO: assert values are [0, 255]

        # Convert RGB to BGR
        key_red, key_green, key_blue = tf.split(3, 3, key_img)
        key_bgr = tf.concat(3, [
            key_blue - VGG_MEAN[0],
            key_green - VGG_MEAN[1],
            key_red - VGG_MEAN[2],
        ])
        assert key_bgr.get_shape().as_list() == [1, KEY_FRAME_SIZE, KEY_FRAME_SIZE, 3]

        search_red, search_green, search_blue = tf.split(3, 3, search_img)
        search_bgr = tf.concat(3, [
            search_blue - VGG_MEAN[0],
            search_green - VGG_MEAN[1],
            search_red - VGG_MEAN[2],
        ])
        assert search_bgr.get_shape().as_list()[1:] == [SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE, 3]

        # Key frame extractor
        self.key_conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1")
        self.key_conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.key_pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.key_conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.key_conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.key_pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.key_conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.key_conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.key_conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.key_conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
        self.key_pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.key_conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.key_conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.key_conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.key_conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
        self.key_pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.key_conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.key_conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.key_conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.key_conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
        self.key_pool5 = self.max_pool(self.conv5_4, 'pool5')

        # Search frame extractor
        self.search_conv1_1 = self.conv_layer(bgr, 3, 64, "conv1_1")
        self.search_conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.search_pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.search_conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.search_conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.search_pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.search_conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.search_conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.search_conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.search_conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
        self.search_pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.search_conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.search_conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.search_conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.search_conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
        self.search_pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.search_conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.search_conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.search_conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.search_conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
        self.search_pool5 = self.max_pool(self.conv5_4, 'pool5')

        # Cross correlation layers
        # TODO: experiment with no whitening
        self.search_mean_pool5 = tf.reduce_mean(self.search_pool5, axis=[0,1,2], keep_dims=True)
        self.key_whitened_pool5 = self.key_pool5 - self.search_mean_pool5
        self.search_whitened_pool5 = self.search_pool5 - self.search_mean_pool5
        self.cross_corr5 = tf.nn.conv2d(
                self.search_whitened_pool5,
                tf.transpose(self.key_whitened_pool5, perm=[1,2,3,0]), # Conveniently batch size == out channels == 1
                [1, 1, 1, 1],   # TODO: experiment with striding
                padding='SAME')

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

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

        # print var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

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
