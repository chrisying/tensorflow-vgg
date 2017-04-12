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
        self.cnn_var_list = []
        self.gate_var_list = []
        self.iter_num = 0

        self.build()

    def build(self):
        """
        load variable from npy to build the VGG

        key_img: RGB [1, KEY_FRAME_SIZE, KEY_FRAME_SIZE, 3]
        search_img: RGB [batch, SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE, 3]
        key_bb: [2] (width, height) after scale conversion
        search_bb: [batch, 4] (x, y, w, h) after scale conversion
        """

        self.sess = tf.Session()

        # Inputs
        self.key_img = tf.placeholder(tf.float32, [1, KEY_FRAME_SIZE, KEY_FRAME_SIZE, 3])
        self.search_img = tf.placeholder(tf.float32, [None, SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE, 3])
        self.key_bb = tf.placeholder(tf.float32, [2])
        self.search_bb = tf.placeholder(tf.float32, [None, 4])

        # Convert RGB to BGR
        key_red, key_green, key_blue = tf.split(axis=3, num_or_size_splits=3, value=self.key_img)
        key_bgr = tf.concat(axis=3, values=[
            key_blue - VGG_MEAN[0],
            key_green - VGG_MEAN[1],
            key_red - VGG_MEAN[2],
        ])
        assert key_bgr.get_shape().as_list() == [1, KEY_FRAME_SIZE, KEY_FRAME_SIZE, 3]

        search_red, search_green, search_blue = tf.split(axis=3, num_or_size_splits=3, value=self.search_img)
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
        #self.key_pool5 = self.max_pool(self.key_conv5_4, 'pool5')
        self.key_pool5 = self.key_conv5_4   # don't downsample last output, 8 x 8

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
        #self.search_pool5 = self.max_pool(self.search_conv5_4, 'pool5')
        self.search_pool5 = self.search_conv5_4     # don't max pool last output, 16 x 16

        # Downsampled feature maps to same size
        self.key_pool1 = self.max_pool_n(self.key_pool1, "cpool1_1", 3)
        self.key_pool2 = self.max_pool_n(self.key_pool2, "cpool2_1", 2)
        self.key_pool3 = self.max_pool_n(self.key_pool3, "cpool3_1", 1)
        self.key_pool4 = self.key_pool4
        self.key_pool5 = self.key_pool5

        self.search_pool1 = self.max_pool_n(self.search_pool1, "cpool1_2", 3)
        self.search_pool2 = self.max_pool_n(self.search_pool2, "cpool2_2", 2)
        self.search_pool3 = self.max_pool_n(self.search_pool3, "cpool3_2", 1)
        self.search_pool4 = self.search_pool4
        self.search_pool5 = self.search_pool5

        # Cross correlation layers
        self.corr1 = self.cross_corr_layer(self.key_pool1, self.search_pool1, "corr1")
        self.corr2 = self.cross_corr_layer(self.key_pool2, self.search_pool2, "corr2")
        self.corr3 = self.cross_corr_layer(self.key_pool3, self.search_pool3, "corr3")
        self.corr4 = self.cross_corr_layer(self.key_pool4, self.search_pool4, "corr4")
        self.corr5 = self.cross_corr_layer(self.key_pool5, self.search_pool5, "corr5")

        # Loss
        # Upsample to original search size
        self.rcorr1 = tf.image.resize_nearest_neighbor(self.corr1, (SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE))
        self.rcorr2 = tf.image.resize_nearest_neighbor(self.corr2, (SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE))
        self.rcorr3 = tf.image.resize_nearest_neighbor(self.corr3, (SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE))
        self.rcorr4 = tf.image.resize_nearest_neighbor(self.corr4, (SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE))
        self.rcorr5 = tf.image.resize_nearest_neighbor(self.corr5, (SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE))

        # Gating feature vectors from pre-resized feature maps
        # TODO: don't hardcode 2**4
        self.gate1 = self.extract_corr_features(self.corr1, SEARCH_FRAME_SIZE / 2 ** 4, 1)
        self.gate2 = self.extract_corr_features(self.corr2, SEARCH_FRAME_SIZE / 2 ** 4, 2)
        self.gate3 = self.extract_corr_features(self.corr3, SEARCH_FRAME_SIZE / 2 ** 4, 3)
        self.gate4 = self.extract_corr_features(self.corr4, SEARCH_FRAME_SIZE / 2 ** 4, 4)
        self.gate5 = self.extract_corr_features(self.corr5, SEARCH_FRAME_SIZE / 2 ** 4, 5)

        # Confidence of gates
        self.conf1 = self.confidence_layer(self.gate1, 'conf1') # B x 1
        self.conf2 = (1.0 - self.conf1) * self.confidence_layer(self.gate2, 'conf2')
        self.conf3 = (1.0 - self.conf1 - self.conf2) * self.confidence_layer(self.gate3, 'conf3')
        self.conf4 = (1.0 - self.conf1 - self.conf2 - self.conf3) * self.confidence_layer(self.gate4, 'conf4')
        self.conf5 = (1.0 - self.conf1 - self.conf2 - self.conf3 - self.conf4)

        # Rescaled confidence (sum ~ 1.0)
        #sum_conf = self.conf1 + self.conf2 + self.conf3 + self.conf4 + self.conf5 + EPSILON
        #self.conf1 = self.conf1 / sum_conf
        #self.conf2 = self.conf2 / sum_conf
        #self.conf3 = self.conf3 / sum_conf
        #self.conf4 = self.conf4 / sum_conf
        #self.conf5 = self.conf5 / sum_conf

        # Prediction and loss
        self.ground_truth = self.generate_ground_gaussian(self.search_bb)

        # Raw loss for finetuning
        self.raw_loss1 = self.softmax_loss(self.ground_truth, self.rcorr1)
        self.raw_loss2 = self.softmax_loss(self.ground_truth, self.rcorr2)
        self.raw_loss3 = self.softmax_loss(self.ground_truth, self.rcorr3)
        self.raw_loss4 = self.softmax_loss(self.ground_truth, self.rcorr4)
        self.raw_loss5 = self.softmax_loss(self.ground_truth, self.rcorr5)
        self.raw_loss = self.raw_loss1 + self.raw_loss2 + self.raw_loss3 + self.raw_loss4 + self.raw_loss5

        self.raw_prediction = self.rcorr5

        # Soft loss for gating
        self.soft_loss1 = self.weighted_softmax_loss(self.ground_truth, self.rcorr1, self.conf1)
        self.soft_loss2 = self.weighted_softmax_loss(self.ground_truth, self.rcorr2, self.conf2)
        self.soft_loss3 = self.weighted_softmax_loss(self.ground_truth, self.rcorr3, self.conf3)
        self.soft_loss4 = self.weighted_softmax_loss(self.ground_truth, self.rcorr4, self.conf4)
        self.soft_loss5 = self.weighted_softmax_loss(self.ground_truth, self.rcorr5, self.conf5)
        self.soft_loss = self.soft_loss1 + self.soft_loss2 + self.soft_loss3 + self.soft_loss4 + self.soft_loss5

        # Gated loss + computational loss
        self.comp_loss = tf.reduce_mean(
                (COMP_COST_FACTOR ** 0) * self.conf1 +
                (COMP_COST_FACTOR ** 1) * self.conf2 +
                (COMP_COST_FACTOR ** 2) * self.conf3 +
                (COMP_COST_FACTOR ** 3) * self.conf4 +
                (COMP_COST_FACTOR ** 4) * self.conf5)
        #self.soft_loss = self.weighted_softmax_loss(self.ground_truth, self.soft_prediction)
        self.gated_loss = self.soft_loss + LAMBDA * self.comp_loss

        self.soft_prediction = (tf.reshape(self.conf1, [-1,1,1,1]) * self.rcorr1 +
                                tf.reshape(self.conf2, [-1,1,1,1]) * self.rcorr2 +
                                tf.reshape(self.conf3, [-1,1,1,1]) * self.rcorr3 +
                                tf.reshape(self.conf4, [-1,1,1,1]) * self.rcorr4 +
                                tf.reshape(self.conf5, [-1,1,1,1]) * self.rcorr5)

        # Note: only works for batch size 1!
        self.hard_prediction = tf.cond(self.conf1[0,0] > 0.5, lambda: self.rcorr1,
                lambda: tf.cond(self.conf2[0,0] > 0.5, lambda: self.rcorr2,
                    lambda: tf.cond(self.conf3[0,0] > 0.5, lambda: self.rcorr3,
                        lambda: tf.cond(self.conf4[0,0] > 0.5, lambda: self.rcorr4, lambda: self.rcorr5))))

        # IOU calculations
        self.raw_IOU, self.raw_pred_box, self.raw_ground_box = self.IOU(self.raw_prediction, self.key_bb, self.search_bb)
        self.raw_IOU_at_1 = self.raw_IOU[0]
        self.raw_IOU_at_5 = tf.reduce_mean(self.raw_IOU[:5])
        self.raw_IOU_full = tf.reduce_mean(self.raw_IOU)

        self.soft_IOU, self.soft_pred_box, self.soft_ground_box = self.IOU(self.soft_prediction, self.key_bb, self.search_bb)
        self.soft_IOU_at_1 = self.soft_IOU[0]
        self.soft_IOU_at_5 = tf.reduce_mean(self.soft_IOU[:5])
        self.soft_IOU_full = tf.reduce_mean(self.soft_IOU)

        # Trainers
        # TODO: experiment with LR decay?
        self.train_finetune_op = tf.train.AdamOptimizer(FINETUNE_LR).minimize(self.raw_loss, var_list=self.cnn_var_list)
        self.train_gate_op = tf.train.AdamOptimizer(GATE_LR).minimize(self.gated_loss, var_list=self.gate_var_list)

        # Tensorboard summaries
        self.raw_loss_summary = tf.summary.scalar('raw_loss', self.raw_loss)
        self.comp_loss_summary = tf.summary.scalar('comp_loss', self.comp_loss)
        self.soft_loss_summary = tf.summary.scalar('soft_loss', self.soft_loss)
        self.gated_loss_summary = tf.summary.scalar('gated_loss', self.gated_loss)

        self.xcorr1_summary = tf.summary.histogram('xcorr1', self.rcorr1)
        self.xcorr2_summary = tf.summary.histogram('xcorr2', self.rcorr2)
        self.xcorr3_summary = tf.summary.histogram('xcorr3', self.rcorr3)
        self.xcorr4_summary = tf.summary.histogram('xcorr4', self.rcorr4)
        self.xcorr5_summary = tf.summary.histogram('xcorr5', self.rcorr5)

        self.conf1_summary = tf.summary.histogram('conf1', self.conf1)
        self.conf2_summary = tf.summary.histogram('conf2', self.conf2)
        self.conf3_summary = tf.summary.histogram('conf3', self.conf3)
        self.conf4_summary = tf.summary.histogram('conf4', self.conf4)
        self.conf5_summary = tf.summary.histogram('conf5', self.conf5)

        self.raw_summary = tf.summary.merge([
            self.raw_loss_summary, self.xcorr1_summary, self.xcorr2_summary, self.xcorr3_summary,
            self.xcorr4_summary, self.xcorr5_summary])
        self.gated_summary = tf.summary.merge([
            self.comp_loss_summary, self.soft_loss_summary, self.gated_loss_summary, self.conf1_summary,
            self.conf2_summary, self.conf3_summary, self.conf4_summary, self.conf5_summary])
        self.summary_writer = tf.summary.FileWriter('logs/')

        self.data_dict = None
        self.sess.run(tf.global_variables_initializer())

    ## Training methods
    # WARN: do not mix up running train_finetune and train_gate (iter_num and summaries gets messed up)

    def train_finetune(self, key_img, search_img, key_bb, search_bb):
        if self.iter_num % 10 == 0:
            _, loss, iou1, iou5, iou25, summ = self.sess.run([
                self.train_finetune_op, self.raw_loss, self.raw_IOU_at_1, self.raw_IOU_at_5,
                self.raw_IOU_full, self.raw_summary],
                feed_dict={
                    self.key_img: key_img,
                    self.search_img: search_img,
                    self.key_bb: key_bb,
                    self.search_bb: search_bb})
            self.summary_writer.add_summary(summ, self.iter_num)
        else:
            _, loss, iou1, iou5, iou25  = self.sess.run([
                self.train_finetune_op, self.raw_loss, self.raw_IOU_at_1, self.raw_IOU_at_5, self.raw_IOU_full],
                feed_dict={
                    self.key_img: key_img,
                    self.search_img: search_img,
                    self.key_bb: key_bb,
                    self.search_bb: search_bb})

        self.iter_num += 1

        return loss, iou1, iou5, iou25

    def train_gate(self, key_img, search_img, key_bb, search_bb):
        if self.iter_num % 10 == 0:
            _, loss, iou1, iou5, iou25, summ = self.sess.run([
                self.train_gate_op, self.gated_loss, self.soft_IOU_at_1, self.soft_IOU_at_5,
                self.soft_IOU_full, self.gated_summary],
                feed_dict={
                    self.key_img: key_img,
                    self.search_img: search_img,
                    self.key_bb: key_bb,
                    self.search_bb: search_bb})
            self.summary_writer.add_summary(summ, self.iter_num)
        else:
            _, loss, iou1, iou5, iou25  = self.sess.run([
                self.train_gate_op, self.gated_loss, self.soft_IOU_at_1, self.soft_IOU_at_5, self.soft_IOU_full],
                feed_dict={
                    self.key_img: key_img,
                    self.search_img: search_img,
                    self.key_bb: key_bb,
                    self.search_bb: search_bb})

        self.iter_num += 1

        return loss, iou1, iou5, iou25

    def validation_raw(self, key_img, search_img, key_bb, search_bb):
        loss, iou1, iou5, iou25  = self.sess.run([
            self.raw_loss, self.raw_IOU_at_1, self.raw_IOU_at_5, self.raw_IOU_full],
            feed_dict={
                self.key_img: key_img,
                self.search_img: search_img,
                self.key_bb: key_bb,
                self.search_bb: search_bb})
        return loss, iou1, iou5, iou25

    def validation_gated(self, key_img, search_img, key_bb, search_bb):
        loss, iou1, iou5, iou25, c1, c2, c3, c4, c5 = self.sess.run([
            self.gated_loss, self.soft_IOU_at_1, self.soft_IOU_at_5, self.soft_IOU_full,
            self.conf1, self.conf2, self.conf3, self.conf4, self.conf5],
            feed_dict={
                self.key_img: key_img,
                self.search_img: search_img,
                self.key_bb: key_bb,
                self.search_bb: search_bb})
        return loss, iou1, iou5, iou25, np.sum(c1), np.sum(c2), np.sum(c3), np.sum(c4), np.sum(c5)

    def diagnostic_raw(self, key_img, search_img, key_bb, search_bb):
        assert(search_img.shape[0] == 1)
        cm1, cm2, cm3, cm4, cm5, con1, con2, con3, con4, con5, pred_box, ground_box = self.sess.run(
                [self.rcorr1, self.rcorr2, self.rcorr3, self.rcorr4, self.rcorr5,
                 self.conf1, self.conf2, self.conf3, self.conf4, self.conf5,
                 self.raw_pred_box, self.raw_ground_box],
                feed_dict={
                    self.key_img: key_img,
                    self.search_img: search_img,
                    self.key_bb: key_bb,
                    self.search_bb: search_bb})

        return cm1, cm2, cm3, cm4, cm5, con1, con2, con3, con4, con5, pred_box, ground_box

    def diagnostic_gated(self, key_img, search_img, key_bb, search_bb):
        assert(search_img.shape[0] == 1)
        cm1, cm2, cm3, cm4, cm5, con1, con2, con3, con4, con5, pred_box, ground_box = self.sess.run(
                [self.rcorr1, self.rcorr2, self.rcorr3, self.rcorr4, self.rcorr5,
                 self.conf1, self.conf2, self.conf3, self.conf4, self.conf5,
                 self.soft_pred_box, self.soft_ground_box],
                feed_dict={
                    self.key_img: key_img,
                    self.search_img: search_img,
                    self.key_bb: key_bb,
                    self.search_bb: search_bb})

        return cm1, cm2, cm3, cm4, cm5, con1, con2, con3, con4, con5, pred_box, ground_box

    ## Custom layers

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool_n(self, bottom, name, n):
        # Max pool by factor of 2, n times
        return tf.nn.max_pool(bottom, ksize=[1, 2**n, 2**n, 1], strides=[1, 2**n, 2**n, 1], padding='SAME', name=name)

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
            key_white = (key_pool - key_mean) / (tf.sqrt(key_var) + EPSILON)
            search_mean, search_var = tf.nn.moments(search_pool, [1,2,3], keep_dims=True)
            search_white = (search_pool - search_mean) / (tf.sqrt(search_var) + EPSILON)

            cross_corr = tf.nn.conv2d(
                    search_white,
                    tf.transpose(key_white, perm=[1,2,3,0]),
                    [1, 1, 1, 1],
                    padding='SAME')

            #corr_mean, corr_var = tf.nn.moments(cross_corr, [1,2,3], keep_dims=True)
            #corr_white = (cross_corr - corr_mean) / (tf.sqrt(corr_var) + EPSILON)
            corr_min = tf.reduce_min(cross_corr, [1,2,3], keep_dims=True)
            corr_max = tf.reduce_max(cross_corr, [1,2,3], keep_dims=True)
            corr_white = (cross_corr - corr_min) / (corr_max - corr_min + EPSILON)

            return corr_white

    # TODO: add more features
    def extract_corr_features(self, corr, corr_size, depth):
        # Kurtosis
        corr_mean, corr_var = tf.nn.moments(corr, [1,2,3])
        kurt = tf.pow(corr_mean, tf.constant(4.0)) / tf.pow(corr_var, tf.constant(2.0))
        kurt = tf.reshape(kurt, [-1, 1])    # B x 1

        # Entropy
        hist = tf.map_fn(
                lambda cmap : tf.histogram_fixed_width(     # cmap: W x H x 1
                    tf.reshape(cmap, [corr_size ** 2]),
                    value_range=[0.0, 1.0],
                    nbins=100,
                    dtype=tf.float32) + EPSILON,
                elems=corr,
                dtype=tf.float32)
        entropy = -1 * tf.reduce_sum(hist * tf.log(hist), [1], keep_dims=True)  # B x 1

        # Depth
        dep = tf.ones_like(kurt, dtype=tf.float32) * ((depth - 1.0) / 4.0)  # {0.0, 0.25, 0.5, 0.75, 1.0}

        # Top 5 peaks (raw)
        peaks, inds = tf.nn.top_k(tf.reshape(corr, [-1, corr_size ** 2]), k=5)     # B x 5

        # Distance of top peaks from the center
        # 0.5 adjusts for box center
        offset_x = tf.cast(inds % corr_size, tf.float32) - corr_size / 2 + 0.5
        offset_y = tf.cast(tf.floordiv(inds, corr_size), tf.float32) - corr_size / 2 + 0.5
        offsets = tf.square(offset_x) + tf.square(offset_y)     # B x 5

        return tf.concat([kurt, entropy, dep, peaks, offsets], axis=1)

    def confidence_layer(self, gate, name):
        with tf.variable_scope(name):
            input_dim = gate.shape[1].value
            weights, bias = self.get_gate_var(name, input_dim)
            muled = tf.matmul(gate, weights)
            output = tf.sigmoid(tf.nn.bias_add(tf.matmul(gate, weights), bias))
            return output   # B x 1
            #return tf.reshape(output, [-1, 1, 1, 1])    # For scalar multiplication later

    def generate_ground_gaussian(self, bbs):
        def bb_to_gaussian(bb):
            x = tf.range(-SEARCH_FRAME_SIZE / 2, limit=SEARCH_FRAME_SIZE / 2, dtype=tf.float32)
            xs, ys = tf.meshgrid(x, x)
            gaus = GAUSSIAN_AMP * tf.exp(-((xs - bb[0])**2 / (2 * GAUSSIAN_VAR) + (ys - bb[1])**2 / (2 * GAUSSIAN_VAR)))
            return gaus     # [SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE]

        grounds = tf.map_fn(bb_to_gaussian, elems=bbs, dtype=tf.float32, back_prop=False)
        return tf.reshape(grounds, [-1, SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE, 1])

    def softmax_loss(self, ground_truth, prediction):
        shape = ground_truth.get_shape().as_list()  # [None, 256, 256, 1]
        flattened_shape = [-1, shape[1] * shape[2] * shape[3]]

        normalized_ground_truth = ground_truth / tf.reduce_sum(ground_truth, axis=[1,2,3], keep_dims=True)

        reshaped_ground_truth = tf.reshape(normalized_ground_truth, flattened_shape)
        reshaped_prediction = tf.reshape(prediction, flattened_shape)

        softmax_loss = tf.nn.softmax_cross_entropy_with_logits(logits=reshaped_prediction, labels=reshaped_ground_truth)
        loss = tf.reduce_mean(softmax_loss)

        return loss

    def weighted_softmax_loss(self, ground_truth, prediction, weight):
        shape = ground_truth.get_shape().as_list()  # [None, 256, 256, 1]
        flattened_shape = [-1, shape[1] * shape[2] * shape[3]]

        normalized_ground_truth = ground_truth / tf.reduce_sum(ground_truth, axis=[1,2,3], keep_dims=True)

        reshaped_ground_truth = tf.reshape(normalized_ground_truth, flattened_shape)
        reshaped_prediction = tf.reshape(prediction, flattened_shape)
        softmax_loss = tf.nn.softmax_cross_entropy_with_logits(logits=reshaped_prediction, labels=reshaped_ground_truth)
        weighted_loss = tf.reshape(weight, [-1]) * softmax_loss
        loss = tf.reduce_mean(weighted_loss)

        return loss

    def IOU(self, prediction, key_bb, search_bb):
        shape = prediction.get_shape().as_list()    # [None, SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE, 1]
        assert(shape[3] == 1)
        offset = tf.argmax(tf.reshape(prediction, [-1, shape[1] * shape[2]]), axis=1)
        offset_x = tf.cast(offset % SEARCH_FRAME_SIZE, tf.float32)
        offset_y = tf.cast(tf.floordiv(offset, SEARCH_FRAME_SIZE), tf.float32)
        pred_block_size = 2 ** 4    # 256 / 16, centers pred box on max block
        self.offset = offset
        self.offset_x = offset_x
        self.offset_y = offset_y

        # top left + bottom right coords for prediction
        boxA_x1 = offset_x - key_bb[0] / 2 + pred_block_size / 2
        boxA_y1 = offset_y - key_bb[1] / 2 + pred_block_size / 2
        boxA_x2 = offset_x + key_bb[0] / 2 + pred_block_size / 2
        boxA_y2 = offset_y + key_bb[1] / 2 + pred_block_size / 2

        # top left + bottom right coords for ground truth
        boxB_x1 = search_bb[:, 0] - search_bb[:, 2] / 2 + SEARCH_FRAME_SIZE / 2
        boxB_y1 = search_bb[:, 1] - search_bb[:, 3] / 2 + SEARCH_FRAME_SIZE / 2
        boxB_x2 = search_bb[:, 0] + search_bb[:, 2] / 2 + SEARCH_FRAME_SIZE / 2
        boxB_y2 = search_bb[:, 1] + search_bb[:, 3] / 2 + SEARCH_FRAME_SIZE / 2

        # interior bb
        inter_x1 = tf.maximum(boxA_x1, boxB_x1)
        inter_y1 = tf.maximum(boxA_y1, boxB_y1)
        inter_x2 = tf.minimum(boxA_x2, boxB_x2)
        inter_y2 = tf.minimum(boxA_y2, boxB_y2)

        inter_area = tf.where(
                tf.logical_and(inter_x1 < inter_x2, inter_y1 < inter_y2),    # true iff intersecting boxes
                (inter_x2 - inter_x1) * (inter_y2 - inter_y1),
                tf.zeros_like(offset_x))    # non-intersecting boxes, area = 0

        boxA_area = (boxA_x2 - boxA_x1) * (boxA_y2 - boxA_y1)
        boxB_area = (boxB_x2 - boxB_x1) * (boxB_y2 - boxB_y1)

        iou = inter_area / (boxA_area + boxB_area - inter_area)

        return iou, (boxA_x1, boxA_y1, boxA_x2, boxA_y2), (boxB_x1, boxB_y1, boxB_x2, boxB_y2)

    def non_max_suppression(self, input, window_size):
        # input = B x W x H x C
        # NOTE: if input is negative, suppressed values are larger than max value
        pooled = tf.nn.max_pool(input, ksize=[1, window_size, window_size, 1], strides=[1, 1, 1, 1], padding='SAME')
        output = tf.where(tf.equal(input, pooled), input, tf.zeros_like(input))

        # output = B x W x H x C
        return output

    ## Variable handlers

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        filters = self.get_var(name, [filter_size, filter_size, in_channels, out_channels], 0, name + "_filters")
        biases = self.get_var(name, [out_channels], 1, name + "_biases")

        self.cnn_var_list.append(filters)
        self.cnn_var_list.append(biases)

        return filters, biases

    def get_gate_var(self, name, input_dim):
        weights = self.get_var(name, [input_dim, 1], 0, name + '_weights')
        bias = self.get_var(name, [1], 1, name + '_bias')

        self.gate_var_list.append(weights)
        self.gate_var_list.append(bias)

        return weights, bias

    def get_var(self, name, shape, idx, var_name):
        try:
            if self.data_dict is not None and name in self.data_dict:
                init = tf.constant_initializer(self.data_dict[name][idx])
                var = tf.get_variable(var_name, shape=shape, initializer=init)
                print 'Loaded Variable %s in %s from file' % (var_name, tf.get_variable_scope().name)
            else:
                init = tf.truncated_normal_initializer(0.0, 0.001)
                var = tf.get_variable(var_name, shape=shape, initializer=init)
                print 'Initialized Variable %s in %s' % (var_name, tf.get_variable_scope().name)
        except ValueError:
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                var = tf.get_variable(var_name)
                print 'Reused Variable %s in %s' % (var_name, tf.get_variable_scope().name)

        self.var_dict[(name, idx)] = var
        return var

    def save_npy(self, npy_path="./vgg19-save.npy"):
        data_dict = {}

        for (name, idx), var in self.var_dict.items():
            var_out = self.sess.run(var)
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
