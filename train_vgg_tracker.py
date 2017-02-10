"""
Train VGG19-tracker
"""

import os

import tensorflow as tf
import numpy as np
from PIL import Image

import vgg19_tracker as vgg19
from CONSTANTS import *

#img1 = utils.load_image("./test_data/tiger.jpeg")
#img1_true_result = [1 if i == 292 else 0 for i in xrange(1000)]  # 1-hot result for tiger
#
#batch1 = img1.reshape((1, 224, 224, 3))
key = np.array(Image.open(os.path.join(PROCESSED_DIR, 'fish1', 'key-00000001', 'key-00000001.png'))).reshape([1, KEY_FRAME_SIZE, KEY_FRAME_SIZE, 3])
search = np.array(Image.open(os.path.join(PROCESSED_DIR, 'fish1', 'key-00000001', 'search-00000060.png'))).reshape([1, SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE, 3])

with tf.device('/cpu:0'):
    sess = tf.Session()

    #images = tf.placeholder(tf.float32, [1, 224, 224, 3])
    #true_out = tf.placeholder(tf.float32, [1, 1000])
    #train_mode = tf.placeholder(tf.bool)
    key_image = tf.placeholder(tf.float32, [1, KEY_FRAME_SIZE, KEY_FRAME_SIZE, 3])
    search_image = tf.placeholder(tf.float32, [None, SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE, 3])

    vgg = vgg19.Vgg19('./vgg19.npy')
    vgg.build(key_image, search_image)

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print vgg.get_var_count()

    sess.run(tf.initialize_all_variables())

    # test classification
    corr_map = sess.run(vgg.cross_corr5, feed_dict={key_image: key, search_image: search})
    corr_map = corr_map.reshape((corr_map.shape[1], corr_map.shape[2]))
    corr_map = (corr_map - np.min(corr_map))
    corr_map = corr_map / (np.max(corr_map) + 0.0001)
    im = Image.fromarray(np.uint8(corr_map * 255))
    im.save('corr_map.png')
    #utils.print_prob(prob[0], './synset.txt')

    ## simple 1-step training
    #cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
    #train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
    #sess.run(train, feed_dict={images: batch1, true_out: [img1_true_result], train_mode: True})

    ## test classification again, should have a higher probability about tiger
    #prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
    #utils.print_prob(prob[0], './synset.txt')

    ## test save
    #vgg.save_npy(sess, './test-save.npy')
