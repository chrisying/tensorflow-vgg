"""
Train VGG19-tracker
"""

import os

import tensorflow as tf
import numpy as np
from PIL import Image

import vgg19_tracker as vgg19
from CONSTANTS import *

key = np.array(Image.open(os.path.join(PROCESSED_DIR, 'fish1', 'key-00000091', 'key-00000091.png'))).reshape([1, KEY_FRAME_SIZE, KEY_FRAME_SIZE, 3])
search = np.array(Image.open(os.path.join(PROCESSED_DIR, 'fish1', 'key-00000091', 'search-00000107.png'))).reshape([1, SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE, 3])

def load_batch(category, key_frame):
    data_dir = os.path.join(PROCESSED_DIR, category, key_frame)
    key_frame = Image.open(os.path.join(data_dir, key_frame + '.png'))
    key_data = np.array(key_frame).reshape([1, KEY_FRAME_SIZE, KEY_FRAME_SIZE, 3])

    with open(os.path.join(data_dir, 'groundtruth.txt')) as f:
        key_line = f.readline()
        assert key_line[:12] == key_frame
        x, y, w, h, s = map(float, key_line[14:].split())  # Unused right now
        s_idx = 0
        search_batch = np.zeros([1, SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE, 3])
        for line in f.xreadlines():
            search_name = line[:15]
            search_frame = Image.open(os.path.join(data_dir, search_name))
            search_batch[s_idx, :, :, :] = np.array(search_frame).reshape([1, SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE, 3])

            offset_x, offset_y = map(float, search_line[17:].split())
            # TODO: create ground truth here

    return key_data, search_batch


def save_corr_map(corr_map, filename):
    corr_map = corr_map.reshape((corr_map.shape[1], corr_map.shape[2]))
    corr_map = (corr_map - np.min(corr_map))
    corr_map = corr_map / (np.max(corr_map) + 0.0001)
    im = Image.fromarray(np.uint8(corr_map * 255))
    im.save(filename)

def main():
#    with tf.device('/cpu:0'):
        sess = tf.Session()

        key_image = tf.placeholder(tf.float32, [1, KEY_FRAME_SIZE, KEY_FRAME_SIZE, 3])
        search_image = tf.placeholder(tf.float32, [None, SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE, 3])

        vgg = vgg19.Vgg19('./vgg19.npy')
        vgg.build(key_image, search_image)

        # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
        print vgg.get_var_count()

        sess.run(tf.initialize_all_variables())

        # test classification
        [cm1, cm2, cm3, cm4, cm5] = sess.run(
                [vgg.corr1, vgg.corr2, vgg.corr3, vgg.corr4, vgg.corr5],
                feed_dict={key_image: key, search_image: search})
        save_corr_map(cm1, 'corr_map1.png')
        save_corr_map(cm2, 'corr_map2.png')
        save_corr_map(cm3, 'corr_map3.png')
        save_corr_map(cm4, 'corr_map4.png')
        save_corr_map(cm5, 'corr_map5.png')

        ## simple 1-step training
        #cost = tf.reduce_sum((vgg.prob - true_out) ** 2)
        #train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)
        #sess.run(train, feed_dict={images: batch1, true_out: [img1_true_result], train_mode: True})

        ## test classification again, should have a higher probability about tiger
        #prob = sess.run(vgg.prob, feed_dict={images: batch1, train_mode: False})
        #utils.print_prob(prob[0], './synset.txt')

        ## test save
        #vgg.save_npy(sess, './test-save.npy')

if __name__ == '__main__':
    main()
