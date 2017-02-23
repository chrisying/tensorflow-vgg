"""
Train VGG19-tracker
"""

import os
import time

import tensorflow as tf
import numpy as np
from PIL import Image#, ImageDraw

import vgg19_tracker as vgg19
from CONSTANTS import *

# Debugging inputs
debug_key = np.array(Image.open(os.path.join(PROCESSED_DIR, 'fish1', 'key-00000091', 'key-00000091.png'))).reshape([1, KEY_FRAME_SIZE, KEY_FRAME_SIZE, 3])
debug_search = np.array(Image.open(os.path.join(PROCESSED_DIR, 'fish1', 'key-00000091', 'search-00000107.png'))).reshape([1, SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE, 3])
debug_ground = np.zeros([1, SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE, 1])

def load_batch(category, key_name):
    data_dir = os.path.join(PROCESSED_DIR, category, key_name)
    key_frame = Image.open(os.path.join(data_dir, key_name + '.png'))
    key_data = np.array(key_frame).reshape([1, KEY_FRAME_SIZE, KEY_FRAME_SIZE, 3])

    with open(os.path.join(data_dir, 'groundtruth.txt')) as f:
        key_line = f.readline()
        assert key_line[:12] == key_name
        x, y, w, h, s = map(float, key_line[14:].split())  # Unused right now
        s_idx = 0
        search_batch = np.zeros([MAX_FRAME_GAP, SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE, 3])
        ground_truth = np.full([MAX_FRAME_GAP, SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE, 1], -1)
        for search_line in f.xreadlines():
            search_name = search_line[:15]
            search_frame = Image.open(os.path.join(data_dir, search_name + '.png'))
            search_batch[s_idx, :, :, :] = np.array(search_frame).reshape([1, SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE, 3])

            # Add circle of radium TRUTH_RADIUS of +1 to ground truth using mask
            offset_x, offset_y = map(float, search_line[17:].split())
            offset_x_full, offset_y_full = offset_x * s, offset_y * s
            true_center_x, true_center_y = SEARCH_FRAME_SIZE / 2 + offset_x_full, SEARCH_FRAME_SIZE /2 + offset_y_full
            og_y, og_x = np.ogrid[-true_center_y:SEARCH_FRAME_SIZE-true_center_y, -true_center_x:SEARCH_FRAME_SIZE-true_center_x]
            mask = og_x * og_x + og_y * og_y <= TRUTH_RADIUS**2
            if mask.shape[0] != SEARCH_FRAME_SIZE or mask.shape[1] != SEARCH_FRAME_SIZE:
                print '-----WARNING!-----'
                print 'mask size mismatch'
                print category, key_name, search_line
                print true_center_x, true_center_y
                print mask.shape
                print '------------------'
            ground_truth[s_idx, :, :, :][mask] = 1

            s_idx += 1

            #dr = ImageDraw.Draw(search_frame)
            #dr.rectangle((true_center_x - 10, true_center_y - 10, true_center_x + 10, true_center_y + 10), outline='red')
            #search_frame.save('test_offset.png')

    return key_data, search_batch, ground_truth


def run_validation(sess, vgg, k, s, g):
    test_loss_sum = 0.0
    num_samples = 0
    for category in TEST_CATS:
        print 'Running validation on %s' % category
        data_dir = os.path.join(PROCESSED_DIR, category)
        key_names = os.listdir(data_dir)
        for key_name in key_names:
            print 'Running validation on %s' % key_name
            key, search, ground = load_batch(category, key_name)
            loss = sess.run(vgg.loss, feed_dict={k: key, s: search, g: ground})
            test_loss_sum += loss
            num_samples += 1
    return test_loss_sum / num_samples


def convert_corr_map(corr_map):
    corr_map = corr_map.reshape((corr_map.shape[1], corr_map.shape[2]))
    corr_map = (corr_map - np.min(corr_map))
    corr_map = corr_map / (np.max(corr_map) + 0.0001)
    im = Image.fromarray(np.uint8(corr_map * 255))
    im = im.resize((SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE))
    return im

def diagnostic_corr_maps(sess, vgg, name, k, s, g):
    [cm1, cm2, cm3, cm4, cm5] = sess.run(
            [vgg.corr1, vgg.corr2, vgg.corr3, vgg.corr4, vgg.corr5],
            feed_dict={k: debug_key, s: debug_search, g: debug_ground})
    c1 = convert_corr_map(cm1)
    c2 = convert_corr_map(cm2)
    c3 = convert_corr_map(cm3)
    c4 = convert_corr_map(cm4)
    c5 = convert_corr_map(cm5)

    new_im = Image.new(c1.mode, (SEARCH_FRAME_SIZE * 5, SEARCH_FRAME_SIZE))
    for i, ci in enumerate([c1,c2,c3,c4,c5]):
        new_im.paste(ci, (i * SEARCH_FRAME_SIZE, 0))

    new_im.save(name)


def main():
#    with tf.device('/cpu:0'):
        sess = tf.Session()

        key_image = tf.placeholder(tf.float32, [1, KEY_FRAME_SIZE, KEY_FRAME_SIZE, 3])
        search_image = tf.placeholder(tf.float32, [None, SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE, 3])
        ground_truth = tf.placeholder(tf.float32, [None, SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE, 1])

        vgg = vgg19.Vgg19('./vgg19.npy')
        vgg.build(key_image, search_image, ground_truth)

        # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
        print vgg.get_var_count()

        train = tf.train.AdamOptimizer(0.0001).minimize(vgg.loss)
        sess.run(tf.global_variables_initializer())

        diagnostic_corr_maps(sess, vgg, 'initial_corr_maps.png', key_image, search_image, ground_truth)
        loss, rcorr1, loss1 = sess.run([vgg.loss, vgg.rcorr1, vgg.loss1],
                feed_dict={key_image: debug_key, search_image: debug_search, grouth_truth: debug_ground})
        print 'Trainable variables:'
        print map(lambda x:x.name, tf.trainable_variables())

        #valid_loss = run_validation(sess, vgg, key_image, search_image, ground_truth)
        #print '[VALID] Initial validation loss: %.5f' % valid_loss

        # TODO: use QueueRunner to optimize file loading on CPU
        print 'Starting training'
        start = time.time()
        for epoch in range(TRAIN_EPOCHS):
            epoch_loss_sum = 0.0
            for train_cat in TRAIN_CATS:
                cat_dir = os.path.join(PROCESSED_DIR, train_cat)
                key_names = os.listdir(cat_dir)
                cat_loss_sum = 0.0
                for key_name in key_names:
                    # ordering shouldn't matter
                    key, search, ground = load_batch(train_cat, key_name)
                    _, loss = sess.run([train, vgg.loss],
                            feed_dict={key_image: key, search_image: search, ground_truth: ground})
                    cat_loss_sum += loss
                    print '[TRAIN] Batch loss on %s %s: %.5f' % (train_cat, key_name, loss)
                cat_loss = cat_loss_sum / len(key_names)
                epoch_loss_sum += cat_loss
                print '[TRAIN] Category loss on %s: %.5f' % (train_cat, loss)
            epoch_loss = epoch_loss_sum / len(TRAIN_CATS)
            print '[TRAIN] Epoch loss on %d: %.5f' % (epoch, epoch_loss)
            valid_loss = run_validation(sess, vgg, key_image, search_image, ground_truth)
            print '[VALID] Validation loss after epoch %d: %.5f' % (epoch, valid_loss)
        dur = time.time() - start
        print 'Training completed in %d seconds' % dur

        diagnostic_corr_maps(sess, vgg, 'final_corr_maps.png', key_image, search_image, grouth_truth)

        # save model
        vgg.save_npy(sess, './trained_model_%s.npy' % str(int(time.time())))

if __name__ == '__main__':
    main()
