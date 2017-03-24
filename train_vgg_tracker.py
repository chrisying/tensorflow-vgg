"""
Train VGG19-tracker
"""

import os
import time
import sys

import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.cm as cm

import vgg19_tracker as vgg19
from CONSTANTS import *

# For resolving some floating point precision errors
EPSILON = 1e-5

def load_batch(category, key_name):
    data_dir = os.path.join(PROCESSED_DIR, category, key_name)
    key_frame = Image.open(os.path.join(data_dir, key_name + '.png'))
    key_data = np.array(key_frame).reshape([1, KEY_FRAME_SIZE, KEY_FRAME_SIZE, 3])

    with open(os.path.join(data_dir, 'groundtruth.txt')) as f:
        key_line = f.readline()
        assert key_line[:12] == key_name
        x, y, w, h, s = map(float, key_line[14:].split())

        search_lines = f.readlines()
        num_frames = len(search_lines)

        if num_frames < BATCH_SIZE:
            #print 'Skipping %s %s because batch too small' % (category, key_name)
            return None, None, None, None

        search_batch = np.zeros([BATCH_SIZE, SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE, 3], dtype=np.uint8)
        search_bb = np.zeros([BATCH_SIZE, 4])

        for s_idx in range(BATCH_SIZE):
            search_line = search_lines[s_idx]
            search_name = search_line[:15]
            search_frame = Image.open(os.path.join(data_dir, search_name + '.png'))
            search_batch[s_idx, :, :, :] = np.array(search_frame).reshape([1, SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE, 3])

            offset_x, offset_y, s_width, s_height = map(float, search_line[17:].split())

            search_bb[s_idx, 0] = offset_x * s
            search_bb[s_idx, 1] = offset_y * s
            search_bb[s_idx, 2] = s_width * s
            search_bb[s_idx, 3] = s_height * s

            # Add circle of radium TRUTH_RADIUS of +1 to ground truth using mask
            #offset_x_full, offset_y_full = offset_x * s, offset_y * s
            #true_center_x, true_center_y = SEARCH_FRAME_SIZE / 2 + offset_x_full, SEARCH_FRAME_SIZE /2 + offset_y_full
            #og_y, og_x = np.ogrid[-true_center_y:SEARCH_FRAME_SIZE-true_center_y-EPSILON, -true_center_x:SEARCH_FRAME_SIZE-true_center_x-EPSILON]
            #mask = og_x * og_x + og_y * og_y <= TRUTH_RADIUS**2
            #ground_truth[s_idx, :, :, :][mask] = 1

    return key_data, search_batch, np.array([w*s, h*s]), search_bb

def run_validation(sess, vgg):
    test_loss_sum = 0.0
    iou1_sum = 0.0
    iou5_sum = 0.0
    iou20_sum = 0.0
    num_samples = 0
    for category in TEST_CATS:
        #print 'Running validation on %s' % category
        data_dir = os.path.join(PROCESSED_DIR, category)
        key_names = os.listdir(data_dir)
        for key_name in key_names:
            #print 'Running validation on %s' % key_name
            a = load_batch(category, key_name)
            key, search, key_bb, search_bb = load_batch(category, key_name)
            if key is None:
                continue

            gt, pred, loss, iou1, iou5, iou20 = sess.run([vgg.gt, vgg.pred, vgg.raw_loss, vgg.IOU_at_1, vgg.IOU_at_5, vgg.IOU_full],
            #iou, loss, iou1, iou5, iou20 = sess.run([vgg.IOU, vgg.raw_loss, vgg.IOU_at_1, vgg.IOU_at_5, vgg.IOU_full],
                    feed_dict={vgg.key_img: key,
                               vgg.search_img: search,
                               vgg.key_bb: key_bb,
                               vgg.search_bb: search_bb})
            print gt
            print np.sum(gt)
            print np.max(gt)
            print np.min(gt)
            print '-'
            print pred
            print np.sum(pred)
            print np.max(pred)
            print np.min(pred)
            test_loss_sum += BATCH_SIZE * loss
            iou1_sum += BATCH_SIZE * iou1
            iou5_sum += BATCH_SIZE * iou5
            iou20_sum += BATCH_SIZE * iou20
            num_samples += BATCH_SIZE

    assert(num_samples > 0)
    #print '[VALID] Samples considered: %d' % num_samples
    return test_loss_sum / num_samples, iou1_sum / num_samples, iou5_sum / num_samples, iou20_sum / num_samples

def convert_corr_map(corr_map):
    corr_map = corr_map.reshape((corr_map.shape[1], corr_map.shape[2]))
    corr_map = (corr_map - np.min(corr_map))
    corr_map = corr_map / (np.max(corr_map) + 0.0001)
    im = Image.fromarray(np.uint8(cm.viridis(corr_map) * 255))
    #im = im.resize((SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE))
    return im

def visualize_corr_maps(sess, vgg, name, key_img, search_img, key_bb, search_bb):
    # Expects key_img, search_img, ground_img to be batch size 1
    [cm1, cm2, cm3, cm4, cm5, con1, con2, con3, con4, con5, pred_box, ground_box] = sess.run(
            [vgg.rcorr1, vgg.rcorr2, vgg.rcorr3, vgg.rcorr4, vgg.rcorr5,
             vgg.conf1, vgg.conf2, vgg.conf3, vgg.conf4, vgg.conf5,
             vgg.pred_box, vgg.ground_box],
            feed_dict={
                vgg.key_img: key_img,
                vgg.search_img: search_img,
                vgg.key_bb: key_bb,
                vgg.search_bb: search_bb})

    c1 = convert_corr_map(cm1)
    c2 = convert_corr_map(cm2)
    c3 = convert_corr_map(cm3)
    c4 = convert_corr_map(cm4)
    c5 = convert_corr_map(cm5)

    PAD = 2

    new_im = Image.new('RGB', ((SEARCH_FRAME_SIZE+2*PAD) * 7, (SEARCH_FRAME_SIZE+2*PAD)), (128,128,128))
    d = ImageDraw.Draw(new_im)

    key_img = Image.fromarray(key_img.reshape((KEY_FRAME_SIZE, KEY_FRAME_SIZE, 3)))
    dk = ImageDraw.Draw(key_img)
    dk.rectangle([KEY_FRAME_SIZE / 2 - key_bb[0] / 2,
                  KEY_FRAME_SIZE / 2 - key_bb[1] / 2,
                  KEY_FRAME_SIZE / 2 + key_bb[0] / 2,
                  KEY_FRAME_SIZE / 2 + key_bb[1] / 2],
                 outline='red')
    new_im.paste(key_img, (KEY_FRAME_SIZE/2+PAD, KEY_FRAME_SIZE/2+PAD))

    #red = np.zeros((SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE, 3), dtype='uint8')
    #red[:,:,2] = 255
    #combined_search = np.where(ground_img<0, search_img, red).reshape((SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE, 3))
    #new_im.paste(Image.fromarray(combined_search.astype('uint8')), (SEARCH_FRAME_SIZE+2*PAD + PAD, PAD))
    search_img = Image.fromarray(search_img.reshape((SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE, 3)))
    ds = ImageDraw.Draw(search_img)
    #ds.rectangle([SEARCH_FRAME_SIZE / 2 + search_bb[0, 0] - search_bb[0, 2] / 2,
    #              SEARCH_FRAME_SIZE / 2 + search_bb[0, 1] - search_bb[0, 3] / 2,
    #              SEARCH_FRAME_SIZE / 2 + search_bb[0, 0] + search_bb[0, 2] / 2,
    #              SEARCH_FRAME_SIZE / 2 + search_bb[0, 1] + search_bb[0, 3] / 2],
    #            outline='green')
    ds.rectangle([ground_box[0][0], ground_box[1][0], ground_box[2][0], ground_box[3][0]], outline='green')
    ds.rectangle([pred_box[0][0], pred_box[1][0], pred_box[2][0], pred_box[3][0]], outline='red')
    new_im.paste(search_img, (SEARCH_FRAME_SIZE+2*PAD + PAD, PAD))

    for i, ci in enumerate([c1,c2,c3,c4,c5]):
        new_im.paste(ci, ((i+2) * (SEARCH_FRAME_SIZE+2*PAD) + PAD, PAD))

    fnt = ImageFont.truetype('RobotoMono-Regular.ttf', 16)
    for i, ct in enumerate([con1, con2, con3, con4, con5]):
        d.text(((i+2) * (SEARCH_FRAME_SIZE+2*PAD) + PAD + 10, PAD + 10), "%.5f" % ct.reshape([1])[0], font=fnt, fill=(255, 0, 0, 255))

    new_im.save(name)

def diagnostic_corr_maps(sess, vgg, name):
    debug_key, debug_search, debug_key_bb, debug_search_bb = load_batch('basketballmini', 'key-00000071')
    assert(debug_key is not None)
    #debug_search = debug_search[15:16,:,:,:]
    #debug_search_bb = debug_search_bb[15:16,:]
    debug_search = debug_search[0:1,:,:,:]
    debug_search_bb = debug_search_bb[0:1,:]

    gt = sess.run(vgg.ground_truth, feed_dict={vgg.search_bb: debug_search_bb})
    img = Image.fromarray((gt[0,:,:,0] * 255).astype(np.uint8))
    img.save('gt.png')

    visualize_corr_maps(sess, vgg, 'basketball_' + name, debug_key, debug_search, debug_key_bb, debug_search_bb)

def main():
#    with tf.device('/cpu:0'):
        sess = tf.Session()

        vgg = vgg19.Vgg19('./vgg19.npy')
        vgg.build()

        # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
        print vgg.get_var_count()

        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 1e-1
        decay_steps = 1000
        decay_rate = 0.95

        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, decay_steps, decay_rate, staircase=True)
        train_finetune = tf.train.AdamOptimizer(learning_rate).minimize(vgg.raw_loss, var_list=vgg.cnn_var_list, global_step=global_step)
        #train_gate = tf.train.AdamOptimizer(1e-5).minimize(vgg.gated_loss, var_list=vgg.gate_var_list)
        sess.run(tf.global_variables_initializer())

        print 'Trainable variables (finetune):'
        print map(lambda x:x.name, vgg.cnn_var_list)
        print 'Trainable variables (gate):'
        print map(lambda x:x.name, vgg.gate_var_list)

        diagnostic_corr_maps(sess, vgg, 'initial.png')

        valid_loss, iou1, iou5, iou20 = run_validation(sess, vgg)
        print '[VALID] Initial validation loss: %.5f, IOU@1: %.5f, IOU@5: %.5f, IOU@20: %.5f' % (valid_loss, iou1, iou5, iou20)

        # TODO: use QueueRunner to optimize file loading on CPU
        print 'Starting training'
        start = time.time()
        for epoch in range(TRAIN_EPOCHS):
            epoch_loss_sum = 0.0
            num_samples = 0
            for train_cat in TRAIN_CATS:
                cat_dir = os.path.join(PROCESSED_DIR, train_cat)
                key_names = os.listdir(cat_dir)
                for key_name in key_names:
                    # ordering shouldn't matter
                    key, search, key_bb, search_bb = load_batch(train_cat, key_name)
                    if key is None:
                        continue

                    _, loss = sess.run([train_finetune, vgg.raw_loss],
                            feed_dict={
                                vgg.key_img: key,
                                vgg.search_img: search,
                                vgg.key_bb: key_bb,
                                vgg.search_bb: search_bb})

                    print '[TRAIN] Batch loss %s %s: %.5f' % (train_cat, key_name, loss)
                    epoch_loss_sum += BATCH_SIZE * loss
                    num_samples += BATCH_SIZE

            epoch_loss = epoch_loss_sum / num_samples
            #print '[TRAIN] Samples considered: %d' % num_samples
            print '[TRAIN] Epoch %d loss: %.5f' % (epoch+1, epoch_loss)

            valid_loss, iou1, iou5, iou20 = run_validation(sess, vgg)
            print '[VALID] Epoch %d, validation loss: %.5f, IOU@1: %.5f, IOU@5: %.5f, IOU@20: %.5f' % (epoch+1, valid_loss, iou1, iou5, iou20)

            # checkpointing
            diagnostic_corr_maps(sess, vgg, 'epoch_%s.png' % str(epoch+1).zfill(3))
            #vgg.save_npy(sess, './trained_model_epoch_%d_%s.npy' % (epoch+1, str(int(time.time()))))

        dur = time.time() - start
        print 'Training completed in %d seconds' % dur

        # save model
        #vgg.save_npy(sess, './trained_model_%s.npy' % str(int(time.time())))
        sess.close()

if __name__ == '__main__':
    main()
