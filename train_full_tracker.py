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

import full_tracker as vgg19
from CONSTANTS import *

# Uncomment the MODE to run (finetune = train VGG weights, gating = train gate vars)
MODE = 'finetune'
#MODE = 'gating'

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

    return key_data, search_batch, np.array([w*s, h*s]), search_bb

def run_validation(vgg):
    test_loss_sum = 0.0
    iou1_sum = 0.0
    iou5_sum = 0.0
    iou25_sum = 0.0
    num_samples = 0
    c1_sum = 0.0
    c2_sum = 0.0
    c3_sum = 0.0
    c4_sum = 0.0
    c5_sum = 0.0
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

            #for i in range(BATCH_SIZE):
            #    loss, iou1, iou5, iou25 = vgg.sess.run([vgg.raw_loss, vgg.raw_IOU_at_1, vgg.raw_IOU_at_5, vgg.raw_IOU_full],
            #            feed_dict={vgg.key_img: key,
            #                       vgg.search_img: search[i:i+1, :, :, :],
            #                       vgg.key_bb: key_bb,
            #                       vgg.search_bb: search_bb[i:i+1, :]})
            #    print 'frame %d: %.5f' % (i, loss)

            if MODE == 'finetune':
                loss, iou1, iou5, iou25 = vgg.validation_raw(key, search, key_bb, search_bb)
            else:
                assert(MODE == 'gating')
                loss, iou1, iou5, iou25, c1, c2, c3, c4, c5 = vgg.validation_gated(key, search, key_bb, search_bb)
                c1_sum += c1
                c2_sum += c2
                c3_sum += c3
                c4_sum += c4
                c5_sum += c5

            test_loss_sum += BATCH_SIZE * loss
            iou1_sum += BATCH_SIZE * iou1
            iou5_sum += BATCH_SIZE * iou5
            iou25_sum += BATCH_SIZE * iou25
            num_samples += BATCH_SIZE

    if MODE == 'gating':
        print '[VALID] confidence means: %.5f | %.5f | %.5f | %.5f | %.5f' % (c1_sum / num_samples, c2_sum / num_samples, c3_sum / num_samples, c4_sum / num_samples, c5_sum / num_samples)

    assert(num_samples > 0)
    #print '[VALID] Samples considered: %d' % num_samples
    return test_loss_sum / num_samples, iou1_sum / num_samples, iou5_sum / num_samples, iou25_sum / num_samples

def convert_corr_map(corr_map):
    corr_map = corr_map.reshape((corr_map.shape[1], corr_map.shape[2]))
    corr_map = (corr_map - np.min(corr_map))
    corr_map = corr_map / (np.max(corr_map) + 0.0001)
    im = Image.fromarray(np.uint8(cm.viridis(corr_map) * 255))
    #im = im.resize((SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE))
    return im

def visualize_corr_maps(vgg, name, key_img, search_img, key_bb, search_bb):
    # Expects key_img, search_img, ground_img to be batch size 1
    if MODE == 'finetune':
        cm1, cm2, cm3, cm4, cm5, con1, con2, con3, con4, con5, pred_box, ground_box = vgg.diagnostic_raw(
                key_img, search_img, key_bb, search_bb)
    else:
        assert(MODE == 'gating')
        cm1, cm2, cm3, cm4, cm5, con1, con2, con3, con4, con5, pred_box, ground_box = vgg.diagnostic_gated(
                key_img, search_img, key_bb, search_bb)

    c1 = convert_corr_map(cm1)
    c2 = convert_corr_map(cm2)
    c3 = convert_corr_map(cm3)
    c4 = convert_corr_map(cm4)
    c5 = convert_corr_map(cm5)

    PAD = 2

    new_im = Image.new('RGB', ((SEARCH_FRAME_SIZE+2*PAD) * 7, (SEARCH_FRAME_SIZE+2*PAD)), (255,255,255))
    d = ImageDraw.Draw(new_im)

    key_img = Image.fromarray(key_img.reshape((KEY_FRAME_SIZE, KEY_FRAME_SIZE, 3)))
    dk = ImageDraw.Draw(key_img)
    dk.rectangle([KEY_FRAME_SIZE / 2 - key_bb[0] / 2,
                  KEY_FRAME_SIZE / 2 - key_bb[1] / 2,
                  KEY_FRAME_SIZE / 2 + key_bb[0] / 2,
                  KEY_FRAME_SIZE / 2 + key_bb[1] / 2],
                 outline='green')
    new_im.paste(key_img, (KEY_FRAME_SIZE/2+PAD, KEY_FRAME_SIZE/2+PAD))

    search_img = Image.fromarray(search_img.reshape((SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE, 3)))
    ds = ImageDraw.Draw(search_img)
    ds.rectangle([ground_box[0][0], ground_box[1][0], ground_box[2][0], ground_box[3][0]], outline='green')
    ds.rectangle([pred_box[0][0], pred_box[1][0], pred_box[2][0], pred_box[3][0]], outline='red')
    new_im.paste(search_img, (SEARCH_FRAME_SIZE+2*PAD + PAD, PAD))

    for i, ci in enumerate([c1,c2,c3,c4,c5]):
        new_im.paste(ci, ((i+2) * (SEARCH_FRAME_SIZE+2*PAD) + PAD, PAD))

    if MODE == 'gating':
        fnt = ImageFont.truetype('RobotoMono-Regular.ttf', 16)
        for i, ct in enumerate([con1, con2, con3, con4, con5]):
            d.text(((i+2) * (SEARCH_FRAME_SIZE+2*PAD) + PAD + 10, PAD + 10), "%.5f" % ct.reshape([1])[0], font=fnt, fill=(255, 0, 0, 255))

    new_im.save(name)

def diagnostic_corr_maps(vgg, name):
    debug_key, debug_search, debug_key_bb, debug_search_bb = load_batch('bolt2', 'key-00000071')
    assert(debug_key is not None)
    debug_search = debug_search[15:16,:,:,:]
    debug_search_bb = debug_search_bb[15:16,:]
    #debug_search = debug_search[0:1,:,:,:]
    #debug_search_bb = debug_search_bb[0:1,:]

    gt = vgg.sess.run(vgg.ground_truth, feed_dict={vgg.search_bb: debug_search_bb})
    img = Image.fromarray((gt[0,:,:,0] * 255).astype(np.uint8))
    img.save('gt.png')

    visualize_corr_maps(vgg, 'bolt_' + name, debug_key, debug_search, debug_key_bb, debug_search_bb)

def main():

    weights_file = './vgg19.npy'
    if len(sys.argv) == 2:
        weights_file = sys.argv[1]

    vgg = vgg19.Vgg19(weights_file)

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print vgg.get_var_count()

    print 'Trainable variables (finetune):'
    print map(lambda x:x.name, vgg.cnn_var_list)
    print 'Trainable variables (gate):'
    print map(lambda x:x.name, vgg.gate_var_list)

    diagnostic_corr_maps(vgg, 'initial.png')

    validation_losses = []
    validation_iou1s = []
    validation_iou5s = []
    validation_iou25s = []

    train_losses = []
    train_iou1s = []
    train_iou5s = []
    train_iou25s = []

    valid_loss, iou1, iou5, iou25 = run_validation(vgg)
    print '[VALID] Initial validation loss: %.5f, IOU@1: %.5f, IOU@5: %.5f, IOU@25: %.5f' % (valid_loss, iou1, iou5, iou25)
    validation_losses.append(valid_loss)
    validation_iou1s.append(iou1)
    validation_iou5s.append(iou5)
    validation_iou25s.append(iou25)

    # TODO: use QueueRunner to optimize file loading on CPU
    print 'Starting training'
    start = time.time()
    for epoch in range(TRAIN_EPOCHS):
        epoch_loss_sum = 0.0
        iou1_sum = 0.0
        iou5_sum = 0.0
        iou25_sum = 0.0
        num_samples = 0

        with open(os.path.join(PROCESSED_DIR, 'train.txt')) as f:
            train_samples = f.readlines()
        order = np.random.permutation(len(train_samples))

        for train_idx in order:
            #cat_dir = os.path.join(PROCESSED_DIR, train_cat)
            #key_names = os.listdir(cat_dir)
            #for key_name in key_names:
                # ordering shouldn't matter
            train_cat, key_name = train_samples[train_idx].split('/')
            key_name = key_name.strip()

            key, search, key_bb, search_bb = load_batch(train_cat, key_name)
            if key is None:
                continue

            if MODE == 'finetune':
                #loss, iou1, iou5, iou25 = vgg.train_finetune(key, search, key_bb, search_bb)
                loss, iou1, iou5, iou25 = vgg.train_combine_layer(key, search, key_bb, search_bb)
            else:
                assert(MODE == 'gating')
                loss, iou1, iou5, iou25 = vgg.train_gate(key, search, key_bb, search_bb)

            #print '[TRAIN] Batch loss %s %s: %.5f' % (train_cat, key_name, loss)
            epoch_loss_sum += BATCH_SIZE * loss
            iou1_sum += BATCH_SIZE * iou1
            iou5_sum += BATCH_SIZE * iou5
            iou25_sum += BATCH_SIZE * iou25
            num_samples += BATCH_SIZE

        epoch_loss = epoch_loss_sum / num_samples
        epoch_iou1 = iou1_sum / num_samples
        epoch_iou5 = iou5_sum / num_samples
        epoch_iou25 = iou25_sum / num_samples
        print '[TRAIN] Epoch %d, loss: %.5f, IOU@1: %.5f, IOU@5: %.5f, IOU@25: %.5f' % (epoch+1, epoch_loss, epoch_iou1, epoch_iou5, epoch_iou25)
        train_losses.append(epoch_loss)
        train_iou1s.append(epoch_iou1)
        train_iou5s.append(epoch_iou5)
        train_iou25s.append(epoch_iou25)

        valid_loss, iou1, iou5, iou25 = run_validation(vgg)
        print '[VALID] Epoch %d, loss: %.5f, IOU@1: %.5f, IOU@5: %.5f, IOU@25: %.5f' % (epoch+1, valid_loss, iou1, iou5, iou25)
        validation_losses.append(valid_loss)
        validation_iou1s.append(iou1)
        validation_iou5s.append(iou5)
        validation_iou25s.append(iou25)

        # checkpointing
        diagnostic_corr_maps(vgg, 'epoch_%s.png' % str(epoch+1).zfill(3))

    dur = time.time() - start
    print 'Training completed in %d seconds' % dur

    # save model
    vgg.save_npy('./trained_model_%s.npy' % str(int(time.time())))
    np.save('training_metrics.npy', {
        'validation_losses': validation_losses,
        'validation_iou1s': validation_iou1s,
        'validation_iou5s': validation_iou5s,
        'validation_iou25s': validation_iou25s,
        'train_losses': train_losses,
        'train_iou1s': train_iou1s,
        'train_iou5s': train_iou5s,
        'train_iou25s': train_iou25s
    })
    vgg.sess.close()

if __name__ == '__main__':
    main()
