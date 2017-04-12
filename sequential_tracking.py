"""
Sequential tracking evaluation
"""

import os
import sys
import time

import tensorflow as tf
import numpy as np
from PIL import Image

from CONSTANTS import *
import vgg19_tracker as vgg19
import vot_preprocess as vp

key_frame_name = '00000001.jpg'

def PIL_to_np(pil, size):
    return np.transpose(np.asarray(pil), [1,0,2]).reshape([1, size, size, 3])

def main():

    weights_file = './vgg19.npy'
    if len(sys.argv) == 2:
        weights_file = sys.argv[1]

    vgg = vgg19.Vgg19(weights_file)
    total_frames = 0

    start = time.time()
    for cat in TEST_CATS:
        cat_dir = os.path.join(VOT_DIR, cat)
        ground_truth = open(os.path.join(cat_dir, 'groundtruth.txt')).readlines()
        num_frames = len(ground_truth)

        key_im = Image.open(os.path.join(cat_dir, key_frame_name))
        x, y, w, h = vp.convert_to_xywh(ground_truth[0])
        key_frame, scale = vp.extract_key_frame(key_im, x, y, w, h)
        key_frame_np = PIL_to_np(key_frame, KEY_FRAME_SIZE)

        key_bb = np.array([w * scale, h * scale])

        # Keeps track of the PREDICTED x, y of the previous frame (starting with ground truth key frame)
        prev_x, prev_y = x, y   # w, h do not change

        for frame_idx in xrange(2, num_frames+1):
            search_frame_name = '%s.jpg' % (str(frame_idx).zfill(8))

            sx, sy, sw, sh = vp.convert_to_xywh(ground_truth[frame_idx - 1])
            offset_x = (sx + sw/2) - (prev_x + w/2)
            offset_y = (sy + sh/2) - (prev_y + h/2)

            search_im = Image.open(os.path.join(cat_dir, search_frame_name))
            search_frame = vp.extract_search_frame(search_im, prev_x, prev_y, w, h, scale)
            search_frame_np = PIL_to_np(search_frame, SEARCH_FRAME_SIZE)

            search_bb = np.array([[offset_x * scale, offset_y * scale, sw * scale, sh * scale]])

            # TODO for FPS calculations, take out search_bb since gt not used

            pred_box, iou = vgg.sess.run([vgg.raw_pred_box, vgg.raw_IOUT_at_1],
                    feed_dict={vgg.key_img: key_frame_np,
                               vgg.search_img: search_frame_np,
                               vgg.key_bb: key_bb,
                               vgg.search_bb: search_bb})

            print 'Frame %d IOU %.5f' % (frame_idx, iou)

            prev_x = pred_box[0][0]
            prev_y = pred_box[1][0]
            total_frames += 1

    dur = time.time() - start
    print 'Elapsed time: %d sec, frame considered: %d, FPS: %.5f' % (dur, total_frames, total_frames / float(dur))


if __name__ == '__main__':
    main()
