'''
Preprocess images in VOT2016 by:
    - Taking the first frame per MAX_FRAME_GAP frames as the key frame
    - Process to KEY_FRAME_SIZE
        - Pad at least w+h/4
        - Paste onto averge RGB background (if necessary)
        - Crop
        - Resize
    - Take remaining MAX_FRAME_GAP-1 images and process to SEARCH_FRAME_SIZE
'''

import math
import os

import numpy as np
from PIL import Image, ImageOps

from CONSTANTS import *

def convert_to_xywh(gt_line):
    x1,y1,x2,y2,x3,y3,x4,y4 = map(float, gt_line.split(','))
    x = (x1 + x4) / 2
    y = (y1 + y2) / 2
    w= ((x2 + x3) - (x1 + x4)) / 2
    h= ((y3 + y4) - (y1 + y2)) / 2
    return x, y, w, h

def get_mean_rgb(im):
    return np.array(im).mean(axis=(0,1))

def extract_key_frame(im, x, y, w, h):
    im_w, im_h = im.size

    pad = (w + h) / 4
    key_size = max(w + 2*pad, h + 2*pad)
    new_x = x + w/2 - key_size / 2
    new_y = y + h/2 - key_size / 2

    if new_x >= 0 and new_y >= 0 and new_x + key_size < im_w and new_y + key_size < im_h:
        im = im.crop((new_x, new_y, new_x + key_size, new_y + key_size))
    else:
        # Requires padding
        border = max(0, -new_x, -new_y, new_x + key_size - im_w, new_y + key_size - im_h)
        mean_rgb = tuple(map(int, list(get_mean_rgb(im))))
        im = ImageOps.expand(im, border=int(math.ceil(border)), fill=mean_rgb)
        im = im.crop((new_x + border, new_y + border, new_x + key_size + border, new_y + key_size + border))

    im = im.resize((KEY_FRAME_SIZE, KEY_FRAME_SIZE), resample=Image.BILINEAR)

    # Also returns the scale factor
    return im, key_size / KEY_FRAME_SIZE

def extract_search_frame(im, x, y, w, h, scale):
    # x, y, w, h is from the KEY frame
    im_w, im_h = im.size

    search_size = scale * SEARCH_FRAME_SIZE
    new_x = x + w/2 - search_size / 2
    new_y = y + h/2 - search_size / 2

    if new_x >= 0 and new_y >= 0 and new_x + search_size < im_w and new_y + search_size < im_h:
        im = im.crop((new_x, new_y, new_x + search_size, new_y + search_size))
    else:
        # Requires padding
        border = max(0, -new_x, -new_y, new_x + search_size - im_w, new_y + search_size - im_h)
        mean_rgb = tuple(map(int, list(get_mean_rgb(im))))
        im = ImageOps.expand(im, border=int(math.ceil(border)), fill=mean_rgb)
        im = im.crop((new_x + border, new_y + border, new_x + search_size + border, new_y + search_size + border))

    im = im.resize((SEARCH_FRAME_SIZE, SEARCH_FRAME_SIZE), resample=Image.BILINEAR)

    return im



def main():
    with open(os.path.join(VOT_DIR, 'list.txt')) as list_txt:
        for cat in list_txt.xreadlines():
            cat = cat.strip()
            cat_dir = os.path.join(VOT_DIR, cat)
            num_frames = len(open(os.path.join(cat_dir, 'groundtruth.txt')).readlines())
            with open(os.path.join(cat_dir, 'groundtruth.txt')) as gt:
                for block_idx in range(num_frames / MAX_FRAME_GAP):
                    # Process key frame
                    key_frame_name = str(block_idx * MAX_FRAME_GAP + 1).zfill(8)
                    key_im = Image.open(os.path.join(cat_dir, key_frame_name + '.jpg'))
                    x, y, w, h = convert_to_xywh(gt.readline())
                    new_key_im, scale = extract_key_frame(key_im, x, y, w, h)
                    key_output_name = '%s-%s-key.png' % (cat, key_frame_name)
                    new_key_im.save(os.path.join(PROCESSED_DIR, key_output_name))

                    # Process search frames
                    for img_idx in range(1, MAX_FRAME_GAP):
                        search_frame_name = str(block_idx * MAX_FRAME_GAP + img_idx + 1).zfill(8)
                        search_im = Image.open(os.path.join(cat_dir, search_frame_name + '.jpg'))
                        new_search_im = extract_search_frame(search_im, x, y, w, h, scale)
                        search_output_name = '%s-%s-search.png' % (cat, search_frame_name)
                        new_search_im.save(os.path.join(PROCESSED_DIR, search_output_name))

                        # TODO: write down ground truth somewhere (maybe in terms of offset?)
                        gt.readline()

if __name__ == '__main__':
    main()


