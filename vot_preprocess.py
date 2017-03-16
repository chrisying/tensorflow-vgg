'''
Preprocess images in VOT2016 by:
    - Taking the first frame per MAX_FRAME_GAP frames as the key frame
    - Process to KEY_FRAME_SIZE
        - Pad at least w+h/8
        - Paste onto averge RGB background (if necessary)
        - Crop
        - Resize
    - Take remaining MAX_FRAME_GAP-1 images and process to SEARCH_FRAME_SIZE

New ground truth format:
    key-00000001: [top-left x] [top-left y] [width] [height] [scale]
    search-00000002: [x offset] [y offset]
    ...
    search-[MAX_FRAME_GAP+1]: [x offset] [y offset]
    key-00000002: [top-left x] [top-left y] [width] [height] [scale]
    ...

original image size * scale = key/search image size
x/y offsets are in original image dimensions
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

    pad = (w + h) / 8
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
    return im, KEY_FRAME_SIZE / key_size

def extract_search_frame(im, x, y, w, h, scale):
    # x, y, w, h is from the KEY frame
    im_w, im_h = im.size

    search_size = SEARCH_FRAME_SIZE / scale
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
            print 'Begin processing %s' % cat
            cat_dir = os.path.join(VOT_DIR, cat)
            ground_truth = open(os.path.join(cat_dir, 'groundtruth.txt')).readlines()
            num_frames = len(ground_truth)

            output_dir = os.path.join(PROCESSED_DIR, cat)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)     # Theoretically a race condition

            for block_idx in range(num_frames / KEY_FRAME_GAP):
                print 'Begin processing key frame %s' % block_idx
                # Process key frame
                key_frame_idx = block_idx * KEY_FRAME_GAP + 1
                key_frame_name = str(key_frame_idx).zfill(8)

                key_dir = os.path.join(output_dir, 'key-%s' % key_frame_name)
                if not os.path.exists(key_dir):
                    os.makedirs(key_dir)

                key_im = Image.open(os.path.join(cat_dir, key_frame_name + '.jpg'))
                x, y, w, h = convert_to_xywh(ground_truth[key_frame_idx - 1])
                new_key_im, scale = extract_key_frame(key_im, x, y, w, h)
                key_output_name = 'key-%s.png' % key_frame_name
                new_key_im.save(os.path.join(key_dir, key_output_name))

                new_gt = open(os.path.join(key_dir, 'groundtruth.txt'), 'w')
                new_gt.write('key-%s: %.3f %.3f %.3f %.3f %.3f\n' %
                        (key_frame_name, x, y, w, h, scale))


                # Process search frames
                for img_idx in range(MAX_FRAME_GAP):
                    search_frame_idx = key_frame_idx + img_idx + 1
                    if search_frame_idx > num_frames:
                        break
                    search_frame_name = str(search_frame_idx).zfill(8)
                    search_im = Image.open(os.path.join(cat_dir, search_frame_name + '.jpg'))
                    new_search_im = extract_search_frame(search_im, x, y, w, h, scale)
                    search_output_name = 'search-%s.png' % (search_frame_name)
                    new_search_im.save(os.path.join(key_dir, search_output_name))

                    sx, sy, sw, sh = convert_to_xywh(ground_truth[search_frame_idx - 1])
                    offset_x = (sx + sw/2) - (x + w/2)
                    offset_y = (sy + sh/2) - (y + h/2)
                    new_gt.write('search-%s: %.3f %.3f\n' %
                            (search_frame_name, offset_x, offset_y))

                new_gt.close()

if __name__ == '__main__':
    main()


