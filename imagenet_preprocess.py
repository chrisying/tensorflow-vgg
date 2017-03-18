'''
Preprocess images in Imagenet VID by:
    - Taking the first frame per MAX_FRAME_GAP frames as the key frame
    - Process to KEY_FRAME_SIZE
        - Pad at least w+h/8
        - Paste onto averge RGB background (if necessary)
        - Crop
        - Resize
    - Take remaining MAX_FRAME_GAP-1 images and process to SEARCH_FRAME_SIZE

New ground truth format:
    key-00000000: [top-left x] [top-left y] [width] [height] [scale]
    search-00000001: [x offset] [y offset]
    ...
    search-[MAX_FRAME_GAP]: [x offset] [y offset]
    key-00000001: [top-left x] [top-left y] [width] [height] [scale]
    ...

original image size * scale = key/search image size
x/y offsets are in original image dimensions
'''

import math
import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageOps

from CONSTANTS import *

def convert_to_xywh(elem):
    xmax, xmin, ymax, ymin = elem[0], elem[1], elem[2], elem[3]
    x = (xmin + xmax) / 2.0
    y = (ymin + ymax) / 2.0
    w = float(xmax - xmin)
    h = float(ymax - ymin)
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

    dirs = ['train/ILSVRC2015_VID_train_0000',
            'train/ILSVRC2015_VID_train_0001',
            'train/ILSVRC2015_VID_train_0002',
            'train/ILSVRC2015_VID_train_0003',
            'val/ILSVRC2015_VID_train_0000',
            'val/ILSVRC2015_VID_train_0001',
            'val/ILSVRC2015_VID_train_0002',
            'val/ILSVRC2015_VID_train_0003']

    for d in dirs:
        annot_path = os.path.join(IMAGENET_DIR, 'Annotations/VID', d)
        data_path = os.path.join(IMAGENET_DIR, 'Data/VID', d)

        vids = os.listdir(annot_path)
        for v in vids:
            vid_dir = os.path.join(annot_path, v)
            annots = os.listdir(vid_dir)
            num_frames = len(annots)

            output_dir = os.path.join(IMAGENET_PROCESSED_DIR, d, v)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            for block_idx in range(num_frames / KEY_FRAME_GAP):
                print 'Begin processing key frame %s' % block_idx
                key_frame_idx = block_idx * KEY_FRAME_GAP
                key_frame_name = str(key_frame_idx).zfill(6)

                key_dir = os.path.join(IMAGENET_PROCESSED_DIR, 'key-%s' % key_frame_name)
                if not os.path.exists(key_dir):
                    os.makedirs(key_dir)

                annot_filename = '%s.xml' % key_frame_name
                tree = ET.parse(os.path.join(vid_dir, annot_filename))
                root = tree.getroot()

                key_im = Image.open(os.path.join(data_path, v, '%s' % key_frame_name))
                assert(root[4].tag == 'object' and root[4][0].text == '0')
                x, y, w, h = convert_to_xywh(root[4][2])
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
                    search_frame_name = str(search_frame_idx).zfill(6)
                    search_im = Image.open(os.path.join(data_path, v, '%s' % search_frame_name))
                    new_search_im = extract_search_frame(search_im, x, y, w, h, scale)
                    search_output_name = 'search-%s.png' % (search_frame_name)
                    new_search_im.save(os.path.join(key_dir, search_output_name))

                    search_filename = '%s.xml' % search_frame_name
                    search_tree = ET.parse(os.path.join(vid_dir, search_filename))
                    search_root = tree.getroot()
                    assert(search_root[4].tag == 'object' and search_root[4][0].text == '0')
                    sx, sy, sw, sh = convert_to_xywh(search_root[4][2])
                    offset_x = (sx + sw/2) - (x + w/2)
                    offset_y = (sy + sh/2) - (y + h/2)
                    new_gt.write('search-%s: %.3f %.3f\n' %
                            (search_frame_name, offset_x, offset_y))

                new_gt.close()


if __name__ == '__main__':
    main()


