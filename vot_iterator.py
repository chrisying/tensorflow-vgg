# Scan through VOT dataset for various reasons

from PIL import Image

import os

from CONSTANTS import *

max_width = -1
max_height = -1
sum_width = 0
sum_height = 0
num_images = 0

with open(os.path.join(VOT_DIR, 'list.txt')) as list_txt:
    for cat in list_txt.xreadlines():
        cat_dir = os.path.join(VOT_DIR, cat.strip())
        with open(os.path.join(cat_dir, 'groundtruth.txt')) as gt:
            for line in gt.xreadlines():
                num_images += 1
                x1,y1,_,_,x2,y2,_,_ = map(float, line.split(','))
                width = x2 - x1
                height = y2 - y1
                max_width = max(max_width, width)
                max_height = max(max_height, height)
                sum_width += width
                sum_height += height
        im1 = Image.open(os.path.join(cat_dir, '00000001.png'))
        print im1.size

print 'Max box: (%d, %d)' % (max_width, max_height)
print 'Avg box: (%d, %d)' % (sum_width/num_images, sum_height/num_images)
print 'Num images: %d' % num_images

