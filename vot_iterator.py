# Scan through VOT dataset for various reasons

import os

VOT_DIR = '/home/cying/vot2016/'

max_width = -1
max_height = -1
num_images = 0

with open(os.path.join(VOT_DIR, 'list.txt')) as list_txt:
    for cat in list_txt.xreadlines():
        cat_dir = os.path.join(VOT_DIR, cat)
        with open(os.path.join(cat_dir, 'groundtruth.txt')) as gt:
            for line in gt.xreadlines():
                num_images += 1
                x1,y1,_,_,x2,y2,_,_ = line.split(',')
                width = x2 - x1
                height = y2 - y1
                print width, height
                max_width = max(max_width, width)
                max_height = max(max_height, height)

print 'Max box: (%d, %d)' % (max_width, max_height)
print 'Num images: %d' % num_images



