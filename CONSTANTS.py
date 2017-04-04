# Parameters used in this project all in one place

VOT_DIR = '/Users/cying/space/research/vot'
PROCESSED_DIR = '/Users/cying/space/research/vot/processed'
#VOT_DIR = '/home/cying/vot2016'
#PROCESSED_DIR = '/home/cying/processed_vot'

IMAGENET_DIR = '/home/cying/imagenet/ILSVRC'
IMVID_PROCESSED_DIR = '/home/cying/imagenet/processed'

KEY_FRAME_GAP = 10
MAX_FRAME_GAP = 100
KEY_FRAME_SIZE = 128
SEARCH_FRAME_SIZE = 256
BATCH_SIZE = 25
GAUSSIAN_AMP = 1
GAUSSIAN_VAR = 20
LAMBDA = 0.25       # [0.0, 1.0], higher = more weight to computational cost

TRAIN_EPOCHS = 1
TRAIN_CATS = ['bag',
              'ball1',
              'basketball',
              'birds1',
              'blanket',
              'bmx',
              'bolt1',
              'book',
              'butterfly',
              'car1',
              'crossing',
              'dinosaur',
              'fernando',
              'fish1',
              'fish2',
              'girl',
              'glove',
              'godfather',
              'graduate',
              'gymnastics1',
              'gymnastics2',
              'hand',
              'handball1',
              'helicopter',
              'iceskater1',
              'leaves',
              'marching',
              'matrix',
              'motocross1',
              'nature',
              'octopus',
              'pedestrian1',
              'rabbit',
              'racing',
              'road',
              'shaking',
              'sheep',
              'singer1',
              'singer2',
              'soccer1',
              'soldier',
              'sphere',
              'tiger',
              'traffic',
              'tunnel',
              'wiper']

TEST_CATS = ['ball2',
             'birds2',
             'bolt2',
             'car2',
             'fish3',
             'fish4',
             'gymnastics3',
             'gymnastics4',
             'handball2',
             'iceskater2',
             'motocross2',
             'pedestrian2',
             'singer3',
             'soccer2']
