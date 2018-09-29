from __future__ import division

import os
import cv2
import time
import _pickle as cPickle
from keras_alfnet import config

# pass the settings in the config object
C = config.Config()
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

if C.network=='resnet50':
    w_path = 'data/models/city_res50_%sstep.hdf5' % (C.steps)
elif C.network=='mobilenet' and C.steps in [1,2]:
    w_path = 'data/models/city_mobnet_%sstep.hdf5'% (C.steps)
else:
    raise NotImplementedError('Not support network: {}'.format(C.network))

# define output path for detection results
out_path = 'output/valresults/%s/%dstep' % (C.network, C.steps)
if not os.path.exists(out_path):
    os.makedirs(out_path)

# get the test data
# cache_path = 'data/cache/cityperson/val'
# with open(cache_path, 'rb') as fid:
#    val_data = cPickle.load(fid, encoding='bytes')
# num_imgs = len(val_data)
# print('num of val samples: {}'.format(num_imgs))
val_data = None

# C.random_crop = (1024, 2048)

# img_file = '/Users/wangpeng/Work/rongyi/opensource/LALFNet/data/examples/munster_000042_000019_leftImg8bit.png'
# img_file = '/Users/wangpeng/Downloads/pedestrian.jpg'
img_file = '/Users/wangpeng/Work/rongyi/opensource/SSD-Tensorflow/demo/street.jpg'
# img_file = '/Users/wangpeng/Work/rongyi/opensource/SSD-Tensorflow/demo/person.jpg'
img = cv2.imread(img_file)
C.random_crop = (img.shape[0], img.shape[1])
# resized_img = cv2.resize(img, C.random_crop)
# ratio = (img.shape[0] / float(C.random_crop[0]), img.shape[1] / float(C.random_crop[1]))
# print('ratio: (%s, %s)' % ratio)
print('image shape: ', img.shape)

# define the ALFNet network
if C.steps == 1:
    from keras_alfnet.model.model_1step import Model_1step
    model = Model_1step()
elif C.steps == 2:
    from keras_alfnet.model.model_2step import Model_2step
    model = Model_2step()
elif C.steps == 3:
    from keras_alfnet.model.model_3step import Model_3step
    model = Model_3step()
else:
    raise NotImplementedError('Not implement {} or more steps'.format(C.steps))
model.initialize(C)
model.creat_model(C, val_data, phase='inference')
# model.test_model(C, val_data, w_path, out_path)

start = time.time()
bbx, scores = model.test_model_with_input_image(img, C, w_path)
elapsed = time.time() - start
print('elapsed time: %s' % elapsed)
"""
bbx, scores = model.test_model_with_input_image(resized_img, C, w_path)
bbx[:, 0] = bbx[:, 0] * ratio[0]
bbx[:, 2] = bbx[:, 2] * ratio[0]
bbx[:, 1] = bbx[:, 1] * ratio[1]
bbx[:, 3] = bbx[:, 3] * ratio[1]
"""

for i in range(len(scores)):
    if scores[i] > 0.8:
        box = bbx[i]
        print(box, scores[i])
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        # cv2.rectangle(resized_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
cv2.imshow('image', img)
cv2.waitKey(0)
