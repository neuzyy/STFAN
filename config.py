#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>

from easydict import EasyDict as edict
import os
import socket

__C = edict()
cfg = __C
__C.CONST = edict()
__C.DATASET = edict()
__C.DIR = edict()
__C.DATA = edict()
__C.TEST = edict()
__C.TRAIN = edict()
__C.LOSS = edict()
__C.NETWORK = edict()

'''
Common
'''
__C.CONST.DEVICE = '3'  # '0'
__C.CONST.WEIGHTS = './ckpt/best-ckpt_refined-STFAN.pth.tar'
__C.CONST.TRAIN_BATCH_SIZE = 5
__C.CONST.TEST_BATCH_SIZE = 1
__C.CONST.NUM_WORKER_TRAIN = __C.CONST.TRAIN_BATCH_SIZE  # number of data workers
__C.CONST.NUM_WORKER_TEST = __C.CONST.TEST_BATCH_SIZE

'''
Dataset
'''
__C.DATASET.DATASET_NAME = 'oppo_low_all'  # DVD, oppo_low_all

'''
Directories
'''
__C.DIR.OUT_Tag = 'train_low-all_refined_test_real-blur'  # Empty means using time as folder name
__C.DIR.OUT_PATH = './output/test'

'''
data augmentation
'''
__C.DATA.STD = [255.0, 255.0, 255.0]
__C.DATA.MEAN = [0.0, 0.0, 0.0]
__C.DATA.CROP_IMG_SIZE = [320, 448]  # Crop image size: height, width
__C.DATA.GAUSSIAN = [0, 1e-4]  # mu, std_var
__C.DATA.COLOR_JITTER = [0.2, 0.15, 0.3, 0.1]  # brightness, contrast, saturation, hue
__C.DATA.SEQ_LENGTH = 100            # 100,50,20

'''
Network
'''
__C.NETWORK.DEBLURNETARCH = 'STFAN'  # available options: STFAN
__C.NETWORK.FAC_KS = 5
__C.NETWORK.LEAKY_VALUE = 0.1
__C.NETWORK.BATCHNORM = False
__C.NETWORK.PHASE = 'test'  # available options: 'train', 'test', 'resume'

'''
Training
'''
__C.TRAIN.NUM_EPOCHES = 400  # maximum number of epoches
__C.TRAIN.LEARNING_RATE = 1e-4
__C.TRAIN.LR_MILESTONES = [80, 160, 250]
__C.TRAIN.LR_DECAY = 0.1  # Multiplicative factor of learning rate decay
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.BETA = 0.999
__C.TRAIN.BIAS_DECAY = 0.0  # regularization of bias, default: 0
__C.TRAIN.WEIGHT_DECAY = 0.0  # regularization of weight, default: 0
__C.TRAIN.PRINT_FREQ = 10
# weights will be overwritten every save_freq epoch
__C.TRAIN.SAVE_FREQ = 20

__C.LOSS.USE_PERCET_LOSS = True
__C.LOSS.MULTISCALE_WEIGHTS = [0.3, 0.3, 0.2, 0.1, 0.1]

'''
Testing options
'''
__C.TEST.SSIM_CALU = False
__C.TEST.VISUALIZATION_NUM = 0
__C.TEST.PRINT_FREQ = 5
if __C.NETWORK.PHASE == 'test':
    __C.CONST.TEST_BATCH_SIZE = 1


if cfg.DATASET.DATASET_NAME == 'oppo_low_all':
    __C.DIR.OUT_PATH = './output/train_low-all'
    __C.DIR.DATASET_JSON_FILE_PATH = './datasets/oppo_low_all.json'
    __C.DIR.DATASET_ROOT = './datasets/oppo_low_all'
    __C.DIR.IMAGE_BLUR_PATH = os.path.join(__C.DIR.DATASET_ROOT, '%s/%s/input/%s.png')
    __C.DIR.IMAGE_CLEAR_PATH = os.path.join(__C.DIR.DATASET_ROOT, '%s/%s/GT/%s.png')
elif cfg.DATASET.DATASET_NAME == 'real-blur':
    __C.DIR.OUT_PATH = './output/train_real-blur'
    __C.DIR.DATASET_JSON_FILE_PATH = './datasets/real-blur_oppo-val.json'
    __C.DIR.DATASET_ROOT = './datasets/real-blur_oppo-val'
    __C.DIR.IMAGE_BLUR_PATH = os.path.join(__C.DIR.DATASET_ROOT, '%s/%s/input/%s.png')
    __C.DIR.IMAGE_CLEAR_PATH = os.path.join(__C.DIR.DATASET_ROOT, '%s/%s/input/%s.png')
elif cfg.DATASET.DATASET_NAME == 'DVD':
    __C.DIR.OUT_PATH = './output/train_DVD'
    __C.DIR.DATASET_JSON_FILE_PATH = './datasets/DVD.json'
    __C.DIR.DATASET_ROOT = './datasets/DVD'
    __C.DIR.IMAGE_BLUR_PATH = os.path.join(__C.DIR.DATASET_ROOT, '%s/%s/input/%s.jpg')
    __C.DIR.IMAGE_CLEAR_PATH = os.path.join(__C.DIR.DATASET_ROOT, '%s/%s/GT/%s.jpg')
elif cfg.DATASET.DATASET_NAME == 'Real':
    __C.DIR.OUT_PATH = './output/train_Real'
    __C.DIR.DATASET_JSON_FILE_PATH = './datasets/Real.json'
    __C.DIR.DATASET_ROOT = './datasets/Real'
    __C.DIR.IMAGE_BLUR_PATH = os.path.join(__C.DIR.DATASET_ROOT, '%s/%s/input/%s.jpg')
    __C.DIR.IMAGE_CLEAR_PATH = os.path.join(__C.DIR.DATASET_ROOT, '%s/%s/input/%s.jpg')
else:
    print("错误：数据集{}不存在".format(cfg.DATASET.DATASET_NAME))
    exit(-1)
