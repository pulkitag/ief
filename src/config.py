# --------------------------------------------------------
# IEF
# Copyright (c) 2015
# Licensed under BSD License [see LICENSE for details]
# Written by Joao Carreira, Pulkit Agrawal and Katerina Fragkiadki
# --------------------------------------------------------

"""IEF config system

This is inspired by the RCNN config system. 

This file specifies default config options for IEF. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

"""


import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from config import cfg
cfg = __C

#DataSet Name
__C.DATASET = 'mpii'

#Paths
__C.PATHS = edict()
__C.PATHS.BASE_DIR  = osp.join('/work5/pulkitag/', __C.DATASET)
__C.PATHS.SET_FILE  = osp.join(__C.PATHS.BASE_DIR, 'ImageSets',
															 'Main', '%s.txt')
__C.PATHS.DATA_FILE = osp.join(__C.PATHS.BASE_DIR, 'Annotations',
															 '%s.mat')

#For storing model
__C.PATHS.MODEL_DIR   = 'models/'
__C.SCALE_MODEL = edict() 							
#The path of the model for determining the scale								 
__C.SCALE_MODEL.NET   = osp.join(__C.PATHS.MODEL_DIR,
													  'scalesel-vggs-epoch-14-convert.caffemodel')
__C.SCALE_MODEL.PROTO = osp.join(__C.PATHS.MODEL_DIR,
														'vgg_s.prototxt') 
#The path of the model for estimating the pose
__C.POSE_MODEL = edict()
__C.POSE_MODEL.NET = osp.join(__C.PATHS.MODEL_DIR,
													 'ief-googlenet-dec2015.caffemodel')
__C.POSE_MODEL.PROTO = osp.join(__C.PATHS.MODEL_DIR,
													 'ief-googlenet-dec2015.prototxt')
__C.POSE_MODEL.META  = osp.join(__C.PATHS.MODEL_DIR,
													 'ief-googlenet-dec2015-meta.pkl')

__C.CROP_SIZE = 256
__C.GAUSSIAN_WIDTH = 5


#The lambdas for determining the optimal scale
__C.SCALE_LAMBDA = [1.4142, 1.1892, 1, 0.8409, 0.7071, 0.5946, 0.5, 0.4204, 0.3536,  0.2973] 



