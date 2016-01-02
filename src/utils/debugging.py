#!/usr/bin/env python

# --------------------------------------------------------
# IEF
# Copyright (c) 2015
# Licensed under The MIT License [see LICENSE for details]
# Written by Joao Carreira, Pulkit Agrawal and Katerina Fragkiadki
# --------------------------------------------------------

import numpy as np
from utils import imutils as imu
from utils import imdata as imd
from utils import io
from config import cfg

def get_mpii_obj(num=0):
	ioDat    = io.DataSet(cfg)
	trnNames = ioDat.get_set_files('train')
	kpt      = imd.ImKPtDataMpii.from_file(trnNames[num])
	return kpt

def vis_mpii(num=0):
	ioDat    = io.DataSet(cfg)
	trnNames = ioDat.get_set_files('train')
	kpt      = imd.ImKPtDataMpii.from_file(trnNames[num])
	kpt.plot() 



