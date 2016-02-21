#!/usr/bin/env python

# --------------------------------------------------------
# IEF
# Copyright (c) 2015
# Licensed under BSD License [see LICENSE for details]
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


def save_mpii_data_pythonic(setName='train'):
	ioDat  = io.DataSet(cfg)
	fNames = ioDat.get_set_files(setName)
	data   = imd.ImKPtDataMpii.from_file(fNames[0])
	return data
	print data.imFile_
	print fNames[0] 

