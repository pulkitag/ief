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
import pickle
from os import path as osp

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
	oFile   = '%s_data.pkl' % setName
	allData = {}
	allData['imName']  = []
	allData['bodyPos'] = []
	allData['bodyScale'] = []
	allData['kpts']      = []
	allData['num']       = []  
	N = 0
	for i,f in enumerate(fNames):
		try:
			data   = imd.ImKPtDataMpii.from_file(f)
		except:
			N += 1
			print ('ERROR ENCOUNTERED - SKIPPING')
			print (i, f)
			continue
		name   = osp.join('images', osp.basename(data.imFile_))
		allData['imName'].append(name)
		allData['bodyPos'].append(data.bodyPos_)
		allData['bodyScale'].append(data.bodyScale_)
		allData['kpts'].append(data.kpts_)
		allData['num'].append(len(data.kpts_))
	print 'Missing files - %d' % N
	pickle.dump(allData, open(oFile, 'w'))

