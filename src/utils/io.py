# --------------------------------------------------------
# IEF
# Copyright (c) 2015
# Licensed under BSD License [see LICENSE for details]
# Written by Joao Carreira, Pulkit Agrawal and Katerina Fragkiadki
# --------------------------------------------------------

import numpy as np
import copy
import easydict as edict
from os import path as osp

##
# For reading the dataset data
class DataSet(object):
	def __init__(self, cfg):
		self.setFile_   = edict.EasyDict()
		self.setNames_  = ['train', 'val', 'test'] 
		for s in self.setNames_:
			self.setFile_[s] = cfg.PATHS.SET_FILE % s 
			assert osp.isfile(self.setFile_[s]), '%s not found' % setFile_[s]
		self.dataFile_ = cfg.PATHS.DATA_FILE

	#The names of files belonging to a set	
	def get_set_ids(self, setName):
		with open(self.setFile_[setName]) as f:
			names = f.readlines()
			names = [n.strip() for n in names]
		return names

	#Return the datafiles belonging to a set
	def get_set_files(self, setName):
		names     = self.get_set_ids(setName)
		dataFiles = [self.dataFile_ % ('%06d' % int(n)) for n in names]
		return dataFiles 

