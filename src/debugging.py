#!/usr/bin/env python

# --------------------------------------------------------
# IEF
# Copyright (c) 2015
# Licensed under BSD License [see LICENSE for details]
# Written by Joao Carreira, Pulkit Agrawal and Katerina Fragkiadki
# --------------------------------------------------------

from src import test_demo as td
import numpy as np
from utils import imutils as imu
from utils import imdata as imd
from utils import io
from config import cfg
import pickle
from os import path as osp
import scipy.misc as scm
import matplotlib.pyplot as plt

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

def get_pose_predictor():
	ief    = td.PoseIEF()
	return ief

def det_pose(ief=None, isShow=True):
	dirName = 'datasets/mpii/data'
	#Get detection data
	detFile = osp.join(dirName, 'dets/mpii_person_det.pkl')
	detList = pickle.load(open(detFile, 'r'))['person_det']
	detData = {}
	for l in detList:
		key, _, dets = l
		detData[key] = dets
	#Get validation data
	valFile  = osp.join(dirName, 'annotations/val_data.pkl')
	valData  = pickle.load(open(valFile, 'r'))
	valNames = valData['imName']
	gtPos    = valData['bodyPos']
	#Get figure
	plt.close('all')
	fig = plt.figure()
	ax  = fig.add_subplot(111)	
	for i, name in enumerate(valNames):
		print (i)
		ax.clear()
		bName    = osp.basename(name).split('.')[0]
		if len(detData[bName]) == 0:
			continue
		bbox     = detData[bName][0]
		gtPt     = gtPos[i].squeeze()
		x1, y1, x2, y2, conf = bbox
		x = int((x1 + x2)/2.0)
		y = int(y1 + (y2-y1)/3.0)	
		#Predict the pose
		imName = osp.join(dirName, name)
		pose,_ = ief.predict(imName, (x,y))
		im     = scm.imread(imName)
		td.vis.plot_bbox(bbox, ax=ax, isShow=isShow)
		td.vis.plot_pose_stickmodel(im, pose.squeeze().transpose((1,0)),
        ax=ax, isShow=isShow)
		ax.plot(x, y, 'r', markersize=7, marker='o')
		ax.plot(gtPt[0], gtPt[1], 'black', markersize=7, marker='o')
		if isShow:
			plt.draw()
			plt.show()
		if not isShow:
			plt.savefig('src/tmp/%s.jpg' % bName)
		else:
			ip = raw_input()
			if ip == 'q':
				return	

