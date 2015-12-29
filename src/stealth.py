import my_pycaffe as mp
from utils import imdata as imd
from utils import io
from config import cfg
from os import path as osp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy

LIST_SCALES = [1.4142, 1.1892, 1, 0.8409, 0.7071, 0.5946, 0.5, 0.4204, 0.3536,  0.2973]  

def get_scale_net():
	modelDir = '/work4/pulkitag-code/code/ief/IEF/models/'
	netName  = osp.join(modelDir, 'scalesel-vggs-epoch-14-convert.caffemodel')
	defFile  = osp.join(modelDir, 'vgg_s.prototxt')
	net      = mp.MyNet(defFile, netName, deviceId=1)
	#Set preprocessing in the net
	net.set_preprocess(meanDat='/data1/pulkitag/caffe_models/ilsvrc2012_mean.binaryproto') 
	return net

def get_ief_net():
	modelDir = '/work4/pulkitag-code/code/ief/IEF/models/'
	netName  = osp.join(modelDir, 'ief-googlenet-dec2015.caffemodel')
	defFile  = osp.join(modelDir, 'ief-googlenet-dec2015.prototxt')
	metaFile = osp.join(modelDir, 'ief-googlenet-dec2015-meta.pkl')
	net      = mp.MyNet(defFile, netName, deviceId=1)
	#Set preprocessing in the net
	#As of now the ief net takes RGB images, but the scale net takes BGR
	net.set_preprocess(chSwap=None, meanDat=(122.6768, 116.6703, 104.0108), ipName='image')
	net.set_preprocess(ipName='kp_pos', noTransform=True)
	net.set_preprocess(ipName='label',  noTransform=True)
	#Get the metadata
	metaData = pickle.load(open(metaFile, 'r')) 
	return net, metaData

def predict_pose(setName='val'):
	#Read the scales
	scaleFile = 'tmp-ief-scale-%s.pkl' % setName
	data      = pickle.load(open(scaleFile,'r'))
	fNames, scales = data['fNames'], data['scale']
	#Get the network
	net, metaData  = get_ief_net()
	seedPose       = metaData['seedPose']
	mxStepSz       = metaData['mxStepNorm'] 
	#Go over the images and predict poses	
	allPoses = []
	for num, sc in zip(range(len(fNames)), scales):
		print (num)
		kpt    = imd.ImKPtDataMpii.from_file(fNames[num])
		N      = kpt.N_ #Number of people
		ops = []
		for  n in range(N):
			cropIm = kpt.crop_at_scale(scale=sc[n], cropSz=224)
			imData = np.zeros((1,224,224,3)).astype(np.float32)
			imData[0,:,:,:] = cropIm[n]
			currPose        = np.zeros((1,17,2,1)).astype(np.float32)
			for i in range(17):
				currPose[0,i,0] = seedPose[0,i]
				currPose[0,i,1] = seedPose[1,i]
			labels = np.zeros((1,16,2,1)).astype(np.float32)
			poseSteps = []
			for step in range(4):
				opDat    = net.forward(blobs=['cls3_fc'], image=imData, kp_pos=currPose, label=labels)
				opDat    = opDat['cls3_fc'].squeeze()
				for i in range(16):
					dx, dy = opDat[i], opDat[16 + i]
					currPose[0,i,0] = currPose[0,i,0] + mxStepSz * dx
					currPose[0,i,1] = currPose[0,i,1] + mxStepSz * dy
				poseSteps.append(copy.deepcopy(currPose))
			ops.append(poseSteps)
		allPoses.append(ops)
	return fNames, allPoses 


def predict_scale(net, setName='val'):
	ioDat     = io.DataSet(cfg)
	testNames = ioDat.get_set_files(setName)
	testNames = testNames[0:100]
	opScale   = []
	for num in range(len(testNames)):
		print (num)
		if np.mod(num,100)==1:
			print (num)
		kpt    = imd.ImKPtDataMpii.from_file(testNames[num])
		N      = kpt.N_ #Number of people
		cropIm = []
		for s in LIST_SCALES:
			cropIm.append(kpt.crop_at_scale(scale=s, cropSz=224))
		ops = []
		for  n in range(N):
			imData = np.zeros((len(LIST_SCALES),224, 224,3))
			for i,s in enumerate(LIST_SCALES):
				imData[i,:,:,:] = cropIm[i][n]
			opDat    = net.forward(blobs=['fc-op'], data=imData)
			ops.append(LIST_SCALES[opDat['fc-op'].argmax()])
		opScale.append(ops)
	return testNames, opScale


def save_scales(setName='val'):
	net = get_scale_net()
	fNames, scale = predict_scale(net, setName)	
	svName = 'tmp-ief-scale-%s.pkl' % setName
	pickle.dump({'fNames': fNames, 'scale': scale}, open(svName, 'w'))


def vis_scale(net, scales, isPlot=True):
	plt.ion()
	ax1 = plt.subplot(1,2,1)
	ax2 = plt.subplot(1,2,2)
	ax1.set_title('Original Image')	
	ax2.set_title('Scale Crop')
	ioDat     = io.DataSet(cfg)
	testNames = ioDat.get_set_files('test')
	gtScaleDist, pdScaleDist = [], []
	for num in range(10):
		kpt    = imd.ImKPtDataMpii.from_file(testNames[num])
		N      = kpt.N_ #Number of people
		assert(N == len(scales[num]))
		for n in range(N):
			print (num, n)
			cIm = kpt.crop_at_scale(scale=scales[num][n], cropSz=224)
			ax2.imshow(cIm[n])
			pIm = kpt.get_im_with_point(kpt.bodyPos_[n,:], ptSz=51)
			h, w, ch = pIm.shape
			side     = min(h,w)
			if len(kpt.bodyScale_[0]) > 0:
				#Find gtscale distance
				gtScale = kpt.bodyScale_[0][n] * (200.0 / side)
				gtDist  = np.min(np.abs(np.array(LIST_SCALES) - gtScale))
				gtScaleDist.append(np.exp(-gtDist))
				pdScaleDist.append(np.exp(-np.abs(scales[num][n] - gtScale)))
			if isPlot:	
				ax1.imshow(pIm)
				plt.draw()
				inp = raw_input()	
				if inp=='q':
					return
	return np.array(gtScaleDist), np.array(pdScaleDist)
