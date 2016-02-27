# --------------------------------------------------------
# IEF
# Copyright (c) 2015
# Licensed under BSD License [see LICENSE for details]
# Written by Joao Carreira, Pulkit Agrawal and Katerina Fragkiadki
# --------------------------------------------------------

try:
	import my_pycaffe as mp
except:
	from pycaffe_utils import my_pycaffe as mp
from utils import imdata as imd
from utils import io
from utils import visualization as vis
from utils import imutils as imu
from config import cfg
from os import path as osp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
import scipy.io as sio
import scipy.misc as scm
import pdb

LIST_SCALES = cfg.SCALE_LAMBDA  

##
#Scale Network
def get_scale_net(isGPU=True, deviceId=0):
	netName  = cfg.SCALE_MODEL.NET
	defFile  = cfg.SCALE_MODEL.PROTO
	net      = mp.MyNet(defFile, netName, isGPU = isGPU,  deviceId = deviceId)
	#Set preprocessing in the net
	net.set_preprocess(meanDat=(115.2254, 123.9648, 124.2966)) 
	return net

##
#Pose Network
def get_pose_net(isGPU=True, deviceId=0):
	netName  = cfg.POSE_MODEL.NET
	defFile  = cfg.POSE_MODEL.PROTO
	metaFile = cfg.POSE_MODEL.META
	net      = mp.MyNet(defFile, netName, isGPU = isGPU, deviceId = deviceId)
	#Set preprocessing in the net
	#As of now the ief net takes RGB images, but the scale net takes BGR
	net.set_preprocess(chSwap=None, meanDat=(117.3785, 117.6438, 110.1771), ipName='image')
	net.set_preprocess(ipName='kp_pos', noTransform=True)
	net.set_preprocess(ipName='label',  noTransform=True)
	#Get the metadata
	metaData = pickle.load(open(metaFile, 'r')) 
	return net, metaData

##
# Predicting Poses
class PoseIEF(object):
	def __init__(self, netScale=None, netPose=None, metaPose=None, cropSz=256, poseImSz=224, isGPU=True, deviceId=0):
		if netScale is  None:
			netScale = get_scale_net(isGPU=isGPU,deviceId=deviceId)
		if netPose is None:
			netPose, metaPose = get_pose_net(isGPU=isGPU,deviceId=deviceId)
		#The two nets
		self.netScale_ = netScale
		self.netPose_  = netPose
		#Meta information needed
		self.seedPose_ = metaPose['seedPose']
		self.mxStepSz_ = metaPose['mxStepNorm'] 
		self.cropSz_   = cropSz
		self.poseImSz_ = poseImSz

	##
	#Predict pose
	def predict(self, imName='./test_images/mpii-test-079555750.jpg', 
							bodyPt=(249,249), returnIm=False):
		'''
			imName  : image file name for which the pose needs to be predicted
			bodyPt  : A point on the body of the person (torso) for whom the pose 
							  is to be predicted
			returnIm: If True, return the image also
		'''
		cropSz, poseImSz = self.cropSz_, self.poseImSz_
		#Read the image
                if(isinstance(imName, str)):
                        im = scm.imread(imName)
                else:
                        im = imName
		
		#Crop the image at different scales
		imData  = np.zeros((len(LIST_SCALES), cropSz, cropSz, 3))
		scData  = np.zeros((len(LIST_SCALES), 2))
		posData = np.zeros((len(LIST_SCALES), 2))
		for i,s in enumerate(LIST_SCALES):
			imData[i], scs, crpPos = imu.centered_crop(cropSz, copy.deepcopy(im), bodyPt, s, 
												returnScale=True)
			scData[i]  = np.array(scs).reshape(1,2)	
			posData[i] = np.array(crpPos).reshape(1,2)
	
		#Use the scale net to find the best scale
		scaleOp  = self.netScale_.forward(blobs=['fc-op'], data=imData)
		scaleIdx = scaleOp['fc-op'].squeeze().argmax()
		scale    = LIST_SCALES[scaleIdx]
		#Scale to use to return the image in the original space
		oScale   = scData[scaleIdx]
		#Original location of the cropped image
		oPos     = posData[scaleIdx]

		#Prepare image for pose prediction	
		imScale  = imData[scaleIdx]
		xSt, ySt = (cropSz - poseImSz)/2, (cropSz - poseImSz)/2
		xEn, yEn = xSt + poseImSz, ySt + poseImSz 
		imScale  = imScale[ySt:yEn, xSt:xEn,:].reshape((1,poseImSz,poseImSz,3))
	
		#Seed pose
		currPose        = np.zeros((1,17,2,1)).astype(np.float32)
		for i in range(16):
			currPose[0,i,0] = copy.deepcopy(self.seedPose_[0,i] - xSt)
			currPose[0,i,1] = copy.deepcopy(self.seedPose_[1,i] - ySt)
		#The marking point is the center of the image
		currPose[0, 16, 0] = poseImSz / 2
		currPose[0, 16, 1] = poseImSz / 2
	
		#Dummy labels	
		labels = np.zeros((1,16,2,1)).astype(np.float32)

		#Predict Pose
		for step in range(4):
			poseOp = self.netPose_.forward(blobs=['cls3_fc'], image=imScale,
							 kp_pos=copy.deepcopy(currPose), label=labels)
			kPred    = copy.deepcopy(poseOp['cls3_fc'].squeeze())
			for i in range(16):
				dx, dy = kPred[i], kPred[16 + i]
				currPose[0,i,0] = currPose[0,i,0] + self.mxStepSz_ * dx
				currPose[0,i,1] = currPose[0,i,1] + self.mxStepSz_ * dy
		
		#Convert the pose in the original image coordinated
		origPose = (currPose.squeeze() +  np.array([xSt, ySt]).reshape(1,2)) * oScale + oPos

		if returnIm:
			#return origPose, copy.deepcopy(currPose), imScale[0]
			return origPose, im
		else:
			return origPose, copy.deepcopy(currPose)


##
#Predict poses for all validation set examples in mpii
def pred_mpii_val(isPlot=False):
	if isPlot:
		plt.ion()
		ax = plt.subplot(111)		

	#Get IEF object
	ief = PoseIEF()

	#Get MPII val file names
	ioDat     = io.DataSet(cfg)
	valNames  = ioDat.get_set_files('val')

	poses     = np.zeros((len(valNames), 17, 2))
	cropPoses = np.zeros((len(valNames), 17, 2))
	#ims   = np.zeros((len(valNames), 224, 224, 3)).astype(np.uint8)
	for i, name in enumerate(valNames):
		print (i)
		data    = imd.ImKPtDataMpii.from_file(name)
		bodyPt  = data.bodyPos_.squeeze()
		imFile  = data.imFile_
		#pose, im = ief.predict(imFile, bodyPt, returnIm=True)
		pose, cropPose = ief.predict(imFile, bodyPt, returnIm=False)
		poses[i]     = pose.squeeze()
		cropPoses[i] = cropPose.squeeze()
		#ims[i]   = im.astype(np.uint8)
		if isPlot:
			ax.clear()
			vis.plot_pose_stickmodel(im.astype(np.uint8), pose.squeeze().transpose((1,0)), ax)
			plt.draw()
			plt.show()
			ip = raw_input()
			if ip=='q':
				return
	sio.savemat('val_pred.mat', {'poses': poses, 'cropPoses': cropPoses})
			
