import my_pycaffe as mp
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
def get_scale_net():
	netName  = cfg.SCALE_MODEL.NET
	defFile  = cfg.SCALE_MODEL.PROTO
	net      = mp.MyNet(defFile, netName, deviceId=1)
	#Set preprocessing in the net
	net.set_preprocess(meanDat=(115.2254, 123.9648, 124.2966)) 
	return net

##
#Pose Network
def get_pose_net():
	netName  = cfg.POSE_MODEL.NET
	defFile  = cfg.POSE_MODEL.PROTO
	metaFile = cfg.POSE_MODEL.META
	net      = mp.MyNet(defFile, netName, deviceId=1)
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
	def __init__(netScale=None, netPose=None, metaPose=None,
							 cropSz=256, poseImSz=224):
		if netScale is  None:
			netScale = get_scale_net()
		if netPose is None:
			netPose, metaPose = get_pose_net()
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
	def predict(imName='./test_images/mpii-test-079555750.jpg', 
							bodyPt=(249,249), returnIm=False):
		'''
			imName  : image file name for which the pose needs to be predicted
			bodyPt  : A point on the body of the person (torso) for whom the pose 
							  is to be predicted
			returnIm: If True, return the image also
		'''
		cropSz, poseImSz = self.cropSz_, self.poseImSz_
		#Read the image
		im     = scm.imread(imName)
		
		#Crop the image at different scales
		imData = np.zeros((len(LIST_SCALES), cropSz, cropSz, 3))
		for i,s in enumerate(LIST_SCALES):
			imData[i] = imu.centered_crop(cropSz, copy.deepcopy(im), bodyPt, s)
		
		#Use the scale net to find the best scale
		scaleOp  = self.netScale_.forward(blobs=['fc-op'], data=imData)
		scaleIdx = scaleDat['fc-op'].squeeze().argmax()
		scale    = LIST_SCALES[scaleIdx]

		#Prepare image for pose prediction	
		imScale  = imData[scaleIdx]
		xSt, ySt = (cropSz - poseImSz)/2, (cropSz - poseImSz)/2
		xEn, yEn = xSt + poseImSz, ySt + poseImSz 
		imScale  = imScale[ySt:yEn, xSt:xEn,:].reshape((1,poseImSz,poseImSz,3))
	
		#Seed pose
		currPose        = np.zeros((1,17,2,1)).astype(np.float32)
		for i in range(16):
			currPose[0,i,0] = copy.deepcopy(seedPose[0,i] - xSt)
			currPose[0,i,1] = copy.deepcopy(seedPose[1,i] - ySt)
		#The marking point is the center of the image
		currPose[0, 16, 0] = poseImSz / 2
		currPose[0, 16, 1] = poseImSz / 2
	
		#Dummy labels	
		labels = np.zeros((1,16,2,1)).astype(np.float32)

		#Predict Pose
		for step in range(4):
			poseOp = self.netPose_.forward(blobs=['cls3_fc'], image=imScale,
							 kp_pos=copy.deepcopy(currPose), label=labels)
			kPred    = copy.deepcopy(opDat['cls3_fc'].squeeze())
			pdb.set_trace()
			for i in range(16):
				dx, dy = kPred[i], kPred[16 + i]
				currPose[0,i,0] = currPose[0,i,0] + mxStepSz * dx
				currPose[0,i,1] = currPose[0,i,1] + mxStepSz * dy
		if returnIm:
			return copy.deepcopy(currPose), imScale
		else:
			return copy.deepcopy(currPose)


##
#Predict poses for all validation set examples in mpii
def pred_mpii_val():
	pass	
