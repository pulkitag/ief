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

LIST_SCALES = [1.4142, 1.1892, 1, 0.8409, 0.7071, 0.5946, 0.5, 0.4204, 0.3536,  0.2973]  

def get_scale_net():
	modelDir = '/work4/pulkitag-code/code/ief/IEF/models/'
	netName  = osp.join(modelDir, 'scalesel-vggs-epoch-14-convert.caffemodel')
	defFile  = osp.join(modelDir, 'vgg_s.prototxt')
	net      = mp.MyNet(defFile, netName, deviceId=1)
	#Set preprocessing in the net #Mean is in BGR because of channel swap
	net.set_preprocess(meanDat=(115.2254, 123.9648, 124.2966)) 
	return net

def get_ief_net():
	modelDir = '/work4/pulkitag-code/code/ief/IEF/models/'
	netName  = osp.join(modelDir, 'ief-googlenet-dec2015.caffemodel')
	defFile  = osp.join(modelDir, 'ief-googlenet-dec2015.prototxt')
	metaFile = osp.join(modelDir, 'ief-googlenet-dec2015-meta.pkl')
	net      = mp.MyNet(defFile, netName, deviceId=1)
	#Set preprocessing in the net
	#As of now the ief net takes RGB images, but the scale net takes BGR
	net.set_preprocess(chSwap=None, meanDat=(117.3785, 117.6438, 110.1771), ipName='image')
	net.set_preprocess(ipName='kp_pos', noTransform=True)
	net.set_preprocess(ipName='label',  noTransform=True)
	#Get the metadata
	metaData = pickle.load(open(metaFile, 'r')) 
	return net, metaData

def init_nets():
	netScale          = get_scale_net()
	netPose, metaPose =  get_ief_net() 
	return netScale, netPose, metaPose

##
#Predict pose for a given image
def predict(imName='./test_images/mpii-test-079555750.jpg', 
						bodyPt=(249,249), cropSz=256, poseImSz=224, 
						netScale=None, netPose=None, metaPose=None):

	#Get net information
	if netScale is None:
			netScale, netPose, metaPose = init_nets()
	seedPose       = metaPose['seedPose']
	mxStepSz       = metaPose['mxStepNorm'] 

	#Get scale
	#im     = scm.imread(imName)
	#imData = np.zeros((len(LIST_SCALES),cropSz, cropSz, 3))
	#for i,s in enumerate(LIST_SCALES):
	#	imData[i] = imu.centered_crop(cropSz, copy.deepcopy(im), bodyPt, s)
	svFile = '/home/eecs/pulkitag/Downloads/IEF-release-v1/cropims.mat'
	dat    = sio.loadmat(svFile)
	imData = np.zeros((len(LIST_SCALES),cropSz, cropSz, 3))
	for i, s in enumerate(LIST_SCALES):
		imData[i] = dat['ims'][i]
		

	#scaleDat  = netScale.forward(blobs=['fc-op'], data=imData)
	#scaleSc   = scaleDat['fc-op']
	#scale    = LIST_SCALES[scaleSc.argmax()]
	#print (scale)
	scale    = 1.1892
	scaleIdx = 1

	#Make pose prediction
	#imScale = imu.centered_crop(cropSz, copy.deepcopy(im), bodyPt, scale)
	imScale  = imData[scaleIdx]
	xSt, ySt = (cropSz - poseImSz)/2, (cropSz - poseImSz)/2
	xEn, yEn = xSt + poseImSz, ySt + poseImSz 
	imScale  = imScale[ySt:yEn, xSt:xEn,:].reshape((1,poseImSz,poseImSz,3))
	currPose        = np.zeros((1,17,2,1)).astype(np.float32)
	for i in range(17):
		currPose[0,i,0] = copy.deepcopy(seedPose[0,i] - xSt)
		currPose[0,i,1] = copy.deepcopy(seedPose[1,i] - ySt)
	#The marking point is the center of the image
	currPose[0, 16, 0] = poseImSz / 2
	currPose[0, 16, 1] = poseImSz / 2
	labels = np.zeros((1,16,2,1)).astype(np.float32)
	for step in range(4):
		opDat    = netPose.forward(blobs=['cls3_fc', 'rendering', 'input', 'conv1', 'conv1relu',
								'pool1', 'cls3_pool', 'icp5_reduction1relu', 'icp7_pool', 'icp9_reduction1',
								'icp9_out', 'icp9_out1'], 
							 image=imScale,
							 kp_pos=copy.deepcopy(currPose), label=labels)
		kPred    = copy.deepcopy(opDat['cls3_fc'].squeeze())
		gRender  = copy.deepcopy(opDat['rendering'].squeeze())
		pdb.set_trace()
		for i in range(16):
			dx, dy = kPred[i], kPred[16 + i]
			currPose[0,i,0] = currPose[0,i,0] + mxStepSz * dx
			currPose[0,i,1] = currPose[0,i,1] + mxStepSz * dy

