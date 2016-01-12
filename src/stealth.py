import my_pycaffe as mp
from utils import imdata as imd
from utils import io
from utils import visualization as vis
from config import cfg
from os import path as osp
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy
import scipy.io as sio
#import matlab.engine
#import matlab
#meng = matlab.engine.start_matlab()

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

	
def predict_pose(setName='val', cropSz=256, poseImSz=224):
	#Read the scales
	scaleFile = 'tmp-ief-scale-%s.pkl' % setName
	data      = pickle.load(open(scaleFile,'r'))
	fNames, scales = data['fNames'][0:25], data['scale'][0:25]
	#Get the network
	net, metaData  = get_ief_net()
	seedPose       = metaData['seedPose']
	mxStepSz       = metaData['mxStepNorm'] 
	#Go over the images and predict poses	
	allPoses, allIms, allGrs = [], [], []
	for num, sc in zip(range(len(fNames)), scales):
		print (num)
		kpt    = imd.ImKPtDataMpii.from_file(fNames[num])
		N      = kpt.N_ #Number of people
		ops = []
		ims = []
		grs = []
		for  n in range(N):
			cropIm = kpt.crop_at_scale(scale=sc[n], cropSz=cropSz)
			#Get the center patch
			xSt, ySt = (cropSz - poseImSz)/2, (cropSz - poseImSz)/2
			xEn, yEn = xSt + poseImSz, ySt + poseImSz 
			imData = np.zeros((1,poseImSz,poseImSz,3)).astype(np.float32)
			imData[0,:,:,:] = cropIm[n][ySt:yEn, xSt:xEn, :]
			currPose        = np.zeros((1,17,2,1)).astype(np.float32)
			for i in range(17):
				currPose[0,i,0] = copy.deepcopy(seedPose[0,i] - xSt)
				currPose[0,i,1] = copy.deepcopy(seedPose[1,i] - ySt)
			#The marking point is the center of the image
			currPose[0, 16, 0] = poseImSz / 2
			currPose[0, 16, 1] = poseImSz / 2
			labels = np.zeros((1,16,2,1)).astype(np.float32)
			poseSteps, grSteps = [], []
			poseSteps.append(copy.deepcopy(currPose))
			for step in range(4):
				opDat    = net.forward(blobs=['cls3_fc', 'rendering', 'input'], image=imData,
									 kp_pos=copy.deepcopy(currPose), label=labels)
				kPred    = copy.deepcopy(opDat['cls3_fc'].squeeze())
				gRender  = copy.deepcopy(opDat['rendering'].squeeze())
				grSteps.append(gRender)
				for i in range(16):
					dx, dy = kPred[i], kPred[16 + i]
					currPose[0,i,0] = currPose[0,i,0] + mxStepSz * dx
					currPose[0,i,1] = currPose[0,i,1] + mxStepSz * dy
				poseSteps.append(copy.deepcopy(currPose))
			ops.append(poseSteps)
			ims.append(imData[0])
			grs.append(grSteps)
		allPoses.append(ops)
		allIms.append(ims)
		allGrs.append(grs)
	return fNames, allPoses, scales, allIms, allGrs

def save_predictions(setName='val'):
	svName = 'tmp-ief-pose-%s.pkl' % setName
	fNames, allPoses, scales, allIms, allGrs = predict_pose(setName=setName)
	pickle.dump({'fNames': fNames, 'allPoses': allPoses, 
							 'scales': scales, 'allIms': allIms, 'allGrs': allGrs}, 
								open(svName, 'w')) 

def plot_predictions(setName='val'):
	plt.close('all')
	#figRen = plt.figure()
	figVis = plt.figure()
	plt.ion()
	#Form the axes for keypoint visualization
	ax    = []
	count = 1
	for i in range(2):
		for j in range(2):
			ax.append(figVis.add_subplot(2, 2, count))
			count += 1
	#Form the axes for gaussian visualization
	#axRen = []
	#for i in range(16):
	#	axRen.append(figRen.add_subplot(4,4,i+1))

	fName = 'tmp-ief-pose-%s.pkl' % setName
	dat   = pickle.load(open(fName, 'r'))
	allPoses, allIms, allGrs = dat['allPoses'], dat['allIms'], dat['allGrs']
	for ims, poses, grs in zip(allIms, allPoses, allGrs):
		for n in range(len(ims)):
			im   = ims[n]
			gr   = grs[n]
			pSeq = poses[n]
			#Plot the stick figure
			plt.figure(figVis.number) 							
			for i in range(4):
				print im.shape, pSeq[i].shape
				vis.plot_pose_stickmodel(im, pSeq[i+1].squeeze().transpose((1,0)), ax[i])
				plt.draw()
				plt.show()
			#Plot the gaussian renderings
			#plt.figure(figRen.number) 							
			#vis.plot_gauss_maps(gr[0], ax=axRen)
			ip = raw_input()
			if ip == 'q':
				return
			for i in range(4):
				ax[i].clear()


def predict_scale(net, setName='val', cropSz=256):
	ioDat     = io.DataSet(cfg)
	testNames = ioDat.get_set_files(setName)
	#testNames = testNames[0:100]
	opScale   = []
	for num in range(len(testNames)):
		print (num)
		if np.mod(num,100)==1:
			print (num)
		kpt    = imd.ImKPtDataMpii.from_file(testNames[num])
		N      = kpt.N_ #Number of people
		cropIm = []
		for s in LIST_SCALES:
			cropIm.append(kpt.crop_at_scale(scale=s, cropSz=cropSz))
		ops = []
		for  n in range(N):
			imData = np.zeros((len(LIST_SCALES),cropSz, cropSz, 3))
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

##
#Compare the scale predictions against that from matconvnet
def compare_scale():
	fName = 'tmp/val_scores_scale.mat'
	dat   = sio.loadmat(fName)
	pdScores = dat['all_pred']
	pSc      = np.zeros((2958,10))
	for  i in range(2958):
		pSc[i,:] = pdScores[0][range(i,29580,2958)].flatten()
	idx       = np.argmax(pSc, axis=1)
	matScales = [LIST_SCALES[i] for i in idx]
	#Load the python scales
	pyName    = 'tmp-ief-scale-val.pkl'
	pyData    = pickle.load(open(pyName, 'r'))
	scales    = pyData['scale']
	pyScales  = [sc for sublist in scales for sc in sublist]
	matScales = matScales[0:len(pyScales)]
	return pyScales, matScales			
		
