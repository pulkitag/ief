import my_pycaffe as mp
from utils import imdata as imd
from utils import io
from config import cfg
from os import path as osp
import numpy as np
import matplotlib.pyplot as plt

LIST_SCALES = [1.4142, 1.1892, 1, 0.8409, 0.7071, 0.5946, 0.5, 0.4204, 0.3536,  0.2973]  

def get_net():
	modelDir = '/work4/pulkitag-code/code/ief/IEF/models/'
	netName  = osp.join(modelDir, 'scalesel-vggs-epoch-14-convert.caffemodel')
	defFile  = osp.join(modelDir, 'vgg_s.prototxt')
	net      = mp.MyNet(defFile, netName, deviceId=1)
	#Set preprocessing in the net
	net.set_preprocess(meanDat='/data1/pulkitag/caffe_models/ilsvrc2012_mean.binaryproto') 
	return net

def test(net):
	ioDat     = io.DataSet(cfg)
	testNames = ioDat.get_set_files('test')
	opScale   = []
	for num in range(10):
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
	return opScale


def vis_scale(net, scales):
	plt.ion()
	ax1 = plt.subplot(1,2,1)
	ax2 = plt.subplot(1,2,2)
	ax1.set_title('Original Image')	
	ax2.set_title('Scale Crop')
	ioDat     = io.DataSet(cfg)
	testNames = ioDat.get_set_files('test')
	for num in range(10):
		kpt    = imd.ImKPtDataMpii.from_file(testNames[num])
		N      = kpt.N_ #Number of people
		assert(N == len(scales[num]))
		for n in range(N):
			print (num, n)
			cIm = kpt.crop_at_scale(scale=scales[num][n], cropSz=224)
			ax2.imshow(cIm[n])
			pIm = kpt.get_im_with_point(kpt.bodyPos_[n,:], ptSz=51)
			ax1.imshow(pIm)
			plt.draw()
			inp = raw_input()	
			if inp=='q':
				return
