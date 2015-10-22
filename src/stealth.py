import my_pycaffe as mp
from utils import imdata as imd
from utils import io
from config import cfg
from os import path as osp
import numpy as np

scales = [1.4142, 1.1892, 1, 0.8409, 0.7071, 0.5946, 0.5, 0.4204, 0.3536,  0.2973]  

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
		for s in scales:
			cropIm.append(kpt.crop_at_scale(scale=s, cropSz=224))
		ops = []
		for  n in range(N):
			imData = np.zeros((len(scales),224, 224,3))
			for i,s in enumerate(scales):
				imData[i,:,:,:] = cropIm[i][n]
			opDat    = net.forward(blobs=['fc-op'], data=imData)
			ops.append(scales[opDat['fc-op'].argmax()])
		opScale.append(ops)
	return opScale


def vis_scale(net, scales):
	ioDat     = io.DataSet(cfg)
	testNames = ioDat.get_set_files('test')
	
