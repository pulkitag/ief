import caffe
import numpy as np
import argparse, pprint
import scipy.misc as scm
from os import path as osp
from easydict import EasyDict as edict
import time
import glog
import pdb
import pickle
import matplotlib.pyplot as plt
import copy

class GaussRenderLayer(caffe.Layer):
	@classmethod
	def parse_args(cls, argsStr):
		parser = argparse.ArgumentParser(description='GaussRender Layer')
		parser.add_argument('--K', default=100.0, type=float)
		parser.add_argument('--T', default=-50.0, type=float)
		parser.add_argument('--sigma', default=0.001, type=float)
		parser.add_argument('--imgSz', default=224, type=int)
		args   = parser.parse_args(argsStr.split())
		print('Using Config:')
		pprint.pprint(args)
		return args

	def setup(self, bottom, top):
		self.param_ = GaussRenderLayer.parse_args(self.param_str)
		assert len(bottom) == 1, 'There should be 1 bottom blob'
		kpShape = bottom[0].data.shape
		batchSz, numKp, numCoords, _ = kpShape
		assert numCoords == 2, 'Keypoints are defined by 2-D coordinates'
		self.batchSz_ = batchSz
		self.nKp_     = numKp
		assert len(top)==1, 'There should be only one output blob'
		top[0].reshape(self.batchSz_, self.nKp_ , self.param_.imgSz, self.param_.imgSz)
		#Form the gaussian window	
		x = np.linspace(-self.param_.imgSz, self.param_.imgSz, 2 * self.param_.imgSz + 1)
		y = np.linspace(-self.param_.imgSz, self.param_.imgSz, 2 * self.param_.imgSz + 1)
		xx, yy = np.meshgrid(x, y)
		dist    = xx * xx + yy * yy;
		self.g_ = ((self.param_.K  * np.exp(-self.param_.sigma * dist)) + self.param_.T).astype(np.float32)
	
	def forward(self, bottom, top):
		for b in range(self.batchSz_):
			kps = bottom[0].data[b]
			top[0].data[b][...] = self.param_.T
			for k in range(self.nKp_):
				x, y     = kps[k]
				x, y     = int(round(x + 1)), int(round(y + 1))
				#Center at the gaussian windown
				delY, delX = self.param_.imgSz - y, self.param_.imgSz - x
				xSt, ySt   = max(0, delX), max(0, delY)
				yEn, xEn = delY + self.param_.imgSz, delX + self.param_.imgSz
				yEn      = min(2 * self.param_.imgSz + 1, yEn) 
				xEn      = min(2 * self.param_.imgSz + 1, xEn) 
				yImSt, xImSt = max(0, y - self.param_.imgSz), max(0, x - self.param_.imgSz)
				yImEn, xImEn = yImSt + (yEn - ySt), xImSt + (xEn - xSt)
				top[0].data[b][k,yImSt:yImEn,xImSt:xImEn] = copy.deepcopy(self.g_[ySt:yEn, xSt:xEn])
	
	def backward(self, top, propagate_down, bottom):
		pass	
	
	def reshape(self, bottom, top):
		pass


def test_render_layer(x=225, y = 250):
	net = caffe.Net('test/gauss_render.prototxt', caffe.TEST)
	plt.ion()
	fig = plt.figure()
	ax  = fig.add_subplot(1,1,1)
	pos = np.zeros((1,1,2,1)).astype(np.float32)
	pos[0,0,0] = x
	pos[0,0,1] = y
	data = net.forward(blobs=['gauss'], **{'kp': pos})
	print (data['gauss'][0].transpose((1,2,0)).shape)
	ax.imshow((data['gauss'][0].transpose((1,2,0)).squeeze() + 50.0).astype(np.uint8))
