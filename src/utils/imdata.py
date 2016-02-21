## @package imdata
# Defines Classes for storing the data
#
# --------------------------------------------------------
# IEF
# Copyright (c) 2015
# Licensed under BSD License [see LICENSE for details]
# Written by Joao Carreira, Pulkit Agrawal and Katerina Fragkiadki
# --------------------------------------------------------

import matplotlib.pyplot as plt
import copy
import scipy.misc as scm
from . import visualization as vis
from . import imutils as imu
import numpy as np
try:
	import h5py as h5
except:
	print('WARNING: h5py not found, some functions may not work')
##
# Parent class for data objects
class ImData(object):
	def __init__ (self, imFile=''):
		self.imFile_ = imFile

class Bbox(object):
	def __init__ (self, x1, y1, x2, y2):
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2

	def plot(self, col='r', th=3.0):
		pass	
			
	
class ImKPtData(ImData):
	def __init__ (self, imFile, kpts, kptsVis=None):
		'''
			imFile: Path to image file
			kpts  : N * K * 2
							N - number of persons in the image
							K - number of keypoints per person
							2 - x,y coordinate of the keypoint
		'''
		self.imFile_ = imFile
		self.im_     = scm.imread(imFile)
		self.kpts_   = copy.deepcopy(kpts)
		assert kpts.ndim==3, 'kpts should be 3-D array'
		N, K, d = kpts.shape
		assert d==2
		if kptsVis is None:
			kptsVis = np.ones((N,K))	
		self.kptsVis_ = copy.deepcopy(kptsVis)
		self.N_       = N
		self.K_       = K

	@classmethod
	def from_file(cls, fName):
		with h5.File(fName, 'r') as f:
			imFile  = [chr(c) for c in f['imgName']]
			imFile  = ''.join(imFile)
			kpts    = np.array(f['kpts'])
			kpts    = kpts.transpose((2,1,0))
			nPeople = kpts.shape[0]
			kptsVis = np.array(f['kptsVis'])
			if kptsVis.shape[1] == nPeople:
				kptsVis = kptsVis.transpose((1,0))
			else:
				print kpts.shape
				assert kptsVis.shape[0] == nPeople, '%s, shp: %d,%d' % (nPeople, kptsVis.shape[0], kptsVis.shape[1])
			self = cls(imFile, kpts=kpts, kptsVis=kptsVis)
			return self

	##
	#Generate the image for plotting the keypoints
	def get_plot_data(self, col='r', kSz=11):
		pIm = copy.copy(self.im_)
		#Some keypoints maybe marked outside 
		#so account for that
		x1 = np.min(self.kpts_[:,:,0] - kSz/2)
		y1 = np.min(self.kpts_[:,:,1] - kSz/2)
		x2 = np.max(self.kpts_[:,:,0] + kSz/2)
		y2 = np.max(self.kpts_[:,:,1] + kSz/2)
		pIm, pd = imu.pad_to_fit(pIm, (x1, x2, y1, y2)) 
		xpd,_,ypd,_ = pd

		for n in range(self.N_):
			for k in range(self.K_):
				#if self.kptsVis_[n,k]:
				x, y = self.kpts_[n,k]
				x += xpd
				y += ypd	
				pIm[y-kSz/2:y+kSz/2, x-kSz/2:x+kSz/2,0] = 255	
		return pIm

	##
	#Plot the keypoints
	def plot(self, col='r', kSz=11):
		pIm = self.get_plot_data(col=col, kSz=kSz)
		plt.ion()
		plt.imshow(pIm)


class ImKPtDataMpii(ImKPtData):
	def __init__(self, imFile, kpts, kptsVis, bodyPos=None, bodyScale=None):
		super(ImKPtDataMpii, self).__init__(imFile=imFile, kpts=kpts, kptsVis=kptsVis)
		self.bodyPos_   = copy.copy(bodyPos)
		self.bodyScale_ = copy.copy(bodyScale)

	@classmethod
	def from_file(cls, fName):
		self = super(ImKPtDataMpii, cls).from_file(fName)			 	
		with h5.File(fName, 'r') as f:
			self.bodyPos_   = copy.copy(np.array(f['objPosxy'])).transpose((1,0))
			self.bodyScale_ = copy.copy(np.array(f['scale'])) 
		return self

	def get_plot_data(self, kPtCol='r', kPtSz=11, bodyPtCol='b', bodyPtSz=21):
		#The image with keypoints plotted
		pIm = super(ImKPtDataMpii, self).get_plot_data(kPtCol, kPtSz)
		#Now plot the body location
		x1 = np.min(self.bodyPos_[:,0] - bodyPtSz/2)
		y1 = np.min(self.bodyPos_[:,1] - bodyPtSz/2)
		x2 = np.max(self.bodyPos_[:,0] + bodyPtSz/2)
		y2 = np.max(self.bodyPos_[:,1] + bodyPtSz/2)
		pIm, pd = imu.pad_to_fit(pIm, (x1, x2, y1, y2)) 
		xpd,_,ypd,_ = pd
		for n in range(self.N_):
				#if self.kptsVis_[n,k]:
				x, y = self.bodyPos_[n]
				x += xpd
				y += ypd	
				pIm[y-bodyPtSz/2:y+bodyPtSz/2, x-bodyPtSz/2:x+bodyPtSz/2,2] = 255	
		return pIm

	def plot(self, **kwargs):
		pIm = self.get_plot_data(**kwargs)
		plt.ion()
		plt.imshow(pIm)

	def get_im_with_point(self, pt, kPtCol='r', kPtSz=11, ptSz=21):
		pIm  = copy.copy(self.im_)
		x, y = pt
		x1 = max(0,np.min(x - ptSz/2))
		y1 = max(0,np.min(y - ptSz/2))
		x2 = min(pIm.shape[1],np.max(x + ptSz/2))
		y2 = min(pIm.shape[0],np.max(y + ptSz/2))
		pIm[y1:y2, x1:x2, 0] = 255	
		return pIm
	
	def crop_at_scale(self, scale=1.2, cropSz=224):
		crpIm = []
		for n in range(self.N_):
			crpIm = crpIm + [imu.centered_crop(cropSz, self.im_, self.bodyPos_[n], scale)]
		return crpIm
