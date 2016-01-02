from PIL import Image
import numpy as np
import copy
import easydict as edict
from os import path as osp


def save_blob_as_ims(datablob, filename, PIXEL_MEANS):
   img = datablob[:,:,0:3]
   save_image_from_array(img+PIXEL_MEANS,"./"+filename+"img.png")
   shape = datablob.shape
   nmaps = shape[2]-3
   for i in range(0,nmaps):
     map = datablob[:,:,2+i+1:2+i+2]
     imgmap=(img+PIXEL_MEANS+255*np.dstack((map,map,map)))/2 
     save_image_from_array(imgmap,"./"+filename+"map"+str(i)+".png")


def save_im (img,filename):
  imgshow = copy.deepcopy(img)
  imgshow = imgshow[:,:,::-1]
  imgshow = Image.fromarray(imgshow.astype(np.uint8))
  imgshow.save(filename)


##
# For reading the dataset data
class DataSet(object):
	def __init__(self, cfg):
		self.setFile_   = edict.EasyDict()
		self.setNames_  = ['train', 'val', 'test'] 
		for s in self.setNames_:
			self.setFile_[s] = cfg.PATHS.SET_FILE % s 
			assert osp.isfile(self.setFile_[s]), '%s not found' % setFile_[s]
		self.dataFile_ = cfg.PATHS.DATA_FILE

	#The names of files belonging to a set	
	def get_set_ids(self, setName):
		with open(self.setFile_[setName]) as f:
			names = f.readlines()
			names = [n.strip() for n in names]
		return names

	#Return the datafiles belonging to a set
	def get_set_files(self, setName):
		names     = self.get_set_ids(setName)
		dataFiles = [self.dataFile_ % ('%06d' % int(n)) for n in names]
		return dataFiles 
