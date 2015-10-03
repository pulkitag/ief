import numpy as np
import scipy.io as sio
import caffe
import my_pycaffe as mp

class Scale(object):
	def __init__(self, cfg):
		#Setup the network
		self.net_    = mp.MyNet(cfg.SCALE_MODEL.PROTO, 
										cfg.SCALE_MODEL.NET)
		self.net_.setup_network()
		self.batchSz_ = self.net_.get_batchsz()
		
		#Setup the preprocessing
		self.net_.set_preprocess(meanDat=cfg.SCALE_MODEL.MEAN) 	
