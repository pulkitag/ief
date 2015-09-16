# --------------------------------------------------------
# IEF
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Test a IEF network on an imkptdb (image  database)."""

from IEF.config import cfg#, get_output_dir
import argparse
from utils.timer import Timer
import numpy as np
import cv2
import caffe
#from utils.cython_nms import nms
import cPickle
import heapq
#from utils.blob import inputoutput_list_to_blob
import os
import datasets.imkptdb as imkptdb


##
#estimate the pose on an image inside the given bbox
#	Arguments:
#   net (caffe.Net): IEF network to use
#   im (ndarray): color image to test (in BGR order)
#   bbox (ndarray): 1X4 box of where to test the model
# Returns:
#		poses (ndarray):  Numkeypoints X 2 
#		scores(ndarray): R X 1
def im_parse_pose(net, imdata):
	for j in range(1,cfg.ERROR_UPDATE_STEPS):
		if j==1:
		 iminout= minibatch.get_im_inout(imdata._imname,imdata._bbox,imdata._coords_init,gaussian_width)
		 imdata._coords_predicted=imdata._coords_init
		else:
		 iminout= minibatch.get_im_inout(imdata._imname,imdata._bbox,imdata._coords_predicted,gaussian_width)

		data_blob = im_list_to_blob(iminout)  
	
		# reshape network inputs
		#net.blobs['data'].reshape(*(blobs['data'].shape))
		blobs_out = net.forward(data=data_blob)

		update_coords = blobs_out['update']
		imdata._coords_predicted=imdata._coords_predicted+
		np.reshape((np.multiply(update_coords,std)+mean),(2,13))

##
# Test IEF Model on an image database
def test_net(net, imkptdb):
	num_images = len(imkptdb.imdata_list)
	output_dir = imkptdb._output_dir
	if not os.path.exists(output_dir):
			os.makedirs(output_dir)

	# timers
	_t = {'im_detect' : Timer(), 'misc' : Timer()}

	for i in xrange(num_images):
		im = cv2.imread(imkptdb.imdata_list[i]._imname)
		_t['im_parse_pose'].tic()
		predicted_keypoints[i] = im_parse_pose(net, imkptdb.imdata_list[i])
		_t['im_parse_pose'].toc()

	if 0:
		#keep = nms(all_boxes[j][i], 0.3)
		imkptdb.imdata_list[i].plot()
		_t['misc'].toc()



	#print 'Evaluating detections'
	#imdb.evaluate_poses(predicted_keypoints)
