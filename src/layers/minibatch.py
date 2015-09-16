
"""Compute minibatch blobs for training IEF network."""

import numpy as np
import numpy.random as npr
import cv2
from IEF.config import cfg
from utils.blob import  im_list_to_blob
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc 
import utils.visualization as vis
import copy
import time
from utils import imutils as imu


def get_minibatch(imdata,gaussian_width):
    """Given imdata, construct a minibatch."""
    num_images = len(imdata)
    assert(cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
        format(num_images, cfg.TRAIN.BATCH_SIZE)
    print("preparing image blob...")
    tic = time.clock()
    im_blob = get_image_blob_from_imdata(imdata,gaussian_width)
    toc = time.clock()
    print("elapsed time: ",toc-tic)
    print("get target blob")
    tic = time.clock()
    target_blob= get_target_blob_from_imdata(imdata)
    toc = time.clock()
    print("elapsed time: ",toc-tic)
    #target_blob= get_target_blob_from_imdata(imdata)
    
    blobs = {'data': im_blob,
             'targets': target_blob}

    return blobs

  #plt.plot(np.ndarray(meankpts_coords[0,[0,3]], kpts_coords[1,[0,3]],'r-')
  

def get_target_blob_from_imdata(imdata, mode="error"):
  
	num_images    = len(imdata)
	shapei        = (imdata[0]._coords_init).shape
	num_keypoints = shapei[1]
	nchannels     = num_keypoints*2
	blob          = np.zeros((num_images, nchannels, 1, 1),
						dtype=np.float32)

	for i in xrange(num_images):
		if mode=="init":
			kpts_coords = imdata[i]._coords_init
		if mode=="target":
			kpts_coords = imdata[i]._coords_target 
		if mode=="predicted":
			kpts_coords = imdata[i]._coords_predicted
		if mode=="error":
			kpts_coords = imdata[i]._error_targets
		blob[i,0:nchannels,0,0] = np.reshape(kpts_coords,
					                       (1,shapei[0]*shapei[1]))
	return blob  


def get_image_blob_from_imdata(imdata, gaussian_width):
	"""
		Builds an input blob from the images in the imdata at the specified
		scales.
	"""
	num_images = len(imdata)
	processed_ims = []
	for i in xrange(num_images):
		#im = cv2.imread(imdata[i]._imname)
		inputoutput=get_im_inout(imdata[i]._imname,imdata[i]._bbox,imdata[i]._coords_init,gaussian_width)
		processed_ims.append(inputoutput)

	# Create a blob to hold the input images
	blob = im_list_to_blob(processed_ims)
	#toc = time.clock()
	#toc - tic
	return blob


def vis_minibatch(im_blob,  targets_blob):
    # to do: add visualizatoion with arrows!!
    #here it is assumed the target holds the final pose!
    """Visualize a mini-batch for debugging."""
    import matplotlib.pyplot as plt
    target_shape=targets_blob.shape
    num_keypoints=target_shape[1]/2
    for i in xrange(im_blob.shape[0]):
        kpt=np.reshape(targets_blob[i,:,0,0],(2, num_keypoints))
        iminout = copy.deepcopy(im_blob[i, :])
        iminout=iminout.transpose((1, 2, 0))
        #print(iminout.shape)
        #exit(1)
        im=iminout[:,:,0:3]
        
        #???
        #im=im.transpose((1, 2, 0))
        # add the mean
        im += cfg.PIXEL_MEANS
        # make rgb
        im = im[:, :, (2, 1, 0)]
        im = im.astype(np.uint8)
        #print("kpt:",kpt)
        #im=vis.draw_pts_onimg(im,kpt)
        #plt.imshow(im)
        #plt.show()
        #exit(1)

	for j in range(0,num_keypoints):
	  map=iminout[:,:,2+j+1:2+j+2]+cfg.PIXEL_MEAN
	  #print("map.shape:",map.shape,"img.shape:",im.shape)
	  #exit(1)
	  imgmap=(im+np.dstack((map,map,map)))/2 
	  #imgmap=vis.draw_pts_onimg(imgmap,kpt)
	  plt.imshow(imgmap.astype(np.uint8))
	  plt.title('map'+str(j))
	  plt.show()
	  

       
    
def render_kpts_square(kpts,gaussian_width,square_width):
      #kpts: 2 X num_keypoints
      # from keypoints to gaussian maps!	
      size=kpts.shape
      num_keypoints=size[1]
      #print("numkeypts:",numkeypts)
      gaussianmaps=np.ndarray((square_width,square_width,num_keypoints))
      for i in range(0,num_keypoints):
	gaussianmaps[:,:,i]=vis.makeGaussian(square_width,
				      gaussian_width, 
				      kpts[:,i])-cfg.PIXEL_MEAN
	#plt.imshow(gaussianmaps[:,:,i])
	#plt.show()
	if 0:
	  gaussianmap=gaussianmaps[:,:,i]+cfg.PIXEL_MEAN
	  result = Image.fromarray((gaussianmap).astype(np.uint8))
	  result.save('./plots/gaussianmap'+str(i)+'.png')
	  imgdraw=vis.draw_pts_onimg(gaussianmap,kpts)
	  result = Image.fromarray((imgdraw).astype(np.uint8))
	  result.save('./plots/imdraw'+str(i)+'.png')
      #exit(1)
      return gaussianmaps


def get_im_inout(imname,bbox,kpts,gaussian_width):
	#tic = time.clock() 
	im = cv2.imread(imname) 
	#toc = time.clock()
	#print("elapsed time: ",toc-tic)
	im_windowed = imu.crop_with_warp(cfg, im, bbox)
	#tic = time.clock() 
	gaussian_maps=render_kpts_square(kpts,gaussian_width,cfg.CROP_SIZE)
	#toc = time.clock()
	#print("elapsed time: ",toc-tic)

	inputoutput=np.dstack((im_windowed,gaussian_maps))
	return inputoutput	





