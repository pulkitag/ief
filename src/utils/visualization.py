# --------------------------------------------------------
# IEF
# Copyright (c) 2015
# Licensed under BSD License [see LICENSE for details]
# Written by Joao Carreira, Pulkit Agrawal and Katerina Fragkiadki
# --------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

def plot_pose_stickmodel(im, kpts, ax=None, pad=5, lw=3):
	'''
		im  : image
		kpts: key points 2 x N, where N is the number of keypoints
					(x,y) format 
		pad : padding during visualization
		lw  : line width 
	'''
	plt.ion()
	if ax is None:
		ax = plt.subplot(111)
	#Increase the size of the underlying image to account for extended keypoints
	nr, nc, _ = im.shape
	mnX, mnY  = np.min(kpts, axis=1)
	mxX, mxY  = np.max(kpts, axis=1)
	dx, dy    = abs(min(0, mnX)) + pad,  abs(min(0, mnY)) + pad
	xEn, yEn  = max(nc, mxX) + dx + pad, max(nr, mxY) + dy + pad
	imPlot    = np.zeros((yEn, xEn, im.shape[2])).astype(np.uint8)
	imPlot[dy:dy+nr, dx:dx+nc,:] = im
	dd   = np.array([dx, dy]).reshape(2,1)
	kpts = kpts + dd 
	ax.imshow(imPlot)
	#Plot the keypoints - this works for MPII style keypoints
	#Right leg
	ax.plot(kpts[0, [0,1]], kpts[1, [0, 1]], 'r-', linewidth=lw)		
	ax.plot(kpts[0, [1,2]], kpts[1, [1, 2]], 'r-', linewidth=lw)		
	#Right arm
	ax.plot(kpts[0, [10,11]], kpts[1, [10, 11]], 'r-', linewidth=lw)		
	ax.plot(kpts[0, [11,12]], kpts[1, [11, 12]], 'r-', linewidth=lw)		
	#thorax-right arm	
	ax.plot(kpts[0, [7,12]], kpts[1, [7, 12]], 'r-', linewidth=lw)		
	#right hip-pelvis
	ax.plot(kpts[0, [2,6]], kpts[1, [2, 6]], 'r-', linewidth=lw)		
	#left arm	
	ax.plot(kpts[0, [13,14]], kpts[1, [13, 14]], 'g-', linewidth=lw)		
	ax.plot(kpts[0, [14,15]], kpts[1, [14, 15]], 'g-', linewidth=lw)		
 	#thorax - leftarm 
	ax.plot(kpts[0, [7,13]], kpts[1, [7, 13]], 'g-', linewidth=lw)		
	#left leg
	ax.plot(kpts[0, [3, 4]], kpts[1, [3, 4]], 'g-', linewidth=lw)		
	ax.plot(kpts[0, [4, 5]], kpts[1, [4, 5]], 'g-', linewidth=lw)		
	#left hip - pelvis
	ax.plot(kpts[0, [3, 6]], kpts[1, [3, 6]], 'g-', linewidth=lw)		
	#pelvis - thorax
	ax.plot(kpts[0, [6, 7]], kpts[1, [6, 7]], 'y-', linewidth=lw)		
	#thorax - upper neck
	ax.plot(kpts[0, [7, 8]], kpts[1, [7, 8]], 'y-', linewidth=lw)		
	#upper neck - head
	ax.plot(kpts[0, [8, 9]], kpts[1, [8, 9]], 'y-', linewidth=lw)		
	plt.draw()

def plot_gauss_maps(imgs, ax=None, offset=50.0):
	'''
		imgs  : Gaussian renderings
		offset: to make all values be > 0 
	'''
	assert imgs.shape[0] == 17
	if ax is None:
		ax = []
		for i in range(16):
			ax.append(plt.subplot(4, 4, i+1))
	right = [0, 1, 2, 10, 11, 12]
	left  = [3, 4, 4, 13, 14, 15]
	for i in range(16):
		im = np.zeros(imgs[i].shape + (3,))
		if i in right:
			im[:,:,0] = 1.0 #Make the color red
		elif i in left:
			im[:,:,1] = 1.0 #Make the color green
		else:
			im[:,:,0:2] = 1.0 #Make yellow	
		im = (im * (imgs[i].reshape(imgs[i].shape + (1,)) + offset)).astype(np.uint8)
		ax[i].imshow(im)
	plt.draw()
	plt.show()	
 
