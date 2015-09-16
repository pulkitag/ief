import numpy as np
import numpy.random as npr
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc 
import copy


def save_datablob_to_file(datablob,filename,PIXEL_MEANS):
   img=datablob[:,:,0:3]
   save_image_from_array(img+PIXEL_MEANS,"./"+filename+"img.png")
   shape=datablob.shape
   nmaps=shape[2]-3
   for i in range(0,nmaps):
     map=datablob[:,:,2+i+1:2+i+2]
     #print("map.shape:",map.shape,"img.shape:",img.shape)
     #exit(1)
     imgmap=(img+PIXEL_MEANS+255*np.dstack((map,map,map)))/2 
     save_image_from_array(imgmap,"./"+filename+"map"+str(i)+".png")


def save_image_from_array(img,filename):
  imgshow = copy.deepcopy(img)
  imgshow = imgshow[:,:,::-1]
  imgshow = Image.fromarray(imgshow.astype(np.uint8))
  imgshow.save(filename)
  

def makeGaussian(size, fwhm = 3, center=None):
    """ 
			Make a square gaussian kernel.
			size is the length of a side of the square
			fwhm is full-width-half-maximum, which
			can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return 255*(np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2))

def get_keypoint_colors(size):
  #colors=np.ndarray(17,3)
  cols=np.array([[0,1,0],[1,0,0],[0,0,1],[0,1,0],[1,0,0],[0,0,1],
		[0,1,1],[1,0,1],[0,1,1],[0,0.5,1],[0.5,0,1],[1,0,1],
		[0.8,0.1,1],[1,0.2,1],[1,1,1],[0.5,0.5,1],[0.5,1,1],[1,1,0.5]], np.int32)
  #print("cols size:",cols.shape)
  #exit(1)
  cols=cols[0:size,:]
  return cols


def draw_pts_onimg(img,kpts):
  #print("kpts:",kpts)
  #exit(1)
  size=kpts.shape
  imchannels=3
  #print("img.ndim:",img.ndim)
  #exit(1)
  if not (img.ndim==3):
    img=np.dstack((img,img,img))
  #exit(1)
  cols=get_keypoint_colors(size[1])
  #print("max(np.ndarray.flatten(img)):",max(np.ndarray.flatten(img)))
  if max(np.ndarray.flatten(img))>2:
    cols=cols*255
  #print(cols)  
  #print("kpt size:",kpts.shape,"imkg shape:",img.shape,"cols.shape:",cols.shape,"imchannels:",imchannels)
  #exit(1)
  for i in range(0,size[1]):
    for c in  range(0,imchannels):
      #print("c:",c)
      if imchannels==3:
	img[int(kpts[1,i]-3):int(kpts[1,i]+3),int(kpts[0,i]-3):int(kpts[0,i]+3),c]=cols[i,c]
      #else:
	#img[int(kpts[0,i]-3):int(kpts[0,i]-3),int(kpts[1,i]-3):int(kpts[1,i]-3)]=cols[i,c]
  return img



