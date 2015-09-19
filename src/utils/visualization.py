import numpy as np
import numpy.random as npr
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import scipy.misc 
import copy

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

def plot_bbox(bbox):
    #print("bbox:",bbox,"shape:",bbox.shape,bbox[[0,2,2,0,0]])
    v=np.vstack((bbox[[0,2,2,0,0]], bbox[[1,1,3,3,1]]))
    #print("v:",v,"v.shape:",v.shape)
    plt.plot(v[0,:], v[1,:],'r-')
   
#def plot_im(im,kpts_coords):
  #plt.imshow(im)
  #plt.plot(kpts_coords[0,:], kpts_coords[1,:],'ro')
  ##larm
  #plt.plot(kpts_coords[0,[0,1]], kpts_coords[1,[0,1]],'b-',linewidth=3)
  #plt.plot(kpts_coords[0,[1,2]], kpts_coords[1,[1,2]],'b-',linewidth=3)
  ##rlarm
  #plt.plot(kpts_coords[0,[3,4]], kpts_coords[1,[3,4]],'g-',linewidth=3)
  #plt.plot(kpts_coords[0,[4,5]], kpts_coords[1,[4,5]],'g-',linewidth=3)
  ##lleg
  #plt.plot(kpts_coords[0,[6,7]], kpts_coords[1,[6,7]],'b-',linewidth=3)
  #plt.plot(kpts_coords[0,[7,8]], kpts_coords[1,[7,8]],'b-',linewidth=3)
  ##rleg
  #plt.plot(kpts_coords[0,[9,10]], kpts_coords[1,[9,10]],'g-',linewidth=3)
  #plt.plot(kpts_coords[0,[10,11]], kpts_coords[1,[10,11]],'g-',linewidth=3)
  
  #plt.plot(kpts_coords[0,[0,3]], kpts_coords[1,[0,3]],'r-',linewidth=3)
  ##plt.plot(kpts_coords[0,[0,6]], kpts_coords[1,[0,6]],'r-',linewidth=3)
  #plt.plot(kpts_coords[0,[6,9]], kpts_coords[1,[6,9]],'r-',linewidth=3)
  ##plt.plot(kpts_coords[0,[3,9]], kpts_coords[1,[3,9]],'r-',linewidth=3)
  #plt.plot(np.array([np.mean(kpts_coords[0,[0,3]]), np.mean(kpts_coords[0,[6,9]])]),
	   #np.array([np.mean(kpts_coords[1,[0,3]]), np.mean(kpts_coords[1,[6,9]])]),'r-',linewidth=3) 
  #plt.show()
  
  
#def plot_imdata(imname,kpts_coords,bbox):
  ##kpts_coords: 2 X num_keypoints w.r.t. the bbox upper left cporner
  ##make keypoints coords w.r.t the image
  #kpts_coords[0,:]=kpts_coords[0,:]+bbox[0]
  #kpts_coords[1,:]=kpts_coords[1,:]+bbox[1]
  #im = cv2.imread(imname)
  #im=im[:,:,::-1]
  #plt.imshow(im)
  #plt.plot(kpts_coords[0,:], kpts_coords[1,:],'ro')
  ##larm
  #plt.plot(kpts_coords[0,[0,1]], kpts_coords[1,[0,1]],'b-',linewidth=3)
  #plt.plot(kpts_coords[0,[1,2]], kpts_coords[1,[1,2]],'b-',linewidth=3)
  ##rlarm
  #plt.plot(kpts_coords[0,[3,4]], kpts_coords[1,[3,4]],'g-',linewidth=3)
  #plt.plot(kpts_coords[0,[4,5]], kpts_coords[1,[4,5]],'g-',linewidth=3)
  ##lleg
  #plt.plot(kpts_coords[0,[6,7]], kpts_coords[1,[6,7]],'b-',linewidth=3)
  #plt.plot(kpts_coords[0,[7,8]], kpts_coords[1,[7,8]],'b-',linewidth=3)
  ##rleg
  #plt.plot(kpts_coords[0,[9,10]], kpts_coords[1,[9,10]],'g-',linewidth=3)
  #plt.plot(kpts_coords[0,[10,11]], kpts_coords[1,[10,11]],'g-',linewidth=3)
  
  #plt.plot(kpts_coords[0,[0,3]], kpts_coords[1,[0,3]],'r-',linewidth=3)
  ##plt.plot(kpts_coords[0,[0,6]], kpts_coords[1,[0,6]],'r-',linewidth=3)
  #plt.plot(kpts_coords[0,[6,9]], kpts_coords[1,[6,9]],'r-',linewidth=3)
  ##plt.plot(kpts_coords[0,[3,9]], kpts_coords[1,[3,9]],'r-',linewidth=3)
  #plt.plot(np.array([np.mean(kpts_coords[0,[0,3]]), np.mean(kpts_coords[0,[6,9]])]),
	   #np.array([np.mean(kpts_coords[1,[0,3]]), np.mean(kpts_coords[1,[6,9]])]),'r-',linewidth=3)
  ##bbox
  #plot_bbox(bbox)
    


