import _init_paths
import os
import os.path as osp
import PIL
import numpy as np
import scipy.sparse
import h5py
import cv2
import matplotlib.pyplot as plt
import copy
from PIL import Image
import utils.visualization as vis

class imdata(object):
    def __init__(self, imname="", error_targets=[],coords_init=[], 
								 coords_target=[], coords_predicted=[], istrain=0, 
								 isflipped=0, bbox=[]):

	#TODO: ADD COORDINATE SUYSTEMS INFO 9IMAGE vs BBO)
	self._imname           = imname 
	self._coords_init      = coords_init
	self._coords_target    = coords_target
	self._coords_predicted = coords_predicted
	self._error_targets    = error_targets
	self._istrain          = istrain
	self._isflipped        = isflipped
	self._bbox             = bbox
    
    def set_targetkptcoords_wrt_bbox(self):
      #the keypoints are with respect to the upper left corner of the box
      #kpts: 2 X num_keypoints
      kpts=self._coords_target
      kpts[0,:]=kpts[0,:]-self._bbox[0]	
      kpts[1,:]=kpts[1,:]-self._bbox[1]	
      self._coords_target=kpts
      
    def set_init_coords_from_meanpose(self,mean_pose,scalar):
      mean_pose=copy.deepcopy(mean_pose)
      #print("bbox:",bbox,"mean_pose:",mean_pose)
      #exit(1)
      width=self._bbox[2]-self._bbox[0]
      height=self._bbox[3]-self._bbox[1]
      bbox_center=np.hstack((np.mean(self._bbox[[0,2]]), np.mean(self._bbox[[1,3]])))
      #print("bbox_center:",bbox_center)
      #exit(1)

      #zero mean
      pose_center=np.hstack((np.mean(mean_pose[0,:]), np.mean(mean_pose[1,:])))
      mean_pose[0,:]=mean_pose[0,:]-pose_center[0]
      mean_pose[1,:]=mean_pose[1,:]-pose_center[1]

      #scale it
      #exit(1)
      #print("bbox_center:",bbox_center,"mean_pose:",mean_pose)
      scalex=(scalar*(width/2))/max(abs(mean_pose[0,:]))
      scaley=(scalar*(height/2))/max(abs(mean_pose[1,:]))
      #print("scalex:",scalex,bbox_center[0],max(abs(mean_pose[0,:]-bbox_center[0])),"scaley:",scaley)
      #exit(1)
      mean_pose[0,:]=scalex*mean_pose[0,:]
      mean_pose[1,:]=scaley*mean_pose[1,:]
      #print("max(abs(mean_pose[0,:])):",max(abs(mean_pose[0,:])),
	    #"max(abs(mean_pose[1,:])):",max(abs(mean_pose[1,:])),width,height)
    

      #center it at the center of the box
      mean_pose[0,:]=mean_pose[0,:]+bbox_center[0]
      mean_pose[1,:]=mean_pose[1,:]+bbox_center[1]
      #print("pose",max(mean_pose[0,:]),min(mean_pose[0,:]),max(mean_pose[1,:]),min(mean_pose[1,:]),
      #"bb",max(abs(mean_pose[1,:])),width,height,self._bbox)
      #exit(1)
      if max(mean_pose[0,:])>self._bbox[2] or min(mean_pose[0,:])<self._bbox[0] or max(mean_pose[1,:])>self._bbox[3] or min(mean_pose[1,:])<self._bbox[1]:
	print("POSE EXCEEDS THE BOX!")
      
      self._coords_init=mean_pose
      #self.plot("init")
  
      
    def warp_kpts_box2square(self,mode,square_width):
      #assumes the kpts are in IMAGE CORDINATES and makes them in BOX COORDINATES 
      #(WRT TO UPPER LEFT CORNER)
      #TODO: add assertion
      bbox_width=self._bbox[2]-self._bbox[0]
      bbox_height=self._bbox[3]-self._bbox[1]
      if mode=="init":
	kpts_coords=self._coords_init
      if mode=="target":
	kpts_coords=self._coords_target 
      if mode=="predicted":
	kpts_coords=self._coords_predicted    
      #kpts_warped=copy.deepcopy(kpts_coords)
      #if centering:
	##mean subtracted
	#pose_center=np.hstack((np.mean(kpts_warped[0,:]), np.mean(kpts_warped[1,:])))  			    
      kpts_coords[0,:]=kpts_coords[0,:]-self._bbox[0]
      kpts_coords[1,:]=kpts_coords[1,:]-self._bbox[1]
      
      scalex=square_width/bbox_width
      scaley=square_width/bbox_height
      #print("scalex:",scalex,"scaley:",scaley)
      kpts_coords[0,:]=scalex*kpts_coords[0,:]
      kpts_coords[1,:]=scaley*kpts_coords[1,:]

	  
    
    def plot_cropped_resized(self,mode,square_width):
      if mode=="init":
	kpts_coords=self._coords_init
      if mode=="target":
	kpts_coords=self._coords_target 
      if mode=="predicted":
	kpts_coords=self._coords_predicted    
      im=cv2.imread(self._imname)
      im=im[:,:,::-1]
      imcropped=vis.crop_im(im,self._bbox)
      imsquare=vis.resize_im(imcropped,[square_width, square_width]);
      plt.imshow(imsquare)
      plt.plot(kpts_coords[0,:], kpts_coords[1,:],'ro')
      #larm
      plt.plot(kpts_coords[0,[0,1]], kpts_coords[1,[0,1]],'b-',linewidth=3)
      plt.plot(kpts_coords[0,[1,2]], kpts_coords[1,[1,2]],'b-',linewidth=3)
      #rlarm
      plt.plot(kpts_coords[0,[3,4]], kpts_coords[1,[3,4]],'g-',linewidth=3)
      plt.plot(kpts_coords[0,[4,5]], kpts_coords[1,[4,5]],'g-',linewidth=3)
      #lleg
      plt.plot(kpts_coords[0,[6,7]], kpts_coords[1,[6,7]],'b-',linewidth=3)
      plt.plot(kpts_coords[0,[7,8]], kpts_coords[1,[7,8]],'b-',linewidth=3)
      #rleg
      plt.plot(kpts_coords[0,[9,10]], kpts_coords[1,[9,10]],'g-',linewidth=3)
      plt.plot(kpts_coords[0,[10,11]], kpts_coords[1,[10,11]],'g-',linewidth=3)
      
      plt.plot(kpts_coords[0,[0,3]], kpts_coords[1,[0,3]],'r-',linewidth=3)
      #plt.plot(kpts_coords[0,[0,6]], kpts_coords[1,[0,6]],'r-',linewidth=3)
      plt.plot(kpts_coords[0,[6,9]], kpts_coords[1,[6,9]],'r-',linewidth=3)
      #plt.plot(kpts_coords[0,[3,9]], kpts_coords[1,[3,9]],'r-',linewidth=3)
      plt.plot(np.array([np.mean(kpts_coords[0,[0,3]]), np.mean(kpts_coords[0,[6,9]])]),
	      np.array([np.mean(kpts_coords[1,[0,3]]), np.mean(kpts_coords[1,[6,9]])]),'r-',linewidth=3) 
      
      plt.show()
      
    def set_error_target_coords(self):
      self._error_targets=self._coords_target-self._coords_init

    def plot(self,mode):
      if mode=="init":
	kpts_coords=self._coords_init
      if mode=="target":
	kpts_coords=self._coords_target 
      if mode=="predicted":
	kpts_coords=self._coords_predicted    
      im=cv2.imread(self._imname)
      im=im[:,:,::-1]
      plt.imshow(im)
      plt.plot(kpts_coords[0,:], kpts_coords[1,:],'ro')
      #larm
      plt.plot(kpts_coords[0,[0,1]], kpts_coords[1,[0,1]],'b-',linewidth=3)
      plt.plot(kpts_coords[0,[1,2]], kpts_coords[1,[1,2]],'b-',linewidth=3)
      #rlarm
      plt.plot(kpts_coords[0,[3,4]], kpts_coords[1,[3,4]],'g-',linewidth=3)
      plt.plot(kpts_coords[0,[4,5]], kpts_coords[1,[4,5]],'g-',linewidth=3)
      #lleg
      plt.plot(kpts_coords[0,[6,7]], kpts_coords[1,[6,7]],'b-',linewidth=3)
      plt.plot(kpts_coords[0,[7,8]], kpts_coords[1,[7,8]],'b-',linewidth=3)
      #rleg
      plt.plot(kpts_coords[0,[9,10]], kpts_coords[1,[9,10]],'g-',linewidth=3)
      plt.plot(kpts_coords[0,[10,11]], kpts_coords[1,[10,11]],'g-',linewidth=3)
      
      plt.plot(kpts_coords[0,[0,3]], kpts_coords[1,[0,3]],'r-',linewidth=3)
      #plt.plot(kpts_coords[0,[0,6]], kpts_coords[1,[0,6]],'r-',linewidth=3)
      plt.plot(kpts_coords[0,[6,9]], kpts_coords[1,[6,9]],'r-',linewidth=3)
      #plt.plot(kpts_coords[0,[3,9]], kpts_coords[1,[3,9]],'r-',linewidth=3)
      plt.plot(np.array([np.mean(kpts_coords[0,[0,3]]), np.mean(kpts_coords[0,[6,9]])]),
	      np.array([np.mean(kpts_coords[1,[0,3]]), np.mean(kpts_coords[1,[6,9]])]),'r-',linewidth=3) 
      plot_bbox(self._bbox)
      bbox_center=np.hstack((np.mean(self._bbox[[0,2]]), np.mean(self._bbox[[1,3]])))
      plt.plot(bbox_center[0],bbox_center[1],'y.',markersize=10)
      plt.show()
    
  
   
   

class Imkptdb(object):
    """list of imdata with methods"""
    def __init__(self, name, my_imdata_list=None, mean_pose_file=None,mean=None,std=None):
        self._name = name
        self._imdata_list = my_imdata_list
        self._mean_pose_file=mean_pose_file
        finput = h5py.File(mean_pose_file,'r')
        self._mean_pose = np.asarray(finput['mean_pose'])
        self._mean=mean
        self._std=std
        #print(np.asarray(self._mean_pose))
        #exit(1)
        
     
    @classmethod
    def from_files(cls,name,imname_file,data_file,mean_pose_file):
       finput = h5py.File(data_file,'r')
       keypointsX = np.asarray(finput['keypointsX'])
       keypointsY = np.asarray(finput['keypointsY'])
       istrain = np.asarray(finput['istrain'])
       bbox = np.asarray(finput['bbox'])
       #print("bbox.shape:",bbox.shape,"keypointsX.shape:",keypointsX,"istrain:",istrain)
       #exit(1)
       with open(imname_file) as f:
         image_list = f.read().splitlines()
       #print image_list  
       print("num images:",len(image_list))
       imdata_list=[]
       for i in range(0,len(image_list)):
         imdatanew=imdata(image_list[i],[],[],np.vstack((keypointsX[i,:],keypointsY[i,:])),
			  [],istrain[i],0,bbox[i,:])
         imdata_list.append(imdatanew)
       return cls(name,imdata_list,mean_pose_file,[],[])

 

    @property
    def name(self):
        return self._name


    @property
    def num_images(self):
      return len(self._imdata_list)
    #def append_flipped_images(self):
        #num_images = self.num_images
        #widths = [PIL.Image.open(self.image_path_at(i)).size[0]
                  #for i in xrange(num_images)]
        #for i in xrange(num_images):
            #bbox = self._imdata[i].bbox.copy()
            #oldx1 = bbox[:, 0].copy()
            #oldx2 = bbox[:, 2].copy()
            #bbox[:, 0] = widths[i] - oldx2 - 1
            #bbox[:, 2] = widths[i] - oldx1 - 1
            #assert (bbox[:, 2] >= bbox[:, 0]).all()
            ##entry = {'boxes' : boxes,
                     ##'gt_overlaps' : self.roidb[i]['gt_overlaps'],
                     ##'gt_classes' : self.roidb[i]['gt_classes'],
                     ##'flipped' : True}
	    
	    #imdatanew=imdata(self._imdata[i].imname,keypointsflipped,
		      #self._imdata[i].istrain,bbox)

            #self._imdata.append(imdatanew)
        ##self._image_index = self._image_index * 2


    def standardize(self):
        shape=(self._imdata_list[0]._error_targets).shape
        num_keypoints=shape[1]
        data=[np.reshape(datum._error_targets,(1,2*num_keypoints)) for datum in self._imdata_list]
        data=np.vstack(data)
        mean=np.mean(data, axis=0)
        #print(mean,data.shape)
        data=data-mean
        mean=np.mean(data, axis=0)
        self._mean=mean
        stds=np.std(data, axis=0)
        #print(stds,mean,data.shape)
        data=np.divide(data,stds)
        stds=np.std(data, axis=0)
        self._std=stds
        for i in range(0, len(self._imdata_list)):
	  self._imdata_list[i]._error_targets=np.reshape(data[i,:],(2,num_keypoints))
	  
	  
	  
    def unstandardize(self):
        shape=(self._imdata_list[0]._error_targets).shape
        num_keypoints=shape[1]
        data=[np.reshape(datum._error_targets,(1,2*num_keypoints)) for datum in self._imdata_list]
        data=np.vstack(data)
        data=np.multiply(data,self._std)
        data=data+self._mean
        for i in range(0, len(self._imdata_list)):
	  self._imdata_list[i]._error_targets=np.reshape(data[i,:],(2,num_keypoints))
        
        
        
        
        #print(stds,mean,data.shape)
       
    def image_path_at(self, i):
        return self._imdata_list[i].imname
    def set_targetkptcoords_wrt_bbox_all(self):
      for i in range(0, len(self._imdata_list)):
	  self._imdata_list[i].set_targetkptcoords_wrt_bbox()

    def warp_kpts_box2square_all(self,mode,square_width):
      for i in range(0, len(self._imdata_list)):
	  self._imdata_list[i].warp_kpts_box2square(mode,square_width) 

    def set_error_target_coords_all(self):
       for i in range(0,len(self._imdata_list)):
	#print("set init coords ",i)
	(self._imdata_list[i]).set_error_target_coords()

    def set_init_coords_from_meanpose_all(self):
      for i in range(0,len(self._imdata_list)):
	#print("set init coords ",i)
	(self._imdata_list[i]).set_init_coords_from_meanpose(self._mean_pose,0.9)
	#plot_imdata(self._imdata_list[i]._imname,self._imdata_list[i]._coords_init,
		    #self._imdata_list[i]._bbox)
   
