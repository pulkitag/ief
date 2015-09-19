from PIL import Image
import numpy as np
import copy

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
 
