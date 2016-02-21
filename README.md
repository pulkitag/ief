  Human Pose Estimation with Iterative Error Feedback (IEF)
  ====================================================

<<<<<<< HEAD
  This repository provides a Caffe implementation of the algorithm described in our paper ([pdf](http://arxiv.org/abs/1507.06550)) for estimating human pose in natural images.
=======
This package is an implementation of the algorithm described in our paper ([pdf](http://arxiv.org/abs/1507.06550)) for estimating human pose in natural images.

Requirements
------------

- Python 2.7
- OpenCV python interface (python-opencv)
- Caffe

Currently, it is necessary to use caffe version at the [link](https://github.com/pulkitag/caffe.git). A patch that you can apply to your own caffe repository will be provided shortly in the future.


Installation
-------------
This package has been tested on Ubuntu 14.04.

- Download IEF models and supporting code: `./setup.sh`

Running Demo
------------
In the main code directory launch ipython and run,
<pre><code>
from src import test_demo as td
#Define pose-predictor class
ief    = td.PoseIEF()
#Name of the image
imName = 'src/test_images/elvis.jpg'
#Point (x,y) on the torso of the person whose pose is to be estimated
bodyPt = (108, 98)
#Predict the pose
pose,_ =  ief.predict(imName, bodyPt)
#Visualize the result
import scipy.misc as scm
im = scm.imread(imName)
td.vis.plot_pose_stickmodel(im, pose.squeeze().transpose((1,0)))
</code></pre>

Note: This code only runs 1 image in a single batch and is hence runs slower than what can be achieved with larger batch sizes.

The README will be shortly updated with more details.
>>>>>>> 77e336189ef32217e62145354ff3d2ca1316d519

  To estimate pose of new images. 

  The README will be shortly updated.



  [comment]: # (Automatically Generated Documentation)

  [comment]: # (-------------)

  [comment]: # (doxygen: sudo apt-get install doxygen)

  [comment]: # (doxypy:  sudo pip install doxypy)
