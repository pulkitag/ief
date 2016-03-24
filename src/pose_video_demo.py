#!/usr/bin/env python2
"""
/************************************************************************
Copyright (c) 2016, Stefan Helmert
                    Chemnitz University of Technology
                    Professorship of Computer Graphics and Visualization

************************************************************************/
"""
import _init_paths
import cv2
import test_demo as td
import scipy.misc as scm
import numpy as np
import csv
import time, os, sys
import argparse
try:
 import _init_py_faster_rcnn_paths
 import detectcore
except:
 print('No person detector found! - Person detection not useable. Please specify the coordinates where humans appear that should be analyzed.')
import collections
import copy
#try:
#except:
# print('py-faster-rcnn not available - no automatic human detection')
class rectangle_c:
 def __init__(self):
  self.x_center = 0
  self.y_center = 0
  self.x_range = 0
  self.y_range = 0

def humdet(frame, threshold=0.5):
 global gnet
 cls_vec, dets_vec = detectcore.detect_object(gnet, frame)
 person = dets_vec[cls_vec.index("person")]
 human_vec = []
 score_vec = []
 detection_vec = []
 det_world_vec = []
 dets = person
 thresh = threshold
 inds = np.where(dets[:, -1] >= thresh)[0]
 for i in inds:
  bbox = dets[i, :4]
  score = dets[i, -1]
  score_vec.append(score)
  detection = rectangle_c()
  detection.x_center = (bbox[0] + bbox[2])/2
  detection.y_center = (bbox[1] + bbox[3])/2
  detection.x_range = (bbox[2] - bbox[0])
  detection.y_range = (bbox[3] - bbox[1])
  detection_vec.append(detection)
 return detection_vec



def sameorder(objs, objs_old):
 objs_new = copy.deepcopy(objs_old)
 while len(objs_new) < len(objs):
  objs_new.append(rectangle_c())
 objs_set = np.zeros(len(objs_new))
 while np.sum(objs_set) < len(objs) and np.sum(objs_set) < len(objs_set):
  for i, obj in enumerate(objs):
   dist_min = 10000000
   idx = i
   for j, obj_old in enumerate(objs_old):
    if 0 == objs_set[j]:
     dist = np.sqrt(np.power(obj_old.x_center - obj.x_center, 2) + np.power(obj_old.y_center - obj.y_center, 2))
     if dist < dist_min:
      dist_min = dist
      idx = j
   objs_new[idx] = copy.deepcopy(obj)
   objs_set[idx] = 1
 return objs_new

def posevideo(input_video_name, output_video_name=None, output_csv_name=None, isGPU=True, deviceId=0, bodyPt=[600, 400], iterations=4, fixedScale=False, scaleIdx=0, fbfactor=0.0, thresh=0.9, detinterv=10, bodyPts= [600, 400]):
 """ processing the video """
 # Find OpenCV version
 global gnet
 (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

 ief    = td.PoseIEF(isGPU=isGPU, deviceId=deviceId)
 cap = cv2.VideoCapture(input_video_name)
 if(output_video_name is not None and '' != output_video_name):
  outv = cv2.VideoWriter() 
 if(output_csv_name is not None and '' != output_csv_name):
  pose_csv_file = open(output_csv_name, 'w')
  pose_csv = csv.writer(pose_csv_file)
  pose_csv.writerows([['no_frm', 'no_prs', 'x_rft', 'y_rft', 'x_rkn', 'y_rkn', 'x_rhp', 'y_rhp',  'x_lhp', 'y_lhp', 'x_lkn', 'y_lkn', 'x_lft', 'y_lft', 'x_plv', 'y_plv', 'x_trx', 'y_trx', 'x_un', 'y_un', 'x_hd', 'y_hd', 'x_rhn', 'y_rhn', 'x_rlb', 'y_rlb', 'x_rsh', 'y_rsh', 'x_lsh', 'y_lsh', 'x_llb', 'y_llb', 'x_lhn', 'y_lhn', 'x_hum', 'y_hum']])
 cnt = 0

 humans_old = []
 currPoses = []
 read_bodyPts = True
 while(True):
  ret, frame = cap.read()
  if ret is False:
   return
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  if(output_video_name is not None and '' != output_video_name):
   if(False == outv.isOpened()):
    if int(major_ver) < 3:
     fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
     outv.open(output_video_name, cv2.cv.CV_FOURCC('A', 'P', '4', '1'), fps, (np.size(frame, 1), np.size(frame, 0)), True) #, frame.shape, True)
    else:
     fps = cap.get(cv2.CAP_PROP_FPS)
     outv.open(output_video_name, cv2.VideoWriter_fourcc('A', 'P', '4', '1'), fps, (np.size(frame, 1), np.size(frame, 0)), True) #, frame.shape, True)
  poses = []
  if 0 < detinterv:
   if 0 == cnt % detinterv:
    humans = humdet(frame, thresh)
  else:
   humans = []
   if read_bodyPts:
    human = rectangle_c()
    for i, val in enumerate(bodyPts):
     if 0 == i % 2:
      human.x_center = val
     else:
      human.y_center = val
      humans.append(copy.deepcopy(human))
    read_bodyPts = False

  humans = sameorder(humans, humans_old)
  humans_old = humans
  for i, human in enumerate(humans):
   bodyPt = [human.x_center, human.y_center]
   try:
    currPose = currPoses[i]
    pose, currPose =  ief.predict(frame, bodyPt, False, iterations, fixedScale, scaleIdx, True, currPose, fbfactor)
   except:
    pose, currPose =  ief.predict(frame, bodyPt, False, iterations, fixedScale, scaleIdx)
   humans_old[i].x_center = pose[7][0]
   humans_old[i].y_center = pose[7][1]
   try:
    currPoses[i] = currPose
   except:
    currPoses.append(currPose)
   cnt += 1
   print('Frame number: '+str(cnt))
   poses.append(pose)
   if(output_csv_name is not None and '' != output_csv_name):
    pose_arr = np.append([cnt ,i], pose)
    pose_csv.writerows([pose_arr])
  frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
  if(output_video_name is not None and '' != output_video_name):
   for i, pose in enumerate(poses):
    frame = td.vis.plot_pose_stickmodel_cv2mat(frame, pose.squeeze().transpose((1,0)), 2, False)
    cv2.putText(frame, str(i), (int(humans[i].x_center), int(humans[i].y_center)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0)) 
   cv2.imshow('stickmodel', frame)
   cv2.waitKey(1)
   outv.write(frame)
 if(output_video_name is not None and '' != output_video_name):
  outv.close()
 if(output_csv_name is not None and '' != output_csv_name): 
  pose_csv_file.close()

def parse_args():
 """ Parse input arguments """
 parser = argparse.ArgumentParser(description='Estimate the human pose in a rgb video via \'Human Pose Estimation with Iterative Error Feedback (IEF)\'')
 parser.add_argument('--isGPU', dest='isGPU', help='Boolean value that specifies if a GPU should be used for detection - isGPU=False means the network runs on CPU', default=True, type=bool)
 parser.add_argument('--deviceId', dest='deviceId', help='Natural value that specifies the number of the GPU which should be used. It starts with 0.', default='0', type=int)
 parser.add_argument('--input_video', dest='input_video_name', help='The name of the video which should be analyzed.', default='video/demo.avi', type=str)
 parser.add_argument('--output_video', dest='output_video_name', help='The name of the video to be newly created containing the stick model.', default='?', type=str)
 parser.add_argument('--output_csv', dest='output_csv_name', help='The name of the csv file to be newly created containing the joint postions.', default='?', type=str)
 parser.add_argument('--x_bodyPt', dest='x_bodyPt', help='Natural value that represents the x-coordinate of the pointer telling which human should be analyzed.', default=600, type=int)
 parser.add_argument('--y_bodyPt', dest='y_bodyPt', help='Natural value that represents the y-coordinate of the pointer telling which human should be analyzed.', default=400, type=int)
 parser.add_argument('--iterations', dest='iterations', help='Natural value that specifies how much IEF iterations per image should be done.', default=4, type=int)
 parser.add_argument('--scaleIdx', dest='scaleIdx', help='Natural value that specifies the IEF scaleIdx if fixed scale flag is set.', default=0, type=int)
 parser.add_argument('--fbfactor', dest='fbfactor', help='Fractional value between 0.0 and 1.0 that specifies the weight of the last pose of the last image for the current image.', default=0.0, type=float)
 parser.add_argument('--fixedScale', dest='fixedScale', help='Boolean value that deactivates the autoscale network netScale. The scale index scaleIdx has to be specified manualy (default=0)', default=False, type=bool)
 parser.add_argument('--bodyPts', dest='bodyPts', help='Natural value coordinates of human appearance/starting points for pose detection. Order: x_person_1 y_person_1 x_person_2 ...', default=[600, 400], type=int, nargs='+')
 # arguments for human detection
 parser.add_argument('--thresh', dest='thresh', help='Fractional value between 0.0 and 1.0 that specifies the selectivity of human detection.', default=0.9, type=float)
 parser.add_argument('--detinterv', dest='detinterv', help='Natural value that specifies every how much images human detection is done.', default=10, type=int)
 parser.add_argument('--net', dest='net', help='The name of network used for human detection.', default='vgg16', type=str)
 

 if len(sys.argv) == 1:
  parser.print_help()
  sys.exit(1)

 args = parser.parse_args()
 default_output_name = args.input_video_name.rsplit('.', 1)[0]
 if '?' == args.output_video_name:
  args.output_video_name = default_output_name+'_PoseIEF.avi'
 if '?' == args.output_csv_name:
  args.output_csv_name = default_output_name+'_PoseIEF.csv'

 return args

if __name__ == '__main__':
 global gnet
 args = parse_args()
 print('Called with args:')
 print(args)
 frargs = collections.namedtuple('args', 'demo_net cpu_mode gpu_id')
 frargs.cpu_mode = not args.isGPU
 frargs.demo_net = args.net
 frargs.gpu_id = args.deviceId
 gnet = detectcore.init(frargs)
 posevideo(args.input_video_name, args.output_video_name, args.output_csv_name, isGPU=args.isGPU, deviceId=args.deviceId, bodyPt=[args.x_bodyPt, args.y_bodyPt], iterations=args.iterations, fixedScale=args.fixedScale, scaleIdx=args.scaleIdx, fbfactor=args.fbfactor, thresh=args.thresh, detinterv=args.detinterv, bodyPts=args.bodyPts)

