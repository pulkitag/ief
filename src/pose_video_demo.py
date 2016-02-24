#!/usr/bin/env python2
"""
/************************************************************************
Copyright (c) 2016, Stefan Helmert
                    Chemnitz University of Technology
                    Professorship of Computer Graphics and Visualization

************************************************************************/
"""
import cv2
import test_demo as td
import scipy.misc as scm
import numpy as np
import csv
import time, os, sys
import argparse

def posevideo(input_video_name, output_video_name=None, output_csv_name=None, isGPU=True, deviceId=0, bodyPt=[600, 400]):
 """ processing the video """
 # Find OpenCV version
 (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

 ief    = td.PoseIEF(isGPU=isGPU, deviceId=deviceId)
 cap = cv2.VideoCapture(input_video_name)
 if(output_video_name is not None and '' != output_video_name):
  outv = cv2.VideoWriter() 
 if(output_csv_name is not None and '' != output_csv_name):
  pose_csv_file = open(output_csv_name, 'w')
  pose_csv = csv.writer(pose_csv_file)
  pose_csv.writerows([['x_rft', 'y_rft', 'x_rkn', 'y_rkn', 'x_rhp', 'y_rhp',  'x_lhp', 'y_lhp', 'x_lkn', 'y_lkn', 'x_lft', 'y_lft', 'x_plv', 'y_plv', 'x_trx', 'y_trx', 'x_un', 'y_un', 'x_hd', 'y_hd', 'x_rhn', 'y_rhn', 'x_rlb', 'y_rlb', 'x_rsh', 'y_rsh', 'x_lsh', 'y_lsh', 'x_llb', 'y_llb', 'x_lhn', 'y_lhn', 'x_hum', 'y_hum']])
 cnt = 0
 while(True):
  ret, frame = cap.read()
  if ret is False:
   return
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  if(output_video_name is not None and '' != output_video_name):
   if(False == outv.isOpened()):
    if(major_ver<3):
     fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
     outv.open(output_video_name, cv2.cv.CV_FOURCC('A', 'P', '4', '1'), fps, (np.size(frame, 1), np.size(frame, 0)), True) #, frame.shape, True)
    else:
     fps = cap.get(cv2.CAP_PROP_FPS)
     outv.open(output_video_name, cv2.VideoWriter_fourcc('A', 'P', '4', '1'), fps, (np.size(frame, 1), np.size(frame, 0)), True) #, frame.shape, True)
  pose,_ =  ief.predict(frame, bodyPt)
  cnt += 1
  print('Frame number: '+str(cnt))
  if(output_csv_name is not None and '' != output_csv_name):
   pose_arr = np.append(pose,[])
   pose_csv.writerows([pose_arr])
  if(output_video_name is not None and '' != output_video_name):
   frame = td.vis.plot_pose_stickmodel_cv2mat(frame, pose.squeeze().transpose((1,0)))
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
 default_output_name = (parser.parse_args().input_video_name).rsplit('.', 1)[0]
 parser.add_argument('--output_video', dest='output_video_name', help='The name of the video to be newly created containing the stick model.', default=default_output_name+'_PoseIEF.avi', type=str)
 parser.add_argument('--output_csv', dest='output_csv_name', help='The name of the csv file to be newly created containing the joint postions.', default=default_output_name+'_PoseIEF.csv', type=str)
 parser.add_argument('--x_bodyPt', dest='x_bodyPt', help='Natural value that represents the x-coordinate of the pointer telling which human should be analyzed.', default=600, type=int)
 parser.add_argument('--y_bodyPt', dest='y_bodyPt', help='Natural value that represents the y-coordinate of the pointer telling which human should be analyzed.', default=400, type=int)
 if len(sys.argv) == 1:
  parser.print_help()
  sys.exit(1)

 args = parser.parse_args()
 return args

if __name__ == '__main__':
 args = parse_args()
 print('Called with args:')
 print(args)
 posevideo(args.input_video_name, args.output_video_name, args.output_csv_name, isGPU=args.isGPU, deviceId=args.deviceId, bodyPt=[args.x_bodyPt, args.y_bodyPt])

