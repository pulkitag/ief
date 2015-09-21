function [paths] = get_paths()

%All the annotations
paths.annFile = '/home/carreira/mpii/mpii_human_pose_v1_u12_1/mpii_human_pose_v1_u12_1.mat';
%Validation annotation
paths.valAnnFile = '/home/carreira/mpii/tompson/mpii_predictions/data/detections.mat';

baseDir = '/work5/pulkitag/mpii/';
%Raw images
paths.imDir      = fullfile(baseDir, 'images');
paths.imDirSz    = fullfile(baseDir, 'sz%d',  'images'); %Resize the longest side.
paths.imDirSqSz  = fullfile(baseDir, 'sqSz%d','images');
%setlist
paths.setFile = fullfile(baseDir,'ImageSets','Main', '%s.txt');
paths.masterFile = sprintf(paths.setFile, 'master'); 
paths.badImages  = sprintf(paths.setFile, 'BAD_IMAGES');

%Save annotation file name
paths.svAnnFile  = fullfile(baseDir, 'Annotations', '%s.mat');

end
