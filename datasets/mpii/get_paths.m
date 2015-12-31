function [paths] = get_paths()

%All the annotations
paths.annFile = 'data/mpii_human_pose_v1_u12_1.mat';
%Validation annotation
paths.valAnnFile = 'data/tompson_detections.mat';

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
