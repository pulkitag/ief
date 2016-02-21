% --------------------------------------------------------
% IEF
% Copyright (c) 2015
% Licensed under BSD License [see LICENSE for details]
% Written by Joao Carreira, Pulkit Agrawal and Katerina Fragkiadki
% --------------------------------------------------------


pths = get_paths();
ids  = get_set_ids('train');

for i=1:1:100
	disp(ids(i));
	name  = id2name(pths, ids(i));
	fName = sprintf(pths.svAnnFile, name{1});
	dat   = load(fName);
	imName = dat.imgName;
	kpts   = dat.kpts;
	im     = imread(imName);
	imshow(im);
	hold on;
	for k=1:1:size(kpts,1)
		plot_pose_stickmodel(squeeze(kpts(k,:,:)));
	end
	keyboard;
	clf();
end
