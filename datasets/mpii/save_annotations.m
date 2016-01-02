function []  = save_annotations()

%% Store the following
%1. img_name
%2. objPosxy 
%3. Keypoints and their visibility
%4. Scale of the person
%%

paths = get_paths();
annDat   = load(paths.annFile);
annList  = annDat.RELEASE.annolist;

%read the master file
[id, releaseNum, imName, pNum] = textread(paths.masterFile, ...
															'%d \t %d \t %s \t %d');

for idx=1:1:length(id)
	i  = id(idx);
	rn = releaseNum(idx); 
	%if rn==24731 || rn==1153 || rn==24298
	%	continue;
	%end
	imgName  = annList(rn).image.name;
	outName  = sprintf(paths.svAnnFile, sprintf('%06d', i));
	objPosxy = zeros(1,2);
	scale    = zeros(1,1);
	kpts     = zeros(1, 16, 2);
	kptsVis  = zeros(1, 16);
	%Person num
	n = pNum(idx);	
	if ~isfield(annList(rn).annorect(n), 'scale')
		continue
	else
		%Scale and position of the human
		if ~isempty(annList(rn).annorect(n).scale)
			objPosxy(1,1) = annList(rn).annorect(n).objpos.x;
			objPosxy(1,2) = annList(rn).annorect(n).objpos.y;
			scale(1,1) = annList(rn).annorect(n).scale;
		end		
		%Keypoints
		if isfield(annList(rn).annorect(n), 'annopoints')
			%For the images which annotation points are given
			%the scale should also be given
			if isempty(annList(rn).annorect(n).scale)
				assert(isempty(annList(rn).annorect(n).annopoints));
				continue;
			end
			%Find the right permutation of the keypoints based 
			%on the ids
			pts  = annList(rn).annorect(n).annopoints;
			perm = zeros(length(pts.point),1);
			for k=1:1:length(pts.point)
				assert(pts.point(k).id + 1 <= 16);
				perm(pts.point(k).id + 1)     = k;
			end
			for k=1:1:length(pts.point)
				kIdx = perm(k);
				if kIdx==0
					continue;
				end
				kpts(1,k,1)  = pts.point(kIdx).x;
				kpts(1,k,2)  = pts.point(kIdx).y;
				if isempty(pts.point(kIdx).is_visible)
					kptsVis(1,k) = 0;
				else
					kptsVis(1,k) = pts.point(kIdx).is_visible;
				end
			end
		end	
	end
	%Format the data
	imgName  = fullfile(paths.imDir, imgName);
	%disp(imgName);
	save(outName, 'imgName', 'objPosxy', 'scale', 'kpts',...
								'kptsVis', '-v7.3');
	if mod(idx,100)==1
		disp(idx);
	end
end

end 
