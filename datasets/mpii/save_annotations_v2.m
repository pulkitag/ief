function [] = save_annotations_v2()
%% Store the following
%1. img_name
%2. nObj: number of objects
%3. objPosxy 
%4. Keypoints and their visibility
%5. Set Name
%6. Scale of the person
%%

paths = get_paths();
annDat   = load(paths.annFile);
annList  = annDat.RELEASE.annolist;

setNames = {'train','val','test'}; 
for s=1:1:length(setNames)
	ids = get_set_ids(setNames{s});
	for idx=1:1:length(ids)
		i = ids(idx);
		if i==24731 || i==1153 || i==24298
			continue;
		end
		imgName  = annList(i).image.name;
		outName  = sprintf(paths.svAnnFile, imgName(1:end-4));
		nObj     = length(annList(i).annorect);
		objPosxy = zeros(nObj,2);
		scale    = zeros(nObj,1);
		kpts     = zeros(nObj,16, 2);
		kptsVis  = zeros(nObj,16);
		count    = 0;
		for n=1:1:nObj
			if ~isfield(annList(i).annorect(n), 'scale')
				continue
			else
				%Scale and position of the human
				if ~isempty(annList(i).annorect(n).scale)
					count = count + 1;
					objPosxy(count,1) = annList(i).annorect(n).objpos.x;
					objPosxy(count,2) = annList(i).annorect(n).objpos.y;
					scale(count) = annList(i).annorect(n).scale;
				end		
				%Keypoints
				if isfield(annList(i).annorect(n), 'annopoints')
					%For the images which annotation points are given
					%the scale should also be given
					if isempty(annList(i).annorect(n).scale)
						assert(isempty(annList(i).annorect(n).annopoints));
						continue;
					end
					%Find the right permutation of the keypoints based 
					%on the ids
					pts  = annList(i).annorect(n).annopoints;
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
						kpts(count,k,1)  = pts.point(kIdx).x;
						kpts(count,k,2)  = pts.point(kIdx).y;
						if isempty(pts.point(kIdx).is_visible)
							kptsVis(count,k) = 0;
						else
							kptsVis(count,k) = pts.point(kIdx).is_visible;
						end
					end
				end	
			end
		end
		%Format the data
		nObj  = count;
		objPosxy = objPosxy(1:nObj,:);
		scale    = scale(1:nObj);
		kpts     = kpts(1:nObj,:,:);
		kptsVis  = kptsVis(1:nObj,:);
		setName  = setNames{s};
		imgName  = fullfile(paths.imDir, imgName);
		disp(imgName);
		save(outName, 'imgName', 'nObj', 'objPosxy', 'scale', 'kpts',...
									'kptsVis', 'setName', '-v7.3');
		if mod(idx,100)==1
			disp(idx);
		end
	end
end
end 
