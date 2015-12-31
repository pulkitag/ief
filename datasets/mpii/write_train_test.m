function [] = write_train_test()
paths = get_paths();
%All files
annDat   = load(paths.annFile);
annList  = annDat.RELEASE.annolist;
isTrain  = annDat.RELEASE.img_train;
%Validation Annotation
valAnn   = load(paths.valAnnFile);

tvFile = sprintf(paths.setFile, 'trainval');
teFile = sprintf(paths.setFile,'test');
vlFile = sprintf(paths.setFile,'val');
trFile = sprintf(paths.setFile, 'train');

tvFid  = fopen(tvFile, 'w');
teFid  = fopen(teFile, 'w');
vlFid  = fopen(vlFile, 'w');
trFid  = fopen(trFile, 'w');
N      = length(annList);
%The image index
valNum  = valAnn.RELEASE_img_index;
%The person index
valPNum = valAnn.RELEASE_person_index;

%sort the validation image index
[valNum, sIdx] = sort(valNum);
valPNum        = valPNum(sIdx);
disp(length(valNum));
disp(length(valPNum));

%read the master file
[id, releaseNum, imName, pNum] = textread(paths.masterFile, ...
																'%d \t %d \t %s \t %d');

count = 1;
for i=1:1:length(id)
	if isTrain(releaseNum(i))
		if (count <= length(valNum) && releaseNum(i)==valNum(count) ...
					 && pNum(i)==valPNum(count))
			%validation set
			fprintf(vlFid, '%d\n', id(i));
			fprintf(tvFid, '%d\n', id(i));
			count = count + 1;
		else
			fprintf(trFid, '%d\n', id(i));
			fprintf(tvFid, '%d\n', id(i));
		end
	else
		fprintf(teFid, '%d\n', id(i));
	end	
end	
fclose(tvFid);
fclose(teFid);
fclose(trFid);
fclose(vlFid);

%Verfiy trainval against test
tvFiles = textread(tvFile, '%s\n');
teFiles = textread(teFile, '%s\n');
vlFiles = textread(vlFile, '%s\n');
trFiles = textread(trFile, '%s\n');

%val with trainval
[~,ia,ib] = intersect(vlFiles, tvFiles);
assert(length(ia)==length(vlFiles));
%val with tefiles
[~,ia,ib] = intersect(vlFiles, teFiles);
assert(isempty(ia));
%train with tefiles
[~,ia,ib] = intersect(trFiles, teFiles);
assert(isempty(ia));
%train with valfiles
[~,ia,ib] = intersect(trFiles, vlFiles);
assert(isempty(ia));
%trainval with test
[~,ia,ib] = intersect(tvFiles, teFiles);
assert(isempty(ia));

end
