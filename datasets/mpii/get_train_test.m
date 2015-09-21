paths = get_paths();
%All files
annDat   = load(paths.annFile);
annList  = annDat.RELEASE.annolist;
%Validation Annotation
valAnn   = load(paths.valAnnFile);

tvFile = sprintf(paths.setFile, 'trainval');
teFile = sprintf(paths.setFile,'test');
vlFile = sprintf(paths.setFile,'val');
trFile = sprintf(paths.setFile, 'train');

tvFid  = fopen(tvFile, 'w');
teFid  = fopen(teFile, 'w');
vlFid  = fopen(vlFile, 'w');
N      = length(annList);
valNum = valAnn.RELEASE_img_index;
count  = 1;
for i=1:1:N
	names = jImNames{i};
	name  = [];
	if isempty(names)
		fprintf(teFid, '%s\n', annList(i).image.name(1:end-4));
	else
		for j=1:1:length(names)
			if isstr(names{j}) && isempty(name)
				name = names{j};
			end
			if ~isempty(name) && ~isempty(names{j})
				assert(strcmp(name, names{j}));
			end
		end
		%disp(sprintf('%s, %s', name, annList(i).image.name));
		%assert(strcmp(name, annList(i).image.name));
		if ~(strcmp(name, annList(i).image.name))
			keyboard;
		end
		fprintf(tvFid, '%s\n', annList(i).image.name(1:end-4));
	end
	%Write the val names.
	if count <= length(valNum) && i==valNum(count)
		repeatFlag = true;
		fprintf(vlFid, '%s\n', annList(i).image.name(1:end-4));
		while repeatFlag
			count = count + 1;
			if count > length(valNum) || valNum(count) > i
				repeatFlag = false;
			end
		end
	end
end
fclose(tvFid);
fclose(teFid);
fclose(vlFid);

%Verfiy trainval against test
tvFiles = textread(tvFile, '%s\n');
teFiles = textread(teFile, '%s\n');
vlFiles = textread(vlFile, '%s\n');

%val with trainval
[~,ia,ib] = intersect(vlFiles, tvFiles);
assert(length(ia)==length(vlFiles));
%val with tefiles
[~,ia,ib] = intersect(vlFiles, teFiles);
assert(isempty(ia));
%trainval with test
[~,ia,ib] = intersect(tvFiles, teFiles);
assert(isempty(ia));

%Write the train file.
trFid   = fopen(trFile, 'w');
trNames = setdiff(tvFiles, vlFiles);
disp(length(trNames));
for i=1:1:length(trNames)
	fprintf(trFid, '%s\n', trNames{i});
end 
fclose(trFid);
