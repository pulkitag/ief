% --------------------------------------------------------
% IEF
% Copyright (c) 2015
% Licensed under BSD License [see LICENSE for details]
% Written by Joao Carreira, Pulkit Agrawal and Katerina Fragkiadki
% --------------------------------------------------------


%Master file simply lists all the files along with the imageids
paths  = get_paths();
annDat = load(paths.annFile);
annList = annDat.RELEASE.annolist; 

fid   = fopen(paths.masterFile, 'w');
count = 0;
for i=1:1:length(annList)
	name      = annList(i).image.name(1:end-4);
	%An image may have multiple people
	numPerson = length(annList(i).annorect);
	for p=1:1:numPerson
		count = count + 1;
		fprintf(fid, '%d \t %d \t %s \t %d\n',count, i, name, p);
	end
end
fclose(fid);
