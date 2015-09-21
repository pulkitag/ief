%Master file simply lists all the files along with the imageids
paths  = get_paths();
annDat = load(paths.annFile);
annList = annDat.RELEASE.annolist; 

fid = fopen(paths.masterFile, 'w');
for i=1:1:length(annList)
	name = annList(i).image.name(1:end-4);
	fprintf(fid, '%s \t %d\n',name, i);
end
fclose(fid);
