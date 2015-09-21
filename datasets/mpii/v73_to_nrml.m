%Convert v7.3 files back to normal.
paths = get_paths();
ssDir = paths.ssDir;
fNames = dir(fullfile(ssDir, '*.mat'));

for i=1:1:length(fNames)
	outName = fullfile(ssDir, fNames(i).name);
	data    = load(fullfile(ssDir,fNames(i).name));
	boxes   = data.boxes;
	save(outName, 'boxes');
	if mod(i,10)==1
		disp(i);
	end
end
