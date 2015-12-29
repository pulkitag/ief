function [name] = id2name(pths, id)

	[allNames, allIds] = textread(pths.masterFile, '%s \t %d\n');
	[~,ia,ib] = intersect(id, allIds, 'stable');
	assert(length(ia)==length(id));
	name = allNames(ib);
end
