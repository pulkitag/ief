% --------------------------------------------------------
% IEF
% Copyright (c) 2015
% Licensed under BSD License [see LICENSE for details]
% Written by Joao Carreira, Pulkit Agrawal and Katerina Fragkiadki
% --------------------------------------------------------



function [name] = id2name(pths, id)

	[allNames, allIds] = textread(pths.masterFile, '%s \t %d\n');
	[~,ia,ib] = intersect(id, allIds, 'stable');
	assert(length(ia)==length(id));
	name = allNames(ib);
end
