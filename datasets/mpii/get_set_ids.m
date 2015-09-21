function [ids] = get_set_ids(setName)

paths = get_paths();
[names, ids] = textread(paths.masterFile, '%s \t %d');

setFile = sprintf(paths.setFile, setName);
sNames  = textread(setFile, '%s');

[~,ia,ib] = intersect(names, sNames, 'stable');
ids = ids(ia);
end
