% finalize qc_1D dataset, only keep variables that are used
fold = '\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20231101_ripples_frame\4TU\data\SRPS\qc_1D\';
foldout = '\\tudelft.net\staff-umbrella\EURECCA\DataCiaran\data\SRPS\qc_1D\';

filez = dir([fold, '*.mat']);

for ifile = 1:length(filez)
    load([fold, filez(ifile).name])

    tokeep = {'xBed','zBed'};
    f=fieldnames(data05);
    toRemove = f(~ismember(f,tokeep));
    data05 = rmfield(data05, toRemove);
    
    save([foldout, filez(ifile).name],'data05');
end

% finalize qc_2D dataset, only keep variables that are used
fold = '\\tudelft.net\staff-umbrella\EURECCA\fieldvisits\20231101_ripples_frame\4TU\data\SRPS\qc_2D\';
foldout = '\\tudelft.net\staff-umbrella\EURECCA\DataCiaran\data\SRPS\qc_2D\';

filez = dir([fold, '*.mat']);

for ifile = 1:length(filez)
    load([fold, filez(ifile).name])

    tokeep = {'x', 'y', 'z'};
    f=fieldnames(data05);
    toRemove = f(~ismember(f,tokeep));
    data05 = rmfield(data05, toRemove);
    
    save([foldout, filez(ifile).name],'data05');
end