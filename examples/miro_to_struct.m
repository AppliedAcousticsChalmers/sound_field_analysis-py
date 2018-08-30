% Converts miro data into a matlab structure, which can then be loaded from
% Python
%
% This file requires the miro class definition, which can be downloaded 
% from http://audiogroup.web.th-koeln.de/wdr_irc.html .

clear;

file_name = 'HRIR_L2702';
path_name = 'data/';

miro_data = load( [ path_name file_name ] );
miro_data = miro_data.(file_name);

name       = miro_data.name;
context    = miro_data.context;
location   = miro_data.location;
date       = miro_data.date;
irChOne    = miro_data.irChOne;
irChTwo    = miro_data.irChTwo;
fs         = miro_data.fs;
azimuth    = miro_data.azimuth;
colatitude = miro_data.elevation;
radius     = miro_data.radius;
quadWeight = miro_data.quadWeight;
scatterer  = miro_data.scatterer;
avgAirTemp = miro_data.avgAirTemp; 

clear miro_data path_name file_name;

save( 'data/HRIR_L2702_struct.mat', '-v7' );

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;

file_name = 'CR1_VSA_110RS_L';
path_name = 'data/';

miro_data = load( [ path_name file_name ] );
miro_data = miro_data.(file_name);

name       = miro_data.name;
context    = miro_data.context;
location   = miro_data.location;
date       = miro_data.date;
irChOne    = miro_data.irChOne;
irChTwo    = miro_data.irChTwo;
fs         = miro_data.fs;
azimuth    = miro_data.azimuth;
colatitude = miro_data.elevation;
radius     = miro_data.radius;
quadWeight = miro_data.quadWeight;
scatterer  = miro_data.scatterer;
avgAirTemp = miro_data.avgAirTemp; 

clear miro_data path_name file_name;

save( 'data/CR1_VSA_110RS_L_struct.mat', '-v7' );