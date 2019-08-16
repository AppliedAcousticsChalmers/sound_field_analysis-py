% Converts MIRO data into a regular Matlab struct, which can then be loaded
% from Python.
%
% This file requires the MIRO Matlab Class Definition, which can be
% downloaded from http://audiogroup.web.th-koeln.de/FILES/miro.m.
clear; clc;

% Specify MIRO Class Data files that should be saved into Matlab structs
miro2struct('data/HRIR_L2702.mat');
miro2struct('data/CR1_VSA_110RS_L.mat');


%% FUNCTION miro2struct
function miro2struct(file)
    % Load specified file
    fprintf('reading file "%s" ...\n',file);
    miro_data = load(file);

    % Extract struct from loaded data
    fields = fieldnames(miro_data);
    miro_data = miro_data.(fields{1});
    clear fields;

    % Pull all existing fields from stuct into this work space
    read_struct_fields(miro_data);
    
    % Pull impulse response again seperately using the builtin function 
    % `getIR()` which applies a time window
    [irCenter,irChOne,irChTwo] = read_ir_fields(miro_data); %#ok<ASGLU>
    clear miro_data;

    % Rename field 'elevation' since the Python API uses 'colatitude'. This
    % was done to prevent mistakes, due to the field actually containing
    % colatitude data, i.e. was falsely named before.
    colatitude = elevation(:,:); %#ok<IDISVAR,NODEF>
    clear elevation;

    % Update file name
    file = insert_before_ending(file,'_struct');

    % Save all worksapce variables instead of 'file' by means or regex
    fprintf('saving file "%s" ...\n\n',file);
    save(file,'-regexp','^(?!(file)$).','-v7');
end

function read_struct_fields(struct)
    fields = fieldnames(struct);
    for f = 1 : length(fields)
        assignin('caller',fields{f},getfield(struct,fields{f})); %#ok<GFLD>
    end
end

function [irCenter,irChOne,irChTwo] = read_ir_fields(struct)
    irCenter = struct.getIR(0); %#ok<*NASGU>
    for ch = 1:struct.nIr
        ir = struct.getIR(ch);
        irChOne(:,ch) = ir(:,1); %#ok<*AGROW>
        if size(ir,2) > 1
            irChTwo(:,ch) = ir(:,2);
        end
    end
    if size(ir,2) <= 1
        irChTwo = struct.irChTwo;  % pull empty vector again
    end
end

function str = insert_before_ending(str,insert_str)
    parts = strsplit(str,'.');
    str = [parts{1:end-1},insert_str,'.',parts{end}];
end
