%%  Gerry Chen
%   readMNIST.m - reads the MNIST data set and saves in a MATLAB-friendly
%   file

images = readIDX('MNIST/train-images.idx3-ubyte');
labels = readIDX('MNIST/train-labels.idx1-ubyte');
save('MNIST/trainingData','images','labels')
images = readIDX('MNIST/t10k-images.idx3-ubyte');
labels = readIDX('MNIST/t10k-labels.idx1-ubyte');
save('MNIST/testingData','images','labels')

%   readIDX - reads idx file and returns data in an array
%       readIDX(infile) reads an IDX file with filename 'infile' and
%       returns the data as an array
function [data] = readIDX(infile)
    % get raw byte data from file
    try
        file = fopen(infile);
        fileData = fread(file);
        fclose(file);
    catch ME
        fprintf('Error: file read error\n');
        rethrow(ME)
    end
    
    % format of first 4 bytes: 00 00 TT DD
    %   TT is data type
    %   DD is # of dimensions
    assert(~any(fileData(1:2)), 'not a valid IDX file')
    assert(fileData(3)==8, 'currently only support files with unsigned ints')
    numDims = uint8(fileData(4));
    
    % next couple bytes are the dimensions (32 bit ints big endian)
    currbit = 5;
    dims = zeros([1,numDims]);
    for i = 1:numDims
        dims(i) = swapbytes(typecast(uint8(fileData(currbit:currbit+3)),'uint32'));
        currbit = currbit + 4;
    end
    if (numDims==1)
        data = zeros(dims,1);
    else
        data = zeros(flip(dims));
    end
    
    % start reading data
    try
        for i = 1:numel(data)
            data(i) = uint8(fileData(currbit));
            currbit = currbit + 1;
        end
    catch ME
        fprintf('data reading - not enough bits in file\n');
        rethrow ME
    end
    assert(currbit == numel(fileData)+1, 'extra bits found');
    
    data = permute(data, [2,1,3]);
end