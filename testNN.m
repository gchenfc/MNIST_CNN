function [successRate, incorrectInds, images, labels] = ...
    testNN(net, images, labels, num2test)
%testNN tests a NN against the MNIST dataset
%   testNN(net) automatically loads from the testing data file
%   testNN(net, images, labels, num2test) uses the user-defined testing
%   cases
    if (nargin<2)
        load MNIST/testingData
        num2test = length(labels);
    end
    testSet = randperm(length(labels));
    testSet = testSet(1:num2test);
    incorrectInds = [];
    for i = testSet
        out = net.evaluate(images(:,:,i),false);
        if ((out(labels(i)+1) ~= max(out)) || ...
                (sum(out==max(out))>1))
            incorrectInds = [incorrectInds,i];
        end
    end
    successRate = 1 - length(incorrectInds)/num2test;
end