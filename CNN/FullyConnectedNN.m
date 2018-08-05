%%  Gerry Chen
%   FullyConnectedNN  -  object for a fully connected neural net

classdef FullyConnectedNN < NNLayer
    
    properties
        numIn
        numOut
        weights
        bias
        SaveStates
        saveInd
    end
    
    methods (Static)
    end
    methods
        function obj = FullyConnectedNN(numIn, numOut)
            obj.numIn = numIn;
            obj.numOut = numOut;
            obj.weights = (2*rand(numOut,numIn) - 1) / 10;
            obj.bias = (2*rand(numOut,1) - 1) / 10;
        end
        function outs = evaluate(obj, ins)
            assert(numel(ins) == obj.numIn, 'incorrect number of inputs');
            outs = CNN.sigmoid(obj.weights * ins + obj.bias);
        end
        function [nextRes] = backPropogate(obj, ins, outs, res, strength)
            res = res';
            ins = ins';
            outs = outs';
            nextRes = (obj.weights' * res)';
            assert(size(ins,2) == size(res,2) && ...
                   size(ins,1) == obj.numIn && ...
                   size(res,1) == obj.numOut, 'vectors incorrect size');
            numTrials = size(res,2);
            dW = zeros(size(obj.weights));
            pastOutput = outs;
            for outInd = 1:obj.numOut
                partial = CNN.dsigmoid(pastOutput(outInd,:))' .* ins';
                dW(outInd,:) = (res(outInd,:) * (1./partial)) / numTrials;
            end
            assert(all(size(dW) == size(obj.weights)), 'column/row vector issue');
            obj.weights = obj.weights + dW*strength;
            obj.bias = obj.bias + mean(CNN.dsigmoid(pastOutput).*res,2)*strength;
        end
    end
end