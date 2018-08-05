%%  Gerry Chen
%   CNN  -  object class holding CNN parameters/objects

% NN format:
%   input image
%   -> 6x conv (5x5)
%   -> 0.5x downsample
%   -> 16x conv (5x5)
%   -> 0.5x downsample
%   -> 150x fully connected
%   -> 10x fully connected (output layer)

classdef CNN < handle
    
    properties
        numIn
        numOut
        numLayers
        Layers
        SaveStates
        saveInd
    end
    
    properties (Constant)
        A = 1.7159;
        S = .01;
    end
    
    methods (Static)
        function [vec] = preprocess(img)
%             vec = (-.1) + (1.175+.1)*(img(:)>0);
            vec = (img(:)>0);
            vec = vec - mean(vec);
        end
        function [imgN] = normalize(img)
            imgN = img>0;
            imgN = imgN - mean(imgN);
        end
        function [vec] = flatten(img)
            vec = img(:);
        end
        function outvec = sigmoid(invec)
            outvec = CNN.A*tanh(CNN.S*invec);
        end
        function outvec = dsigmoid(invec)
            outvec = CNN.A * CNN.S * ...
                (1 - tanh(CNN.S*invec).^2);
            outvec(isnan(outvec)) = 99999;
            outvec(outvec==0) = 0.01;
        end
    end
    
    methods
        function obj = CNN(numIn, numOut)
            obj.numIn = numIn;
            obj.numOut = numOut;
%             obj.Layers = {FullyConnectedNN(prod(numIn), 150),...
%                           FullyConnectedNN(150, numOut)};
            obj.Layers = {FullyConnectedNN(prod(numIn), numOut)};
%             obj.Layers = {DownsampleLayer(1,numIn,2),...
%                           FullyConnectedNN(prod(numIn/2), numOut)};
            obj.numLayers = numel(obj.Layers);
            obj.SaveStates = cell(obj.numLayers+1,1);
            obj.saveInd = 1;
        end
        function [outs,allStates] = evaluate(obj, ins, saveState)
            if (nargin<3)
                saveState = true;
            end
            allStates = cell(obj.numLayers+1,1);
            outs = CNN.preprocess(ins);
            allStates{1} = outs;
            for layerInd = 1:obj.numLayers
                outs = obj.Layers{layerInd}.evaluate(outs);
                allStates{layerInd+1} = outs;
            end
            if (saveState)
                for i = 1:length(obj.SaveStates)
                    obj.SaveStates{i}(obj.saveInd,:) = allStates{i};
                end
                obj.saveInd = obj.saveInd+1;
            end
        end
        function [] = backPropogate(obj, res, states, strength)
            if (nargin<3 || isempty(states))
                % load SaveStates into states and zero out SaveStates
                for in = 1:length(obj.SaveStates)
                    states{in} = obj.SaveStates{in}(1:obj.saveInd-1,:);
                    obj.SaveStates{in}(:) = 0;
                end
                obj.saveInd = 1;
            end
            for layerInd = obj.numLayers:-1:1
                res = obj.Layers{layerInd}.backPropogate(states{layerInd}, states{layerInd+1}, res, strength);
            end
        end
    end
end