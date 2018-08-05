%%  Gerry Chen
%   DownsampleLayer  -  object for a downsampling neural net layer

classdef DownsampleLayer < NNLayer
    
    properties
        numImgs
        imgSizeIn
        downsampleRatio
        mult
        bias
    end
    
    methods (Static)
    end
    methods
        function obj = DownsampleLayer(numImgs, imgSizeIn, downsampleRatio)
            if (nargin<3)
                downsampleRatio = 2;
            end
            obj.numImgs = numImgs;
            obj.imgSizeIn = imgSizeIn;
            obj.mult = ones(numImgs,1);
            obj.bias = zeros(numImgs,1);
            obj.downsampleRatio = downsampleRatio;
        end
        function outs = evaluate(obj, ins)
            ins = reshape(ins,[obj.numImgs,obj.imgSizeIn]);
            figure(5);clf;
            subplot(2,1,1);imagesc(reshape(ins,28,28));
            assert(ndims(ins)==3 && all(size(ins) == [obj.numImgs,obj.imgSizeIn]), ...
                'incorrect inputs sizing')
            outs = zeros([obj.numImgs,obj.imgSizeIn/obj.downsampleRatio]);
            for i = 1:obj.numImgs
                outs(i,:,:) = imresize(squeeze(ins(i,:,:)),1/obj.downsampleRatio,'bilinear') ...
                    * obj.mult(i) + obj.bias(i);
            end
            subplot(2,1,2);imagesc(reshape(outs,14,14));
            outs = outs(:);
            drawnow();
        end
        function [nextRes] = backPropogate(obj, ins, outs, res, strength)
            res = reshape(res,[size(res,1),obj.numImgs,obj.imgSizeIn/obj.downsampleRatio]);
            outs = reshape(outs,[size(outs,1),obj.numImgs,obj.imgSizeIn/obj.downsampleRatio]);
            nextRes = zeros([size(res,1),obj.numImgs,obj.imgSizeIn]);
            numTrials = size(res,1);
            for i = 1:numTrials
                for j = 1:obj.numImgs
                    nextRes(i,j,:,:) = imresize((squeeze(res(i,j,:,:))-obj.bias)/obj.mult, obj.downsampleRatio, 'bilinear');
                end
            end
            
            res = mean(mean(res,3),4); % diff per img
            outs = mean(mean(outs,3),4);
            obj.mult = obj.mult + mean(res.*outs)*strength;
            obj.bias = obj.bias + mean(res)*strength;
        end
    end
end