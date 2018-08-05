%%  Gerry Chen
%   FlattenImagesLayer  -  convenience "fake" layer for flattening an image

classdef FlattenImagesLayer < NNLayer
    
    properties
        numImgs
        imgSize
    end
    
    methods (Static)
    end
    methods
        function obj = FlattenImagesLayer(numImgs, imgSize)
            obj.numImgs = numImgs;
            obj.imgSize = imgSize;
        end
        function outs = evaluate(obj, ins)
            outs = ins(:);
        end
        function [nextRes] = backPropogate(obj, ins, outs, res, strength)
            nextRes = reshape(res,[size(res,1),obj.numImgs,obj.imgSize]);
        end
    end
end