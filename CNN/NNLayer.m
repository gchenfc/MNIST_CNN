%%  Gerry Chen
%   NNLayer  -  interface for a neural net layer

classdef NNLayer < handle
    
    properties
    end
    
    methods (Abstract)
        evaluate(obj, ins)
        backPropogate(obj, ins, outs, res, strength)
    end
end