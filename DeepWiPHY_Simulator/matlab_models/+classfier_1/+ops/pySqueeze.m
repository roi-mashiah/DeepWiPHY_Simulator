function [Yout] = pySqueeze(varargin)
    %Implementation of PyTorch Squeeze Operator
    % at::Tensor at::squeeze(const at::Tensor &self)
    % at::Tensor at::squeeze(const at::Tensor &self, int64_t dim)
    
    %Copyright 2022-2023 The MathWorks, Inc.

    import classfier_1.ops.*

    Xin = varargin{1};
    dim = [];
    if nargin > 1
        dimStruct = varargin{2};
        dim = dimStruct.value;
    end
    
    Xval = Xin.value;
    Xrank = Xin.rank;
    
    Xshape = size(Xval, 1:Xrank);
    
    %If dim is empty, squeeze all singleton dimensions
    if isempty(dim)
        Yshape = Xshape(Xshape ~= 1);
    else
        % Convert dim to reverse-pytorch
        if (dim<0)
            dim = -dim;
        else
            dim = Xrank - dim; 
        end
        Yshape = Xshape;
        Yshape(dim) = [];
    end
    
    Yrank  = numel(Yshape);
    Yshape = [Yshape ones(1, 2-Yrank)];    % Append 1's to shape if numDims<2
    
    Yval         = reshape(Xval, Yshape);
    
    %output in reverse PyTorch Ordering - U-labelled
    Yval = dlarray(Yval, repmat('U',1,Yrank));
    
    Yout = struct('value', Yval, 'rank', Yrank);

end