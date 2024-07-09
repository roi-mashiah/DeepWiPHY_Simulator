function [yOut] = pyLogSoftmax(X ,dim)
% Implements the Pytorch LogSoftmax function:
% LogSoftmax(input, axis) = Log (Exp(input) / Sum(Exp(input), axis=axis)


%   Copyright 2022-2023 The MathWorks, Inc.

import classfier_1.ops.*

Xval = X.value;
Xrank = X.rank;


dim = dim.value;

%Convert negative index to positive
if dim < 0
    dim = dim + Xrank;
end

%Convert Dim to reverse Python
dim = Xrank - dim;

Xval = Xval - max(Xval, [], dim);
expX = exp(Xval);
sfMax = expX ./ sum(expX, dim);
Yval = log(sfMax);
Yrank = Xrank;
Yval = dlarray(Yval, repmat('U',1,Yrank));


yOut = struct('value',Yval,'rank',Yrank);

end

