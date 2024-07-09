function Y = pyUnsqueeze(X, dim)
%PYUNSQUEEZE Inserts a singleton dimension at the position given by dim.
% at::Tensor at::unsqueeze(const at::Tensor &self, int64_t dim)

%   Copyright 2022-2023 The MathWorks, Inc.

import channel_est_D_1.ops.*

dim = dim.value;

Xval = X.value;
Xrank = X.rank;

% Convert dim to reverse-pytorch
if (dim<0)
    dim = -dim;
else
    dim = Xrank - dim + 1; 
end

% Reshape the data, inserting a singleton dim
Yrank = Xrank + 1;
if Yrank == 1
    newShape = size(Xval);
else
    newShape = ones(1, Yrank);
    knownSizes = setdiff(1:Yrank, dim);
    newShape(knownSizes) = size(Xval, 1:numel(knownSizes));
end

Yval = reshape(Xval, newShape);
Yval = dlarray(Yval, repmat('U', 1, max(2,Yrank)));
Y = struct('value', Yval, 'rank', Yrank);
end