function [Y, runningMean, runningVar] = pyBatchNorm(X, scale, offset, runningMean, runningVar, momentum, epsilon, training)
%PYBATCHNORM Normalizes mini-batches of data for each channel.

%Copyright 2022-2023 The MathWorks, Inc.

import ds_E.ops.*

%Input is permuted to reverse PyTorch at the layer level
Xval = X.value;
Xrank = X.rank;
scale = scale.value;
offset = offset.value;
runningMean = runningMean.value;
runningVar = runningVar.value;
momentum = momentum.value;
epsilon = max(double(epsilon.value), 1e-5); % Epsilon must be greater than or equal to 1e-05.

% Running statistics may not be dlarrays
if isdlarray(runningMean)
    runningMean = extractdata(runningMean);
end
if isdlarray(runningVar)
    runningVar = extractdata(runningVar);
end
runningVar(runningVar <= 0) = realmin('single');  % Set nonpositive variance components to a value below eps('single')

% Apply label to X based on its rank
[Xval, ptFormat] = applyDataFormat(Xval, Xrank);

% scale and offset are vectors
if isempty(scale)
    scale = ones(size(Xval, finddim(Xval, "C")), 1, 'like', Xval);
end

if isempty(offset)
    offset = zeros(size(Xval, finddim(Xval, "C")), 1, 'like', Xval);
end

scale = dlarray(scale(:),'CU');
offset = dlarray(offset(:),'CU');


% Calculate Y and running statistics (if training)
if ~training
    Y = batchnorm(Xval, offset, scale, runningMean, runningVar, "Epsilon", epsilon, "MeanDecay", momentum, "VarianceDecay", momentum);    
else
    [Y, runningMean, runningVar] = batchnorm(Xval, offset, scale, runningMean, runningVar, "Epsilon", epsilon, "MeanDecay", momentum, "VarianceDecay", momentum);
end

YrevPyTorch = permuteToReversePyTorch(Y, ptFormat);
% Return outputs as structs
Y = struct('value', YrevPyTorch, 'rank', Xrank);
runningMean = struct('value', runningMean, 'rank', 1);
runningVar = struct('value', runningVar, 'rank', 1);

end

function [X, ptFmt] = applyDataFormat(X, Xrank)
% Makes X a labelled dlarray
% X must have at least 2 dimensions in PyTorch.
assert(Xrank >=2, message("nnet_cnn_pytorchconverter:pytorchconverter:UnsupportedArgument", "pyBatchNorm", "X"));
if Xrank == 3 % BCT PyTorch -> TCB Reverse-PyTorch
    fmt = 'TCB';
    ptFmt = '*CT';
else
    % PyTorch format is BC*, where * is any number of S's
    % Reverse-PyTorch format will be *CB
    numS = Xrank-2;        
    fmt = [repmat('S', 1, numS), 'CB'];
    ptFmt = ['*C',repmat('S',1,numS)];

    % Permute the spatial dims to MATLAB-ordering
    X = permute(X,  [numS:-1:1 numS+1:Xrank]);
end
X = dlarray(X, fmt);
end