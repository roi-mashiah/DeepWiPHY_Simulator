function Y = pyMaxPool1d(X,kernelSize,stride,padding,dilation,ceiling)
%PYMaxPool1d Applies 1D max pooling over an input signal.
% at::Tensor at::max_pool1d(const at::Tensor &self, at::IntArrayRef kernel_size,
% at::IntArrayRef stride = {}, at::IntArrayRef padding = 0,
% at::IntArrayRef dilation = 1, bool ceil_mode = false)

%   Copyright 2023 The MathWorks, Inc.

import channel_est_D_1.ops.*

[X] = labelWithPropagatedFormats(X,"*CT");

% dilation, ceiling are currently not supported by the 
% matlab implementation of the "maxpool" function the following pyTorch
%equivalents are used:dilation.value = 1, ceiling.value = false.

if any(dilation.value~=1)
    warning(message('nnet_cnn_pytorchconverter:pytorchconverter:NumericalMismatchInOperator', ...
        "pyMaxPool1d","aten::max_pool1d","dilation ~= 1"));
end

if ceiling.value == true
    warning(message('nnet_cnn_pytorchconverter:pytorchconverter:NumericalMismatchInOperator', ...
        "pyMaxPool1d","aten::max_pool1d","ceiling = true"));
end

Yval = maxpool(X.value, kernelSize.value,'Stride', stride.value,'Padding', padding.value,'PoolFormat','T');

%Permute to Reverse PyTorch Ordering (Op functions are expected to output
%Rev PyTorch always)
[YrevPyTorch, ~] = permuteToReversePyTorch(Yval,"*CT");
Y = struct('value',YrevPyTorch,'rank',X.rank);

end