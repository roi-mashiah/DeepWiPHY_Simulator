function [Yout] = pyConvolution(X, weights, biasIn, strides, padding, dilations, transpose, outputPadding, groups)
%PyConvolution applies a convolution 

%Copyright 2022-2023 The MathWorks, Inc.

import channel_est_F_1.ops.*

inputDataFormat = igetDataFormat(X.rank);
X = labelWithPropagatedFormats(X,inputDataFormat);
Xval = X.value;
Xrank = X.rank;

weightVal = weights.value;
weightDim = size(weightVal);
if numel(weightDim) < weights.rank
    rankDiff = weights.rank - numel(weightDim);
    weightDim = [weightDim ones(1,rankDiff)];
end

biasVal = biasIn.value;

%Value is set to >1 for grouped convolution
groups = groups.value;

%Value is set to 1 for transpose convolution
transpose = transpose.value;

% Output Padding is not supported in MATLAB
% User will see a size mismatch error
if ~all(outputPadding.value == 0)
    error(message('nnet_cnn_pytorchconverter:pytorchconverter:UnsupportedArgument','pyConvolution','outputPadding'));
end

if transpose
   if numel(weightDim) == 3
       
        filterSize = weightDim(1);
        numFilters = weightDim(2);
        numChannel  = weightDim(3);
        
        if groups > 1
            numChannelsPerGroup = numChannel/groups;
            weightVal = reshape(weightVal, [filterSize, numFilters,...
                numChannelsPerGroup,groups]);
            weightVal = dlarray(weightVal,'TCUU');
            bias = zeros([1 numFilters groups]);
            if ~isempty(biasVal)
                bias = reshape(biasVal,[1, numFilters, groups]);
            end
           
        else
            bias = zeros([1 numFilters]);
    
            if ~isempty(biasVal)
                bias = biasVal(:);
            end
            weightVal = dlarray(weightVal,'TCU');
        end
    
    elseif numel(weightDim) == 4
        filterSize = [weightDim(2) weightDim(1)];
        numFilters = weightDim(3);
        numChannel  = weightDim(4);
        
        
        if groups > 1
            numChannelsPerGroup = numChannel/groups;
            weightVal = reshape(weightVal, [filterSize,numFilters,...
                numChannelsPerGroup,groups]);
            weightVal = permute(weightVal, [2,1,3,4,5]);
            weightVal = dlarray(weightVal,'SSCUU');
            bias = zeros([1 1 numFilters groups]);
            if ~isempty(biasVal)
                bias = reshape(biasVal(:),[1, 1, numFilters, groups]);
            end
           
        else
            bias = zeros([1 1 numFilters]);
            if ~isempty(biasVal)
                bias = reshape(biasVal(:), [1 1 numel(biasVal(:))]);
            end
            weightVal = permute(weightVal, [2,1,3,4]);
            weightVal = dlarray(weightVal,'SSCU');
        end
   else
        error(message('nnet_cnn_pytorchconverter:pytorchconverter:UnsupportedArgument','pyConvolution','weights'));
   end

    Yval = dltranspconv(Xval, weightVal, bias(:), 'Stride',strides.value,...
    'DilationFactor',dilations.value, 'Cropping',padding.value);

else
    if numel(weightDim) == 3
        filterSize = weightDim(1);
        numChannel  = weightDim(2);
        numFilters = weightDim(3);
        
        
        if groups > 1
            numFiltersPerGroup = numFilters/groups;
            weightVal = reshape(weightVal, [filterSize,numChannel,...
                                numFiltersPerGroup,groups]);
            weightVal = dlarray(weightVal,'TCUU');
            bias = zeros([1 numFiltersPerGroup, groups]);
            if ~isempty(biasVal)
                bias = reshape(biasVal(:),[1, numFiltersPerGroup, groups]);
            end
           
        else
            bias = zeros([1 weightDim(3:end)]);
            if ~isempty(biasVal)
                bias = biasVal(:);
            end
            weightVal = dlarray(weightVal,'TCU');
        end
    
    elseif numel(weightDim) == 4
        filterSize = [weightDim(2) weightDim(1)];
        numChannel  = weightDim(3);
        numFilters = weightDim(4);
    
        
        
        if groups > 1
            numFiltersPerGroup = numFilters/groups;
            weightVal = reshape(weightVal, [filterSize,numChannel,...
                                numFiltersPerGroup,groups]);
            weightVal = permute(weightVal, [2,1,3,4,5]);
            weightVal = dlarray(weightVal,'SSCUU');
            bias = zeros([1 1  numFiltersPerGroup groups]);
            if ~isempty(biasVal)
                bias = reshape(biasVal(:),[1, 1, numFiltersPerGroup, groups]);
            end
           
        else
            bias = zeros([1 1 weightDim(4:end)]);
            if ~isempty(biasVal)
                bias = reshape(biasVal(:), [1 1 numel(biasVal(:))]);
            end
            weightVal = permute(weightVal, [2,1,3,4]);
            weightVal = dlarray(weightVal,'SSCU');
        end
    else
        error(message('nnet_cnn_pytorchconverter:pytorchconverter:UnsupportedArgument','pyConvolution','weights'));
    end

    Yval = dlconv(Xval, weightVal, bias(:), 'Stride',strides.value,...
    'DilationFactor',dilations.value, 'Padding',padding.value);
end

[YrevPyTorch, ~] = permuteToReversePyTorch(Yval,inputDataFormat);
Yout = struct("value",YrevPyTorch,"rank",Xrank);

end



function fmt = igetDataFormat(Xrank)
if Xrank == 3 % BCT PyTorch
    fmt = 'BCT';
else
    % PyTorch format is BC*, where * is any number of S's
    % Reverse-PyTorch format will be *CB
    fmt = ['BC',repmat('S', 1, Xrank-2)];
end
end