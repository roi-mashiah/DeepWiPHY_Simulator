clear all;
resultsDir = "../results/traced";
tracedModels = dir(resultsDir);
prefix = "channel_est";
s = struct;
c = 1;
for i=1:length(tracedModels)
    filename = tracedModels(i).name;
    if ~contains(filename,prefix)
        continue
    end
    fullPath = fullfile(tracedModels(i).folder, filename);
    traced = importNetworkFromPyTorch(fullPath);
    traced = addInputLayer(traced,inputLayer([2,242],"CT"),Initialize=true);
    fieldname = replace(filename,".pt","");
    s.(fieldname) = traced; 
end

save("estimators.mat", "s");