function [] = create_traintest_data(dataPath)
allResults = dir(fullfile(dataPath,"**\*.mat"));
delete(gcp("nocreate"));
parpool('local',4);

for f=1:length(allResults)
    full_filename = fullfile(allResults(f).folder, allResults(f).name);
    res = load(full_filename);
    activeFFTInd = res.scenario.tx.ofdmInfo.ActiveFFTIndices;
    ltf_samples = res.scenario.rx.HE_LTF;
    channel_vec = res.scenario.gt.channel_taps_gt;
    channel_est_matlab = res.scenario.rx.channel_est;
    parfor i=1:length(res.scenario.rx.HE_LTF)
        heLtf = ltf_samples{i};
        gt_channel_f = fft(channel_vec{i});
        channelTapsGt = gt_channel_f(activeFFTInd);
        channelEstimation = channel_est_matlab{i};
        if ~isempty(heLtf)
            filename = replace(full_filename,".mat",strcat("_packet_",num2str(i),".csv"));
            data_to_save = [real(channelTapsGt) imag(channelTapsGt)  real(heLtf) imag(heLtf) real(channelEstimation) imag(channelEstimation)];
            writetable(array2table(data_to_save,"VariableNames",{'channel_taps_real','channel_taps_imag','HE-LTF_real','HE-LTF_imag','channel_est_real', 'channel_est_imag'}),filename);
        end
    end
end
end

