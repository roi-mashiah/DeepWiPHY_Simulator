function [] = create_traintest_data(dataPath)
allResults = dir(fullfile(dataPath,"**\*.mat"));
for f=1:length(allResults)
    full_filename = fullfile(allResults(f).folder, allResults(f).name);
    res = load(full_filename);
    activeFFTInd = res.scenario.tx.ofdmInfo.ActiveFFTIndices;
    gt_channel_f = fft(res.scenario.gt.channel_taps_gt);
    channelTapsGt = gt_channel_f(activeFFTInd);
    for i=1:length(res.scenario.rx.HE_LTF)
        heLtf = res.scenario.rx.HE_LTF{i};
        if ~isempty(heLtf)
            real_t_filename = replace(full_filename,".mat",strcat("_packet_",num2str(i),"_real.csv"));
            imag_t_filename = replace(full_filename,".mat",strcat("_packet_",num2str(i),"_imag.csv"));
            data_to_save = [channelTapsGt heLtf];
            writetable(array2table(real(data_to_save),"VariableNames",{'channel_taps','HE-LTF'}),real_t_filename);
            writetable(array2table(imag(data_to_save),"VariableNames",{'channel_taps','HE-LTF'}),imag_t_filename);
        end
    end
end
end

