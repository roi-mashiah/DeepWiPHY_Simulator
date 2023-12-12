function [] = create_traintest_data(dataPath)
allResults = dir(fullfile(dataPath,"**\*.mat"));
for f=1:length(allResults)
    full_filename = fullfile(allResults(f).folder, allResults(f).name);
    res = load(full_filename);
    activeFFTInd = res.scenario.tx.ofdmInfo.ActiveFFTIndices;
    gt_channel_f = abs(fft(res.scenario.gt.channel_taps_gt)).^2;
    channelTapsGt = gt_channel_f(activeFFTInd);
    for i=1:res.scenario.tx.numPackets
        heLtf = res.scenario.rx.HE_LTF{i};
        if ~isempty(heLtf)
            real_t_filename = replace(full_filename,".mat",strcat("_packet_",num2str(i),"_real.csv"));
            imag_t_filename = replace(full_filename,".mat",strcat("_packet_",num2str(i),"_imag.csv"));
            writetable(array2table([channelTapsGt real(heLtf)],"VariableNames",{'channel_taps','HE-LTF'}),real_t_filename);
            writetable(array2table([channelTapsGt imag(heLtf)],"VariableNames",{'channel_taps','HE-LTF'}),imag_t_filename);
        end
    end
end
end

