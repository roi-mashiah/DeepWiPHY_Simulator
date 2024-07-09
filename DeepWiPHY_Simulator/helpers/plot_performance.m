function [] = plot_performance(snr,packetErrorRate,packetErrorRateNN,method,scenario)
cfgHE = scenario.tx.HE_config;
tgaxChannel = scenario.tx.tgax_channel;

figure;
semilogy(snr,packetErrorRate,'-*');
hold on;
semilogy(snr,packetErrorRateNN,'-*');
grid on;
xlabel('SNR (dB)');
ylabel('PER');
% dataStr = arrayfun(@(x)sprintf('MCS %d',x),cfgHE.MCS,'UniformOutput',false);
legend(["Baseline", "NN"]);
title(sprintf('PER for HE Channel %s, %s, %s, PSDULength: %d, Method: %d',tgaxChannel.DelayProfile,cfgHE.ChannelBandwidth,cfgHE.ChannelCoding,cfgHE.APEPLength, method));
end

