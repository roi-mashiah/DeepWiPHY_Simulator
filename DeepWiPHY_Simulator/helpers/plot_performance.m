function [] = plot_performance(snr,packetErrorRate,packetErrorRateNn,packetErrorRateSmoother,packetErrorRateSmootherOrig,packetErrorRateMeanSmoother,packetErrorRateGenieClassifierNN,packetErrorRateGenieSmoother, scenario)
cfgHE = scenario.tx.HE_config;
tgaxChannel = scenario.tx.tgax_channel;

figure;
set(groot,'defaultLineMarkerSize',10);
set(groot,"defaultLineLineWidth",1.1);
semilogy(snr,packetErrorRate,'-*');
hold on;
semilogy(snr,packetErrorRateGenieClassifierNN, 'k-diamond')
semilogy(snr,packetErrorRateGenieSmoother, 'g-diamond',"MarkerSize",14)
semilogy(snr,packetErrorRateMeanSmoother, 'r-hexagram',"MarkerSize",18)
semilogy(snr,packetErrorRateSmootherOrig, 'c-square');
semilogy(snr,packetErrorRateNn,'m-^',"MarkerFaceColor","m",MarkerSize=4);
semilogy(snr,packetErrorRateSmoother,'b-o',"MarkerFaceColor","b", "MarkerSize",4);


grid on;
xlabel('SNR (dB)');
ylabel('PER');
% dataStr = arrayfun(@(x)sprintf('MCS %d',x),cfgHE.MCS,'UniformOutput',false);
legend(["Baseline","Genie Classifier NN","Genie DS Smoother","Fixed DS Smoother (30ns)","M=5 Smoother","Full NN","Classifier based Smoother"]);
title(sprintf('PER - Channel %s, %s',tgaxChannel.DelayProfile,cfgHE.ChannelBandwidth));
output_path = "results/per_figs";
filename = sprintf("CH_%s_PER.png", tgaxChannel.DelayProfile);
outfilename = fullfile(output_path, filename);
saveas(gcf, outfilename);
end

