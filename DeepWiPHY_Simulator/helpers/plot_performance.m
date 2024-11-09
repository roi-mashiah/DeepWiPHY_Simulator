function [] = plot_performance(snr,mcs,packetErrorRate,packetErrorRateNn,packetErrorRateSmoother,packetErrorRateSmootherOrig,packetErrorRateMeanSmoother,packetErrorRateGenieClassifierNN,packetErrorRateGenieSmoother, scenario)
cfgHE = scenario.tx.HE_config;
tgaxChannel = scenario.tx.tgax_channel;

for m=1:unique(mcs)
    ind = mcs == m;
    figure;
    set(groot,'defaultLineMarkerSize',10);
    set(groot,"defaultLineLineWidth",1.1);
    semilogy(snr(ind),packetErrorRate(ind),'-*');
    hold on;
    % semilogy(snr,packetErrorRateGenieClassifierNN, 'k-diamond')
    semilogy(snr(ind),packetErrorRateGenieSmoother(ind), 'g-diamond',"MarkerSize",14)
    semilogy(snr(ind),packetErrorRateMeanSmoother(ind), 'r-hexagram',"MarkerSize",18)
    semilogy(snr(ind),packetErrorRateSmootherOrig(ind), 'c-square');
    % semilogy(snr,packetErrorRateNn,'m-^',"MarkerFaceColor","m",MarkerSize=4);
    semilogy(snr(ind),packetErrorRateSmoother(ind),'b-o',"MarkerFaceColor","b", "MarkerSize",4);


    grid on;
    xlabel('SNR (dB)');
    ylabel('PER');
    % dataStr = arrayfun(@(x)sprintf('MCS %d',x),cfgHE.MCS,'UniformOutput',false);
    % legend(["Baseline","Genie Classifier NN","Genie DS Smoother","Fixed DS Smoother (30ns)","M=5 Smoother","Full NN","Classifier based Smoother"]);
    legend(["Baseline","Genie DS Smoother","Fixed DS Smoother (30ns)","M=9 Smoother","Classifier based Smoother"]);

    title(sprintf('PER - Channel %s, %s, MCS %d',tgaxChannel.DelayProfile,cfgHE.ChannelBandwidth, m));
    output_path = "results/per_figs";
    filename = sprintf("CH_%s_MCS_%d_PER.png", tgaxChannel.DelayProfile, m);
    outfilename = fullfile(output_path, filename);
    saveas(gcf, outfilename);
end
end

