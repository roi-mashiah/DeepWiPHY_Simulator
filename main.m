clear all;close all;clc
addpath("helpers\","scenarios\");
%% load scenarios
scenario_paths = dir("scenarios\*.m");
scenarios = cell(size(scenario_paths,1),1);
for f = 1:numel(scenario_paths)
    clear scenario
    [~,filename,~] = fileparts(scenario_paths(f).name);
    eval(filename);
    scenarios{f} = scenario;
end

%% global configs and preallocs
save_scenario = 1;
maxNumPackets = 10000;
maxNumErrors = 0.5*maxNumPackets;   % The maximum number of packet errors at an SNR point
snr = 12:4:40;
% snr = 12:2:20;
numSNR = numel(snr); % Number of SNR points
packetErrorRate = zeros(1,numSNR);
plot_ch = 0; plot_symb = 0; plot_perf=0;

for sc_ind = 1:numel(scenarios)
    scenario = scenarios{sc_ind};
    cfgHE = scenario.tx.HE_config;
    tgaxChannel = scenario.tx.tgax_channel;
    chanBW = scenario.tx.HE_config.ChannelBandwidth;
    scenario.tx.numPackets = maxNumPackets;

    % Get occupied subcarrier indices and OFDM parameters
    ofdmInfo = wlanHEOFDMInfo('HE-Data',cfgHE);
    scenario.tx.ofdmInfo = ofdmInfo;
    fs = tgaxChannel.SampleRate;
    % Indices to extract fields from the PPDU-returns a struct with indices of the different fields - ex: ind.HELTF = [a b]
    ind = wlanFieldIndices(cfgHE);

    for isnr = 1:numSNR
        % Set random substream index per iteration to ensure that each
        % iteration uses a repeatable set of random numbers
        stream = RandStream('combRecursive','Seed',scenario.seed);
        stream.Substream = isnr;
        RandStream.setGlobalStream(stream);

        % Account for noise energy in nulls so the SNR is defined per
        % active subcarrier
        packetSNR = snr(isnr)-10*log10(ofdmInfo.FFTLength/ofdmInfo.NumTones);
        scenario.gt.realSnr = packetSNR;
        % Loop to simulate multiple packets
        numPacketErrors = 0;
        numPkt = 1; % Index of packet transmitted
        while numPacketErrors<=maxNumErrors && numPkt<=maxNumPackets
            % Generate a packet with random PSDU
            psduLength = getPSDULength(cfgHE); % PSDU length in bytes
            txPSDU = randi([0 1],psduLength*8,1); % times 8 since we send bits (not bytes)
            tx = wlanWaveformGenerator(txPSDU,cfgHE); % IQ Data

            % Add trailing zeros to allow for channel delay
            txPad = [tx; zeros(50,cfgHE.NumTransmitAntennas)];

            % Pass through a fading indoor TGax channel
            reset(tgaxChannel); % Reset channel for different realization
            rx = tgaxChannel(txPad);

            % Get GT Channel Estimation
            x = zeros(256,1);
            x(1) = 1;
            scenario.gt.channel_taps_gt{numPkt} = tgaxChannel(x);

            % Pass the waveform through AWGN channel
            rx = awgn(rx,packetSNR); % noisy IQ RX signal

            % Packet detect and determine coarse packet offset
            coarsePktOffset = wlanPacketDetect(rx,chanBW);
            if isempty(coarsePktOffset) % If empty, no L-STF detected; packet error
                numPacketErrors = numPacketErrors+1;
                numPkt = numPkt+1;
                continue; % Go to next loop iteration
            end

            % Extract L-STF and perform coarse frequency offset correction
            lstf = rx(coarsePktOffset+(ind.LSTF(1):ind.LSTF(2)),:);
            coarseFreqOff = wlanCoarseCFOEstimate(lstf,chanBW);
            rx = frequencyOffset(rx,fs,-coarseFreqOff); % Matlab 2022A complient

            % Extract the non-HT fields and determine fine packet offset
            nonhtfields = rx(coarsePktOffset+(ind.LSTF(1):ind.LSIG(2)),:);
            finePktOffset = wlanSymbolTimingEstimate(nonhtfields,chanBW);

            % Determine final packet offset
            pktOffset = coarsePktOffset+finePktOffset;

            % If packet detected outwith the range of expected delays from
            % the channel modeling; packet error
            if pktOffset>50
                numPacketErrors = numPacketErrors+1;
                numPkt = numPkt+1;
                continue; % Go to next loop iteration
            end

            % Extract L-LTF and perform fine frequency offset correction
            rxLLTF = rx(pktOffset+(ind.LLTF(1):ind.LLTF(2)),:);
            fineFreqOff = wlanFineCFOEstimate(rxLLTF,chanBW);
            rx = frequencyOffset(rx,fs,-fineFreqOff);

            % HE-LTF demodulation and channel estimation
            rxHELTF = rx(pktOffset+(ind.HELTF(1):ind.HELTF(2)),:); % time sig
            heltfDemod = wlanHEDemodulate(rxHELTF,'HE-LTF',cfgHE); % freq domain samples of HE-LTF
            [chanEst,pilotEst] = wlanHELTFChannelEstimate(heltfDemod,cfgHE); % freq domain channel estimation

            % log HE-LTF data for training, channel estimation for
            % reference and comparisons
            scenario.rx.HE_LTF{numPkt} = heltfDemod;
            scenario.rx.channel_est{numPkt} = chanEst;

            % Data demodulate - # symbols = # samples / (fftSize + CPSize)
            rxData = rx(pktOffset+(ind.HEData(1):ind.HEData(2)),:);
            demodSym = wlanHEDemodulate(rxData,'HE-Data',cfgHE);

            % Pilot phase tracking
            demodSym = wlanHETrackPilotError(demodSym,chanEst,cfgHE,'HE-Data');

            % Estimate noise power in HE fields
            nVarEst = heNoiseEstimate(demodSym(ofdmInfo.PilotIndices,:,:),pilotEst,cfgHE);

            % Extract data subcarriers from demodulated symbols and channel
            % estimate
            demodDataSym = demodSym(ofdmInfo.DataIndices,:,:);
            chanEstData = chanEst(ofdmInfo.DataIndices,:,:);

            % Equalization and STBC combining
            [eqDataSym,csi] = heEqualizeCombine(demodDataSym,chanEstData,nVarEst,cfgHE);

            % log symbols to calculate SER
            scenario.rx.data_symbols{numPkt} = eqDataSym;

            if plot_symb
                ref = scenario.gt{numPkt};
                plot_symb_ref(ref,eqDataSym)
            end


            % Recover data
            rxPSDU = wlanHEDataBitRecover(eqDataSym,nVarEst,csi,cfgHE,'LDPCDecodingMethod','norm-min-sum');

            % Determine if any bits are in error, i.e. a packet error
            packetError = ~isequal(txPSDU,rxPSDU);
            if packetError
                numPacketErrors = numPacketErrors+packetError;
            end
            numPkt = numPkt+1;
        end
        if save_scenario
            filename = strcat("sc_",num2str(convertTo(datetime,'epochtime')),"_snr_",num2str(snr(isnr)),"_ch_",tgaxChannel.DelayProfile(end),".mat");
            if ~exist(".\data","dir")
                mkdir(".\data")
            end
            save(fullfile("data\",filename),"scenario");
        end
        if plot_ch
            plot_channel(scenario)
        end

        % Calculate packet error rate (PER) at SNR point
        packetErrorRate(isnr) = numPacketErrors/(numPkt-1);
        disp(['MCS ' num2str(cfgHE.MCS) ','...
            ' SNR ' num2str(snr(isnr)) ...
            ' completed after ' num2str(numPkt-1) ' packets,'...
            ' PER:' num2str(packetErrorRate(isnr))]);

    end

    if plot_perf
        plot_performance(snr,packetErrorRate,scenario)
    end
end