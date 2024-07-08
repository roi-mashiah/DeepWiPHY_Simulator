clear all;close all;clc
addpath("./helpers","./scenarios","./matlab_models");
%% load scenarios
scenario_paths = dir("scenarios/*.m");
scenarios = cell(size(scenario_paths,1),1);
for f = 1:numel(scenario_paths)
    clear scenario
    [~,filename,~] = fileparts(scenario_paths(f).name);
    eval(filename);
    scenarios{f} = scenario;
end
%% load NN models
model_paths = dir("matlab_models/*.mat");
for f = 1:numel(model_paths)
    file_name = fullfile(model_paths(f).folder,model_paths(f).name);
    load(file_name)
end
estimators = {channel_estimator_a,... 
              channel_estimator_b,...
              channel_estimator_c,...
              channel_estimator_d,...
              channel_estimator_e,...
              channel_estimator_f};
nnMode = 0; % 0 - smoother, 1 - clsfr -> est
dsLookup = [0    15    30    50   100   150];
%% global configs and preallocs
save_scenario = 0;
maxNumPackets = 500;
maxNumErrors = 0.1*maxNumPackets;   % The maximum number of packet errors at an SNR point
snr = 10:5:30;
numSNR = numel(snr); % Number of SNR points
packetErrorRate = zeros(1,numSNR);
packetErrorRateNN = zeros(1,numSNR);
plot_ch = 0; plot_symb = 0; plot_perf=1;
output_data_dir = "/home/tauproj3/data/deepWiPhyData/matfiles";

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
    Ts = 1/fs;
    % Indices to extract fields from the PPDU-returns a struct with indices of the different fields - ex: ind.HELTF = [a b]
    ind = wlanFieldIndices(cfgHE);
    seed = scenario.seed;
    fftLength = ofdmInfo.FFTLength;
    numTones = ofdmInfo.NumTones;
    lstf_ind = ind.LSTF(1):ind.LSTF(2);
    nonht_ind = ind.LSTF(1):ind.LSIG(2);
    lltf_ind = ind.LLTF(1):ind.LLTF(2);
    heltf_ind = ind.HELTF(1):ind.HELTF(2);
    hedata_ind = ind.HEData(1):ind.HEData(2);
    pilot_ind = ofdmInfo.PilotIndices;
    data_ind = ofdmInfo.DataIndices;

    delete(gcp("nocreate"))
    parpool('local',4);

    parfor isnr = 1:numSNR
        % Set random substream index per iteration to ensure that each
        % iteration uses a repeatable set of random numbers
        stream = RandStream('combRecursive','Seed',seed);
        stream.Substream = isnr;
        RandStream.setGlobalStream(stream);

        % Account for noise energy in nulls so the SNR is defined per
        % active subcarrier
        packetSNR = snr(isnr)-10*log10(fftLength/numTones);

        % Loop to simulate multiple packets
        numPacketErrors = 0;
        numPacketErrorsNN = 0;
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
            lstf = rx(coarsePktOffset+(lstf_ind),:);
            coarseFreqOff = wlanCoarseCFOEstimate(lstf,chanBW);
            rx = frequencyOffset(rx,fs,-coarseFreqOff); % Matlab 2022A complient

            % Extract the non-HT fields and determine fine packet offset
            nonhtfields = rx(coarsePktOffset+(nonht_ind),:);
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
            rxLLTF = rx(pktOffset+(lltf_ind),:);
            fineFreqOff = wlanFineCFOEstimate(rxLLTF,chanBW);
            rx = frequencyOffset(rx,fs,-fineFreqOff);

            % HE-LTF demodulation and channel estimation
            rxHELTF = rx(pktOffset+(heltf_ind),:); % time sig
            heltfDemod = wlanHEDemodulate(rxHELTF,'HE-LTF',cfgHE); % freq domain samples of HE-LTF
            [chanEst,pilotEst] = wlanHELTFChannelEstimate(heltfDemod,cfgHE); % freq domain channel estimation

            % predict channel using neural networks
            nnInput = [real(heltfDemod)' ; imag(heltfDemod)'];
            chanEstNNBased = zeros(size(chanEst));

            switch nnMode
                case 0
                    [~,channelTypePrediction] = max(predict(channel_classifier,nnInput));
                    smoother = ds_smoother(heltfDemod,packetSNR,dsLookup(channelTypePrediction));
                    % ZOH for edge effects
                    chanEst_padded = [zeros(4,1); chanEst; zeros(4,1)];
                    chanEst_padded(1:4)=chanEst(1);
                    chanEst_padded(end-4:end)=chanEst(end);
                    clean_est = filter(smoother,1,chanEst_padded);
                    chanEstNNBased = clean_est(5:end-4);
                case 1
                    [~,channelTypePrediction] = max(predict(channel_classifier,nnInput));
                    chanEstNN = squeeze(predict(estimators{channelTypePrediction},nnInput));
                    chanEstNNBased = double(chanEstNN(1,:) - 1j*chanEstNN(2,:))';
            end
           
            % Data demodulate - # symbols = # samples / (fftSize + CPSize)
            rxData = rx(pktOffset+(hedata_ind),:);
            demodSym = wlanHEDemodulate(rxData,'HE-Data',cfgHE);
%% NN based demodulation and equalization
            % Pilot phase tracking NN
            demodSymNN = wlanHETrackPilotError(demodSym,chanEstNNBased,cfgHE,'HE-Data');

            % Estimate noise power in HE fields NN
            nVarEstNN = heNoiseEstimate(demodSymNN(pilot_ind,:,:),chanEstNNBased(pilot_ind),cfgHE);

            % Extract data subcarriers from demodulated symbols and channel
            % estimate NN
            demodDataSymNN = demodSymNN(data_ind,:,:);
            chanEstDataNN = chanEstNNBased(data_ind,:,:);
            
            % Equalization and STBC combining NN
            [eqDataSymNN,csiNN] = heEqualizeCombine(demodDataSymNN,chanEstDataNN,nVarEstNN,cfgHE);
            
            % Recover data NN
            rxPSDUNN = wlanHEDataBitRecover(eqDataSymNN,nVarEstNN,csiNN,cfgHE,'LDPCDecodingMethod','norm-min-sum');

%% Original demodulation and equalization            
            % Pilot phase tracking
            demodSym = wlanHETrackPilotError(demodSym,chanEst,cfgHE,'HE-Data');            
           
            % Estimate noise power in HE fields
            nVarEst = heNoiseEstimate(demodSym(pilot_ind,:,:),pilotEst,cfgHE);
            
            % Extract data subcarriers from demodulated symbols and channel
            % estimate
            demodDataSym = demodSym(data_ind,:,:);
            chanEstData = chanEst(data_ind,:,:);

            % Equalization and STBC combining
            [eqDataSym,csi] = heEqualizeCombine(demodDataSym,chanEstData,nVarEst,cfgHE);

            % Recover data
            rxPSDU = wlanHEDataBitRecover(eqDataSym,nVarEst,csi,cfgHE,'LDPCDecodingMethod','norm-min-sum');

%% performance
            % Determine if any bits are in error, i.e. a packet error NN
            packetErrorNN = ~isequal(txPSDU,rxPSDUNN);
            % Determine if any bits are in error, i.e. a packet error
            packetError = ~isequal(txPSDU,rxPSDU);

            if packetError
                numPacketErrors = numPacketErrors+packetError;
            end
            if packetErrorNN
                numPacketErrorsNN = numPacketErrorsNN+packetErrorNN;
            end
            numPkt = numPkt+1;
        end

        % Calculate packet error rate (PER) at SNR point
        packetErrorRate(isnr) = numPacketErrors/(numPkt-1);
        disp(['MCS ' num2str(cfgHE.MCS) ','...
            ' SNR ' num2str(snr(isnr)) ...
            ' completed after ' num2str(numPkt-1) ' packets,'...
            ' BL PER:' num2str(packetErrorRate(isnr))]);
        % Calculate packet error rate (PER) at SNR point NN
        packetErrorRateNN(isnr) = numPacketErrorsNN/(numPkt-1);
        disp(['MCS ' num2str(cfgHE.MCS) ','...
            ' SNR ' num2str(snr(isnr)) ...
            ' completed after ' num2str(numPkt-1) ' packets,'...
            ' NN PER:' num2str(packetErrorRateNN(isnr))]);
    end

    if plot_perf
        plot_performance(snr,packetErrorRate,packetErrorRateNN,scenario)
    end
end
