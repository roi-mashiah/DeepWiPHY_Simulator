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
estimators = {s.channel_est_A_1,... 
              s.channel_est_B_1,...
              s.channel_est_C_1,...
              s.channel_est_D_1,...
              s.channel_est_E_1,...
              s.channel_est_F_1};

dsLookup = [0    15    30    50   100   150].*1e-9;
%% global configs and preallocs
save_scenario = 0;
maxNumPackets = 1000;
maxNumErrors = 0.2*maxNumPackets;   % The maximum number of packet errors at an SNR point
snr = 10:2:24;
numSNR = numel(snr); % Number of SNR points

packetErrorRateBaseline = zeros(1,numSNR);
packetErrorRateSmootherOrig = zeros(1,numSNR);
packetErrorRateNn = zeros(1,numSNR);
packetErrorRateNnSmoother = zeros(1,numSNR);
packetErrorRateGenieClassifierNN = zeros(1,numSNR);
packetErrorRateGenieSmoother = zeros(1,numSNR);
packetErrorRateMeanSmoother = zeros(1,numSNR);

confusionMatrix = zeros(numSNR,6,6);

plot_perf=1;
delete(gcp("nocreate"))
parpool('local',8);

for sc_ind = 1:numel(scenarios)
    scenario = scenarios{sc_ind};
    cfgHE = scenario.tx.HE_config;
    tgaxChannel = scenario.tx.tgax_channel;
    chanBW = scenario.tx.HE_config.ChannelBandwidth;    

    % Get occupied subcarrier indices and OFDM parameters
    ofdmInfo = wlanHEOFDMInfo('HE-Data',cfgHE);
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
        predictions = zeros(1,maxNumPackets);
        
        numPacketErrors = 0;
        numPacketErrorsNN = 0;
        numPacketErrorsSmoother = 0;        
        numPacketErrorsSmootherOrig = 0;        
        numPacketErrorsGenieClassifierNN = 0;
        numPacketErrorsGenieSmoother = 0;
        numPacketErrorsMeanSmoother = 0;

        numPkt = 1; % Index of packet transmitted
        c = 1;
        psduLength = getPSDULength(cfgHE); % PSDU length in bytes

        while numPacketErrors<=maxNumErrors && numPkt<=maxNumPackets            
            % Generate a packet with random PSDU
            txPSDU = randi([0 1],psduLength*8,1); % times 8 since we send bits (not bytes)
            tx = wlanWaveformGenerator(txPSDU,cfgHE); % IQ Data

            % Add trailing zeros to allow for channel delay
            txPad = [tx; zeros(50,cfgHE.NumTransmitAntennas)];

            % Pass through a fading indoor TGax channel
            reset(tgaxChannel); % Reset channel for different realization
            rx = tgaxChannel(txPad);
            clean_rx = rx;

            % Pass the waveform through AWGN channel
            rx = awgn(rx,packetSNR); % noisy IQ RX signal
                       
            [heltfDemod,pktOffset] = get_he_ltf_demod(rx,chanBW,lstf_ind,fs,nonht_ind,lltf_ind,heltf_ind,cfgHE);

            if isempty(heltfDemod)
                % timing error (freq offset too large)
                numPacketErrors = numPacketErrors+1;
                numPkt = numPkt+1;
                continue; % Go to next loop iteration
            end

            chanEstBaseline = wlanHELTFChannelEstimate(heltfDemod,cfgHE); % no smoother
            chanEstSmootherFixed = wlanHELTFChannelEstimate(heltfDemod,cfgHE, "FrequencySmoothingSpan",5); % all ones

            % predict channel using neural networks
            nnInput = [real(heltfDemod)' ; imag(heltfDemod)'];
            [~,channelTypePrediction] = max(predict(channel_classifier,nnInput));

            chanEstSmoother = apply_smoother(chanEstBaseline, packetSNR, dsLookup(channelTypePrediction)); % nn ch type smoother
            chanEstGenieSmoother = apply_smoother(chanEstBaseline, packetSNR, dsLookup(sc_ind)); % cheating ch type smoother
            chanEstMeanSmoother = apply_smoother(chanEstBaseline, packetSNR, 30e-9); % ch type C smoother
            

            chanEstNN = squeeze(predict(estimators{channelTypePrediction},nnInput)); % nn ch type + nn ch est
            chanEstNN = double(chanEstNN(1,:) - 1j*chanEstNN(2,:))';
            
            chanEstGenieNN = squeeze(predict(estimators{sc_ind},nnInput)); % cheating ch type + nn ch est
            chanEstGenieNN = double(chanEstGenieNN(1,:) - 1j*chanEstGenieNN(2,:))';
            

            predictions(c) = channelTypePrediction;
            c = c + 1;
            
            % Data demodulate - # symbols = # samples / (fftSize + CPSize)
            rxData = rx(pktOffset+(hedata_ind),:);
            demodSym = wlanHEDemodulate(rxData,'HE-Data',cfgHE);

            % demodulation and equalization
            rxPSDUBaseline = getPSDU(demodSym, chanEstBaseline,cfgHE, pilot_ind, data_ind);
            rxPSDUSmoother = getPSDU(demodSym, chanEstSmoother,cfgHE, pilot_ind, data_ind);
            rxPSDUSmootherFixed = getPSDU(demodSym, chanEstSmootherFixed,cfgHE, pilot_ind, data_ind);
            rxPSDUGenieSmoother = getPSDU(demodSym, chanEstGenieSmoother,cfgHE, pilot_ind, data_ind);
            rxPSDUMeanSmoother = getPSDU(demodSym, chanEstMeanSmoother,cfgHE, pilot_ind, data_ind);            
            rxPSDUNN = getPSDU(demodSym, chanEstNN,cfgHE, pilot_ind, data_ind);
            rxPSDUGenieNN = getPSDU(demodSym, chanEstGenieNN,cfgHE, pilot_ind, data_ind);

%% performance
            packetError = ~isequal(txPSDU,rxPSDUBaseline);
            packetErrorSmoother = ~isequal(txPSDU,rxPSDUSmoother);
            packetErrorSmootherFixed = ~isequal(txPSDU,rxPSDUSmootherFixed);
            packetErrorGenieSmoother = ~isequal(txPSDU,rxPSDUGenieSmoother);
            packetErrorMeanSmoother = ~isequal(txPSDU,rxPSDUMeanSmoother);
            packetErrorNN = ~isequal(txPSDU,rxPSDUNN);
            packetErrorGenieNN = ~isequal(txPSDU,rxPSDUGenieNN);                     

            if packetError
                numPacketErrors = numPacketErrors+packetError;
            end
            if packetErrorSmoother
                numPacketErrorsSmoother = numPacketErrorsSmoother+packetErrorSmoother;
            end
            if packetErrorSmootherFixed
                numPacketErrorsSmootherOrig = numPacketErrorsSmootherOrig+packetErrorSmootherFixed;
            end
            if packetErrorGenieSmoother
                numPacketErrorsGenieSmoother = numPacketErrorsGenieSmoother+packetErrorGenieSmoother;
            end
            if packetErrorMeanSmoother
                numPacketErrorsMeanSmoother = numPacketErrorsMeanSmoother+packetErrorMeanSmoother;
            end
            if packetErrorGenieNN
                numPacketErrorsGenieClassifierNN = numPacketErrorsGenieClassifierNN+packetErrorGenieNN;
            end
            if packetErrorNN
                numPacketErrorsNN = numPacketErrorsNN+packetErrorNN;
            end
            
            numPkt = numPkt+1;
        end

        for class=1:6
            confusionMatrix(isnr,sc_ind,class) = sum(predictions == class);
        end
        % Calculate packet error rate (PER) at SNR point
        packetErrorRateBaseline(isnr) = numPacketErrors/(numPkt-1);
        packetErrorRateNnSmoother(isnr) = numPacketErrorsSmoother/(numPkt-1);
        packetErrorRateNn(isnr) = numPacketErrorsNN/(numPkt-1);
        packetErrorRateGenieSmoother(isnr) = numPacketErrorsGenieSmoother/(numPkt-1);
        packetErrorRateGenieClassifierNN(isnr) = numPacketErrorsGenieClassifierNN/(numPkt-1);
        packetErrorRateMeanSmoother(isnr) = numPacketErrorsMeanSmoother/(numPkt-1);
        packetErrorRateSmootherOrig(isnr) = numPacketErrorsSmootherOrig/(numPkt-1);
    end

    if plot_perf
        plot_performance(snr, ...
            packetErrorRateBaseline, ...
            packetErrorRateNn, ...
            packetErrorRateNnSmoother, ...
            packetErrorRateSmootherOrig, ...
            packetErrorRateMeanSmoother, ...
            packetErrorRateGenieClassifierNN, ...
            packetErrorRateGenieSmoother, ...
            scenario);
    end
end
plot_confusion_matrix(snr, confusionMatrix);
