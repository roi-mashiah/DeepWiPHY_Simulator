function [gt_qams] = create_scenario_gt(scenario)

maxNumErrors = 10;   % The maximum number of packet errors at an SNR point
maxNumPackets = scenario.tx.numPackets; % The maximum number of packets at an SNR point

% Calculate ground truth in high snr
cfgHE = scenario.tx.HE_config;
tgaxChannel = scenario.tx.tgax_channel;
chanBW = scenario.tx.HE_config.ChannelBandwidth;
fs = tgaxChannel.SampleRate;

% Get occupied subcarrier indices and OFDM parameters
ofdmInfo = wlanHEOFDMInfo('HE-Data',cfgHE);

% Indices to extract fields from the PPDU-returns a struct with indices of the different fields - ex: ind.HELTF = [a b]
ind = wlanFieldIndices(cfgHE);

% Set high SNR
snr = 100;
isnr=1;

% Set random substream index per iteration to ensure that each
% iteration uses a repeatable set of random numbers
stream = RandStream('combRecursive','Seed',scenario.seed);
stream.Substream = isnr;
RandStream.setGlobalStream(stream);

% Account for noise energy in nulls so the SNR is defined per
% active subcarrier
packetSNR = snr(isnr)-10*log10(ofdmInfo.FFTLength/ofdmInfo.NumTones);

% Loop to simulate multiple packets
numPacketErrors = 0;
numPkt = 1; % Index of packet transmitted
gt_qams = {};
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
    heltfDemod = wlanHEDemodulate(rxHELTF,'HE-LTF',cfgHE); % freq domain symbols per channel
    [chanEst,pilotEst] = wlanHELTFChannelEstimate(heltfDemod,cfgHE); % freq domain channel estimation

    % Data demodulate
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
    gt_qams{numPkt} = heEqualizeCombine(demodDataSym,chanEstData,nVarEst,cfgHE);
    numPkt = numPkt+1;
end

