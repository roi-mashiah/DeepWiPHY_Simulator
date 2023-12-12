scenario = struct;
addpath(".\helpers\")

% configure the transmission packet
cfgHE = wlanHESUConfig;
cfgHE.ChannelBandwidth = 'CBW20';  % Channel bandwidth
cfgHE.NumSpaceTimeStreams = 1;     % Number of space-time streams
cfgHE.NumTransmitAntennas = 1;     % Number of transmit antennas
cfgHE.APEPLength = 1e3;            % Payload length in bytes (# of OFDM symbols in data field)
cfgHE.ExtendedRange = false;       % Do not use extended range format
cfgHE.Upper106ToneRU = false;      % Do not use upper 106 tone RU
cfgHE.PreHESpatialMapping = false; % Spatial mapping of pre-HE fields
cfgHE.GuardInterval = 0.8;         % Guard interval duration
cfgHE.HELTFType = 4;               % HE-LTF compression mode
cfgHE.ChannelCoding = 'LDPC';      % Channel coding - FEC
cfgHE.MCS = 3;                     % Modulation and coding scheme (3 - 16QAM, 1/2 coding rate)

% Create and configure the TGax channel
chanBW = cfgHE.ChannelBandwidth;
tgaxChannel = wlanTGaxChannel;
tgaxChannel.DelayProfile = 'Model-A'; % simple channel
tgaxChannel.NumTransmitAntennas = cfgHE.NumTransmitAntennas;
tgaxChannel.NumReceiveAntennas = 1; % SISO 
tgaxChannel.TransmitReceiveDistance = 5; % Distance in meters for NLOS
tgaxChannel.ChannelBandwidth = chanBW;
tgaxChannel.LargeScaleFadingEffect = 'None';
tgaxChannel.NormalizeChannelOutputs = false;
fs = wlanSampleRate(cfgHE);
tgaxChannel.SampleRate = fs;

scenario.tx.HE_config = cfgHE;
scenario.tx.tgax_channel = tgaxChannel;
scenario.tx.numPackets = 10;

% Generate seed
scenario.seed = round((now - datenum('1/1/2020'))*100);

% Get Ground Truth QAMS
scenario.gt.data_symbols = create_scenario_gt(scenario);
% Get GT Channel Estimation
x = zeros(256,1);
x(1) = 1;
scenario.gt.channel_taps_gt = fftshift(abs(scenario.tx.tgax_channel(x)).^2);

