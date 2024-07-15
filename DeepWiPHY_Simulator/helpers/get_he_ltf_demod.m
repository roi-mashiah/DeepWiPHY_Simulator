function [heltfDemod, pktOffset] = get_he_ltf_demod(rx,chanBW,lstf_ind,fs,nonht_ind,lltf_ind,heltf_ind,cfgHE)
% Packet detect and determine coarse packet offset
coarsePktOffset = wlanPacketDetect(rx,chanBW);
if isempty(coarsePktOffset) % If empty, no L-STF detected; packet error
    heltfDemod = [];
    pktOffset = [];
    return;
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
    heltfDemod = [];
    pktOffset = [];
    return
end

% Extract L-LTF and perform fine frequency offset correction
rxLLTF = rx(pktOffset+(lltf_ind),:);
fineFreqOff = wlanFineCFOEstimate(rxLLTF,chanBW);
rx = frequencyOffset(rx,fs,-fineFreqOff);
% HE-LTF demodulation and channel estimation
rxHELTF = rx(pktOffset+(heltf_ind),:); % time sig
heltfDemod = wlanHEDemodulate(rxHELTF,'HE-LTF',cfgHE); % freq domain samples of HE-LTF
end

