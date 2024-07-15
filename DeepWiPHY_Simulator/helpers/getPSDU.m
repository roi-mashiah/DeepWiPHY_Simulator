function [rxPSDU] = getPSDU(demodSym, chanEst, cfgHE,pilot_ind, data_ind)
% Pilot phase tracking NN
demodSym = wlanHETrackPilotError(demodSym,chanEst,cfgHE,'HE-Data');

% Estimate noise power in HE fields NN
nVarEst = heNoiseEstimate(demodSym(pilot_ind,:,:),chanEst(pilot_ind),cfgHE);

% Extract data subcarriers from demodulated symbols and channel
% estimate NN
demodDataSym = demodSym(data_ind,:,:);
chanEstData = chanEst(data_ind,:,:);

% Equalization and STBC combining NN
[eqDataSym,csi] = heEqualizeCombine(demodDataSym,chanEstData,nVarEst,cfgHE);

% Recover data NN
rxPSDU = wlanHEDataBitRecover(eqDataSym,nVarEst,csi,cfgHE,'LDPCDecodingMethod','norm-min-sum');
end