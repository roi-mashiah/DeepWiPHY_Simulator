function [chanEst,pilotEst] = get_gt_channel(cleanRx,chanBW,fs,ind,cfgHE)
            % Packet detect and determine coarse packet offset
            coarsePktOffset = wlanPacketDetect(cleanRx,chanBW);

            % Extract L-STF and perform coarse frequency offset correction
            lstf = cleanRx(coarsePktOffset+(ind.LSTF(1):ind.LSTF(2)),:);
            coarseFreqOff = wlanCoarseCFOEstimate(lstf,chanBW);
            cleanRx = frequencyOffset(cleanRx,fs,-coarseFreqOff); % Matlab 2022A complient
            
            % Extract the non-HT fields and determine fine packet offset
            nonhtfields = cleanRx(coarsePktOffset+(ind.LSTF(1):ind.LSIG(2)),:);
            finePktOffset = wlanSymbolTimingEstimate(nonhtfields,chanBW);

            % Determine final packet offset
            pktOffset = coarsePktOffset+finePktOffset;

            % Extract L-LTF and perform fine frequency offset correction
            rxLLTF = cleanRx(pktOffset+(ind.LLTF(1):ind.LLTF(2)),:);
            fineFreqOff = wlanFineCFOEstimate(rxLLTF,chanBW);
            cleanRx = frequencyOffset(cleanRx,fs,-fineFreqOff);

            % HE-LTF demodulation and channel estimation
            rxHELTF = cleanRx(pktOffset+(ind.HELTF(1):ind.HELTF(2)),:); % time sig
            heltfDemod = wlanHEDemodulate(rxHELTF,'HE-LTF',cfgHE); % freq domain samples of HE-LTF
            [chanEst,pilotEst] = wlanHELTFChannelEstimate(heltfDemod,cfgHE); % freq domain channel estimation
end