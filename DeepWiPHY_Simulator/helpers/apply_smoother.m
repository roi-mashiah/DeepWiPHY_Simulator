function [chanEstNNBased] = apply_smoother(chanEst,packetSNR,rms_ds)
smoother_1 = ds_smoother(packetSNR,rms_ds,1);
smoother_3 = ds_smoother(packetSNR,rms_ds,3);
smoother_5 = ds_smoother(packetSNR,rms_ds,5);

chanEstLeft = chanEst(1:121);
chanEstRight = chanEst(122:end);
chanEstOut = zeros(size(chanEstLeft));

chanEstOut(1) = smoother_1*chanEstLeft(1);
chanEstOut(2) = smoother_3.'*chanEstLeft(1:3);
temp = filter(smoother_5,1,chanEstLeft);
chanEstOut(3:end-2) = temp(5:end);
chanEstOut(end-1) = smoother_3.'*chanEstLeft(end-2:end);
chanEstOut(end) = smoother_1*chanEstLeft(end);

chanEstOutR = zeros(size(chanEstLeft));

chanEstOutR(1) = smoother_1*chanEstRight(1);
chanEstOutR(2) = smoother_3.'*chanEstRight(1:3);
temp = filter(smoother_5,1,chanEstRight);
chanEstOutR(3:end-2) = temp(5:end);
chanEstOutR(end-1) = smoother_3.'*chanEstRight(end-2:end);
chanEstOutR(end) = smoother_1*chanEstRight(end);

chanEstNNBased = [chanEstOut; chanEstOutR];
end