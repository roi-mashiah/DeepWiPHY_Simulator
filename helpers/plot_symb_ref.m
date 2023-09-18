function [] = plot_symb_ref(ref,eqDataSym)
%PLOT_REFERENCE Summary of this function goes here
%   Detailed explanation goes here
figure;
subplot(211);
plot(1:234,real(ref(:,1)),1:234,real(eqDataSym(:,1)))
title("Real - Est vs GT")
grid

subplot(212);
plot(1:234,imag(ref(:,1)),1:234,imag(eqDataSym(:,1)))
title("Imag - Est vs GT")
grid

mse_avg = 0;
for ii =1:size(eqDataSym,2)
    mse_avg = mse_avg + sum(abs(ref(:,ii) - eqDataSym(:,ii)).^2)/numel(ref(:,1));
end
mse_avg = mse_avg/numel(ref,2)
end

