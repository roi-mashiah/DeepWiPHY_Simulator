function [] = plot_channel(scenario)
temp = zeros(256,1);
ofdmInfo = wlanHEOFDMInfo('HE-Data',scenario.tx.HE_config);

temp(ofdmInfo.ActiveFFTIndices) = scenario.rx.channel_est;
figure;
subplot(211)
stem([-128:0 1:127], fftshift(abs(ifft(temp)).^2));
title('Time domain channel response estimation')
ylabel('|h[n]|^2')
xlabel('n')
grid;

subplot(212)
x = zeros(256,1);
x(1) = 1;
stem([-128:0 1:127], fftshift(abs(scenario.tx.tgax_channel(x)).^2))
title('Time domain channel response GT')
ylabel('|h[n]|^2')
xlabel('n')
grid;
end

