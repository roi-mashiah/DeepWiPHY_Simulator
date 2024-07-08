function [] = plot_channels(channel_response_t, baseline_est, nn_est, ind)
h = fftshift(fft(channel_response_t,256));
h = h(ind);
figure;
subplot(211);
plot(abs(h))
hold on
plot(abs(baseline_est));
grid on 
legend(["gt","baseline"])

subplot(212);
plot(abs(h))
hold on
plot(abs(nn_est));
grid on 
legend(["gt","nn"])

figure
subplot(211)
plot(real(h))
hold on
plot(real(nn_est));
plot(real(baseline_est))
grid on 
legend(["gt","nn", "baseline"])
title("Real")

subplot(212);
plot(imag(h))
hold on
plot(imag(nn_est))
plot(imag(baseline_est));
grid on 
legend(["gt","nn","baseline"])
title("Imag")
end

