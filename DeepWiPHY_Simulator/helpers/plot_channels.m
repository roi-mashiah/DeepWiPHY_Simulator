function [] = plot_channels(channel_gt, baseline_est, nn_est)
figure;
subplot(211);
plot(abs(channel_gt))
hold on
plot(abs(baseline_est));
grid on 
legend(["gt","baseline"])

subplot(212);
plot(abs(channel_gt))
hold on
plot(abs(nn_est));
grid on 
legend(["gt","nn"])

figure
subplot(211)
plot(real(channel_gt))
hold on
plot(real(nn_est));
plot(real(baseline_est))
grid on 
legend(["gt","nn", "baseline"])
title("Real")

subplot(212);
plot(imag(channel_gt))
hold on
plot(imag(nn_est))
plot(imag(baseline_est));
grid on 
legend(["gt","nn","baseline"])
title("Imag")
end

