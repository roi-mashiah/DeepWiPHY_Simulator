function rms_ds = calculate_rms_delay_spread(Ts, cir)
% rms delay spread calculation
timeline = (0:(length(cir)-1)).*Ts;
pdp = (abs(cir).^2)./(timeline(end)); % |h(t)|^2 / T
avg_ds = (pdp'*timeline')/sum(pdp);
normalized_t = (timeline - avg_ds).^2;
rms_ds = sqrt((pdp'*normalized_t')/sum(pdp));
end