clear all; close all; clc;
raw_a = readtable("..\\data\sc_070124_2340_snr_28_ch_A_packet_672_imag.csv");
raw_b = readtable("..\\data\sc_070124_2345_snr_28_ch_B_packet_672_imag.csv");
raw_c = readtable("..\\data\sc_070124_2351_snr_28_ch_C_packet_672_imag.csv");

figure
subplot(321)
stem(fftshift(raw_a.HE_LTF))
title("HE-LTF A")
subplot(323)
stem(fftshift(raw_b.HE_LTF))
title("HE-LTF B")
subplot(325)
stem(fftshift(raw_c.HE_LTF))
title("HE-LTF C")
subplot(322)
stem(fftshift(raw_a.channel_taps))
title("CH A")
subplot(324)
stem(fftshift(raw_b.channel_taps))
title("CH B")
subplot(326)
stem(fftshift(raw_c.channel_taps))
title("CH C")

he_ltf_a = reshape(raw_a.HE_LTF,[11,22]);
he_ltf_b = reshape(raw_b.HE_LTF,[11,22]);
he_ltf_c = reshape(raw_c.HE_LTF,[11,22]);

figure
imagesc(abs(he_ltf_a).^2)
title("ABS(HE-LTF)^2 A")
colorbar
figure
imagesc(abs(he_ltf_b).^2)
title("ABS(HE-LTF)^2 B")
colorbar
figure
imagesc(abs(he_ltf_c).^2)
title("ABS(HE-LTF)^2 C")
colorbar
