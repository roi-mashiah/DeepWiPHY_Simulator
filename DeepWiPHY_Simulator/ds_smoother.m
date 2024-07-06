function [equalizer] = ds_smoother(heLTFDemodDataSCs,SNR, rms_ds)
%% Params
sigma_a = std(heLTFDemodDataSCs);                       % signal std, should be 1 (assuming UnitAveragePower above was set to 1)
Nh = 5;                                                 % number of channel taps - to produce correlation between the LTF SCs. Currently set to delta, so there is no correlation, delay spready is inf (or very big), and smoother should be delta!
Nc = 5;                                                 % Smoother taps
Nrx = 1;                                                % number of rx antennas
%% Generate Signals
sigma_w = sigma_a*10^(-SNR/20);                         % noise std
Nm = max(Nh,Nc);                                        % max between number of channel taps and number of equalizer taps, in order to build Ryy matrix and take its center correctly
Raa = sigma_a^2;

%% Analytical Solution
tau = rms_ds;
fsc = 312.5e3;                                           % subcarrier spacing Hz
n = -(Nm-1):(Nm-1);                                      % although the smoother will be only of length Nc, we double it length in order to allow building Ryy later on, which uses (Nc-1) samples before and after the main sample
h_sym= sinc(n*fsc*tau).';                                % assuming PDP is rect(tau)

Nh_sym = 2*Nm-1;                                        % physical channel is of lenth Nm, so there are Nm-1 taps after the cetner tap. When extending the channel to have as many negative taps as there in the positive side, we get Nh_sym = 2*Nm-1;
h_sym_center = Nm;                                      % the center tap of the symmetric channel
h_flp_full = flipud(h_sym);                             % flip the symmetric channel, such that the tap at [0] is kept as mirroring point, which maintain its indexing of zero
desired_l_ind = floor(Nc/2);                            % the l'th delayed symbol, is the desired one. if we set l=0, we only have post-curer taps. however, we want a symmetric filter, with energy both pre and post curser.  if Nc= even number of eqaulizer taps - the main tap will be located at Nc/2+1 (so there are more pre-curser taps than post-curser). e.g., Nc=2 -> we expect to get a delta filter that is [0 1]. for Nc=3: [0 1 0], etc...
inds_h_flp_full = -desired_l_ind:(-desired_l_ind+Nc-1); % required pyx inds to obtain the desired_l_ind symbol (around the h_sym_center)
h_flp = h_flp_full(h_sym_center + inds_h_flp_full,:);     % after flipping, we take the flipped channel at the above indiced, and shift them by the center tap
Pyx_an = (h_flp * Raa);
Pyx_an = Pyx_an(:);

Ryy_an_block = cell(Nrx,Nrx);                           % place holder for block matrices Ry1y1,...,Ry2y2
for n_col = 1:Nrx
    for n_row = 1:Nrx
        ryy = h_sym;                                            % for smoother - Rhh is used in both Pyx and Pyy
        ryy_len = Nh_sym;                                   % length of the convolution results between symetric h and its flipped version
        ryy_center = (ryy_len+1)/2;                             % center tap of the convolution result between symctric h and its flipped version
        ryy_col = ryy(ryy_center:-1:ryy_center-(Nc-1));         % NOTE: here we take ryy in reverse order, as the formula for ryy is with [-m] !!! The fist column in Ryy should hold positive inds of ryy vector, starting from its center
        ryy_raw = ryy(ryy_center:1:ryy_center+(Nc-1));          % first row holds negative inds in ryy vector, starting from it center
        Ryy_an_block{n_row,n_col} = toeplitz(ryy_col, ryy_raw) ;
    end
end

Ryy_an = Ryy_an_block{1,1} + sigma_w^2 * eye(Nc); %TBD: There was a bug here!! no noise was added!!!
equalizer = Ryy_an\Pyx_an;
end

