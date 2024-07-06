function [] = create_traintest_data(dataPath)
allResults = dir(fullfile(dataPath,"**/*.mat"));
output_dir = fullfile(fileparts(dataPath),"jsonfiles");
if ~exist(output_dir, "dir")
    mkdir(output_dir);
end

delete(gcp("nocreate"));
parpool('local',6);

for f=1:length(allResults)
    file_name_mat = allResults(f).name;
    full_filename = fullfile(allResults(f).folder, file_name_mat);      
    res = load(full_filename);
    activeFFTInd = res.scenario.tx.ofdmInfo.ActiveFFTIndices;
    ltf_samples = res.scenario.rx.HE_LTF;
    channel_vec = res.scenario.gt.channel_taps_gt;
    rms_ds_vec = res.scenario.gt.rms_delay_spread;
    channel_est_matlab = res.scenario.rx.channel_est;
    parfor i=1:length(res.scenario.rx.HE_LTF)
        heLtf = ltf_samples{i};
        gt_channel_t = channel_vec{i};
        gt_channel_f = fftshift(fft(gt_channel_t));
        gt_rms_ds = rms_ds_vec{i};
        channelTapsGt = gt_channel_f(activeFFTInd);
        channelEstimation = channel_est_matlab{i};
        if ~isempty(heLtf)
            filename_json = replace(file_name_mat,".mat",strcat("_packet_",num2str(i),".json"));            
            outfilename = fullfile(output_dir, filename_json);
            data_to_save = [real(channelTapsGt) imag(channelTapsGt)  real(heLtf) imag(heLtf) real(channelEstimation) imag(channelEstimation)];
            variable_names = {
                'channel_taps_real', ...
                'channel_taps_imag', ...
                'HE_LTF_real', ...
                'HE_LTF_imag', ...
                'channel_est_real', ...
                'channel_est_imag'};
            data_struct = struct();
            for j = 1:numel(variable_names)
                data_struct.(variable_names{j}) = data_to_save(:, j);
            end
            data_struct.rms_ds = gt_rms_ds;
            data_struct.channel_taps_t_real = real(gt_channel_t);
            data_struct.channel_taps_t_imag = imag(gt_channel_t);

            json_str = jsonencode(data_struct);
            fid = fopen(outfilename,"w");
            fwrite(fid,json_str);
            fclose(fid);           
        end
    end
end
end

