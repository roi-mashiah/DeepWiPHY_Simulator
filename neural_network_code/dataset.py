import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from configuration import Configuration


class WiPhyDataset(Dataset):
    def __init__(self, configuration: Configuration, is_train=True, transform=None, target_transform=None):
        self.is_train = is_train
        self.filtered_data = pd.DataFrame()
        self.all_packets = pd.DataFrame()
        self.packets = pd.DataFrame()
        self.configuration = configuration
        self.transform = transform
        self.target_transform = target_transform
        self._create_filtered_dataset()
        test_samples = self.filtered_data.sample(frac=configuration.test_perc,
                                                 random_state=configuration.manual_seed)
        if is_train:
            self.packets = self.filtered_data[~self.filtered_data.index.isin(
                test_samples.index)]
        else:
            self.packets = test_samples
        self.packets.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.packets)

    def __getitem__(self, idx):
        packet_path = self.packets.loc[idx, "path"]
        packet_info = str.join("_", [f"{k}_{v}" for k, v in
                                     self.packets.loc[idx, ["snr", "ch", "packet"]].to_dict().items()])
        packet = pd.read_csv(packet_path)
        packet["group"] = (packet.index // self.configuration.group_size) + 1
        he_ltf = torch.FloatTensor(np.concatenate((packet["HE-LTF_real"].values,
                                                   packet["HE-LTF_imag"].values)))
        channel = torch.FloatTensor(np.concatenate((packet[packet["group"] == 1]["channel_taps_real"].values,
                                                    packet[packet["group"] == 1]["channel_taps_imag"].values)))
        if self.transform:
            he_ltf = self.transform(he_ltf)
        if self.target_transform:
            channel = self.target_transform(channel)
        return he_ltf, channel, packet_info

    @staticmethod
    def parse_filename(filename):
        # snipped from ChatGPT
        # Define a regular expression pattern to extract information from the filename
        pattern = re.compile(
            r'snr_(\d+(\.\d+)?)_ch_([A-Za-z]*)_packet_(\d+).csv')
        # Use the pattern to match against the filename
        match = pattern.search(filename)
        if not match:
            return
        # Extract matched groups and create the dictionary
        snr = float(match.group(1))
        ch = match.group(3)
        packet_number = int(match.group(4))
        result_dict = {
            'snr': snr,
            'ch': ch,
            'path': filename,
            'packet': packet_number
        }
        return result_dict

    def _filter_data(self) -> pd.DataFrame:
        model_mask = self.all_packets['ch'] == self.configuration.ch_type \
            if self.configuration.ch_type else pd.Series([True] * self.all_packets.shape[0])
        snr_mask = self.all_packets['snr'] >= self.configuration.snr_value \
            if self.configuration.snr_value else pd.Series([True] * self.all_packets.shape[0])
        final_filter = model_mask & snr_mask
        return self.all_packets.loc[final_filter, :]

    def _create_filtered_dataset(self):
        all_packets = [self.parse_filename(os.path.join(self.configuration.data_path, f))
                       for f in os.listdir(self.configuration.data_path)
                       if f.endswith('.csv')]
        self.all_packets = pd.DataFrame(
            all_packets, index=range(len(all_packets)))
        filtered_data = self._filter_data()
        filtered_data.reset_index(drop=True, inplace=True)
        self.filtered_data = filtered_data


if __name__ == '__main__':
    pass
