import json
import os
import re
import numpy as np
from numpy.fft import fftshift
import pandas as pd
import torch
from torch.utils.data import Dataset
from configuration import Configuration
from utils import ModelType


class WiPhyDataset(Dataset):
    def __init__(
            self,
            configuration: Configuration,
            is_train=True,
            transform=None,
            target_transform=None,
    ):
        self.ref_seq = self.load_ref_sequence()
        self.model_type = ModelType[configuration.model_type]
        self.is_train = is_train
        self.filtered_data = pd.DataFrame()
        self.all_packets = pd.DataFrame()
        self.packets = pd.DataFrame()
        self.configuration = configuration
        self.transform = transform
        self.target_transform = target_transform
        self._create_filtered_dataset()
        test_samples = self.filtered_data.sample(
            frac=configuration.test_perc, random_state=configuration.manual_seed
        )
        if is_train:
            self.packets = self.filtered_data[
                ~self.filtered_data.index.isin(test_samples.index)
            ]
        else:
            self.packets = test_samples
        self.packets.reset_index(drop=True, inplace=True)

    def __len__(self):
        return len(self.packets)

    def __getitem__(self, idx):
        idx = idx.detach().numpy()
        packet_path = self.packets.loc[idx, "path"]
        packet_info = self.packets.loc[idx, ["snr", "ch", "packet"]].to_dict()
        with open(packet_path, "r") as file_reader:
            packet = json.load(file_reader)

        packet["group"] = (np.arange(242) // self.configuration.group_size) + 1
        group_mask = packet["group"] == 1
        packet["channel_est_real"] = fftshift(packet["channel_est_real"])
        packet["channel_est_imag"] = fftshift(packet["channel_est_imag"])
        he_ltf = torch.from_numpy(
            np.vstack(
                (
                    np.array(packet["HE_LTF_real"], dtype=np.float32),
                    np.array(packet["HE_LTF_imag"], dtype=np.float32)
                )
            )
        )
        # he_ltf initial size is (2, 242)
        channel = torch.from_numpy(
            np.vstack(
                (
                    np.array(packet["channel_taps_real"], dtype=np.float32)[group_mask],
                    np.array(packet["channel_taps_imag"], dtype=np.float32)[group_mask],
                )
            )
        )
        channel_est = torch.from_numpy(
            np.vstack(
                (
                    packet["channel_est_real"][group_mask],
                    packet["channel_est_imag"][group_mask],
                )
            )
        )
        if self.transform:
            he_ltf = self.transform(he_ltf)
        if self.target_transform:
            channel = self.target_transform(channel)
        if self.model_type == ModelType.delaySpreadEst:
            # label is the calculated delay spread
            return he_ltf, torch.FloatTensor(packet["rms_ds"]), channel_est, packet_info
        elif self.model_type == ModelType.autoEncoder:
            # input to the NN is the least squares estimation
            return channel_est, channel, channel_est, packet_info
        else:
            return he_ltf, channel, channel_est, packet_info

    @staticmethod
    def load_ref_sequence():
        sequence_df = pd.read_csv(
            "/home/tauproj3/Documents/DeepWiPHY_Simulator/HE_LTF_SEQ.csv"
        )
        return sequence_df["seq"].values

    @staticmethod
    def parse_filename(filename):
        # snipped from ChatGPT
        # Define a regular expression pattern to extract information from the filename
        pattern = re.compile(r"snr_(\d+(\.\d+)?)_ch_([A-Za-z]*)_packet_(\d+).json")
        # Use the pattern to match against the filename
        match = pattern.search(filename)
        if not match:
            return
        # Extract matched groups and create the dictionary
        snr = float(match.group(1))
        ch = match.group(3)
        packet_number = int(match.group(4))
        result_dict = {"snr": snr, "ch": ch, "path": filename, "packet": packet_number}
        return result_dict

    def _filter_data(self) -> pd.DataFrame:
        model_mask = (
            self.all_packets["ch"] == self.configuration.ch_type
            if self.configuration.ch_type
            else pd.Series([True] * self.all_packets.shape[0])
        )
        snr_mask = (
            self.all_packets["snr"] >= self.configuration.snr_value
            if self.configuration.snr_value
            else pd.Series([True] * self.all_packets.shape[0])
        )
        final_filter = model_mask & snr_mask
        return self.all_packets.loc[final_filter, :]

    def _create_filtered_dataset(self):
        all_packets = [
            self.parse_filename(os.path.join(self.configuration.data_path, f))
            for f in os.listdir(self.configuration.data_path)
            if f.endswith(".json")
        ]
        self.all_packets = pd.DataFrame(all_packets, index=range(len(all_packets)))
        filtered_data = self._filter_data()
        filtered_data.reset_index(drop=True, inplace=True)
        self.filtered_data = filtered_data


if __name__ == "__main__":
    pass
