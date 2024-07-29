import json
import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from configuration import Configuration
from models import ModelType
from enum import Enum


class ChannelType(Enum):
    A = 0
    B = 1
    C = 2
    D = 3
    E = 4
    F = 5


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

    def __repr__(self):
        snr_dict = {}
        for snr, snr_df in self.packets.groupby("snr"):
            snr_dict[int(snr)] = snr_df.shape[0]
        return f"Channel {self.configuration.ch_type}\n{json.dumps(snr_dict, indent=4)}"

    def __len__(self):
        return len(self.packets)

    def __getitem__(self, idx):
        idx = idx.detach().numpy() if type(idx) is not int else idx
        packet_path = self.packets.loc[idx, "path"]
        packet_info = self.packets.loc[idx, ["snr", "ch", "packet"]].to_dict()
        with open(packet_path, "r") as file_reader:
            packet = json.load(file_reader)

        packet["group"] = (np.arange(242) // self.configuration.group_size) + 1
        group_mask = packet["group"] == 1

        complex_ltf = self.complex_cartesian_rep(packet, "HE_LTF", group_mask)
        complex_channel = self.complex_cartesian_rep(packet, "channel_taps", group_mask)
        complex_channel_est = self.complex_cartesian_rep(packet, "channel_est", group_mask)

        he_ltf = self.polar_coord_representation(complex_ltf)
        channel = self.polar_coord_representation(complex_channel)
        channel_est = self.polar_coord_representation(complex_channel_est)

        if self.transform:
            he_ltf = self.transform(he_ltf)
        if self.target_transform:
            channel = self.target_transform(channel)
        if self.model_type == ModelType.delaySpreadEst:
            # label is the calculated delay spread
            return he_ltf.double(), np.double(packet["rms_ds"]), channel, packet_info
        elif self.model_type == ModelType.autoEncoder:
            # input to the NN is the least squares estimation
            return channel_est, channel, channel_est, packet_info
        elif self.model_type == ModelType.channelClassifier:
            # input is the LTF, target is the channel class
            return he_ltf.double(), ChannelType[packet_info['ch']].value, channel_est, packet_info
        else:
            return he_ltf, channel, channel_est, packet_info

    @staticmethod
    def load_ref_sequence():
        sequence_df = pd.read_csv(
            "/home/tauproj3/Documents/DeepWiPHY_Simulator/DeepWiPHY_Simulator/HE_LTF_SEQ.csv"
        )
        return sequence_df["seq"].values

    @staticmethod
    def complex_cartesian_rep(packet, key, mask):
        return np.array(packet[f"{key}_real"], dtype=np.double)[mask] + 1j * \
            np.array(packet[f"{key}_imag"], dtype=np.double)[mask]

    @staticmethod
    def polar_coord_representation(cart):
        return torch.from_numpy(
            np.vstack(
                (
                    np.absolute(cart),
                    np.angle(cart),
                )
            )
        )

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
