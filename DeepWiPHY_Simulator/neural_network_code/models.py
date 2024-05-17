import torch.nn as nn
from torch import relu
from torch.nn.functional import conv1d
import torch
from utils import get_layer_type


class ChannelEstimationModel(nn.Module):
    def __init__(self, criterion, input_dim=726, output_dim=18, node_counts=None):
        super().__init__()
        self.output_scaler = None
        self.input_scaler = None
        self.criterion = criterion
        self.out = nn.Linear(node_counts[-1], output_dim)
        if node_counts is None:
            node_counts = [50, 50, 50]
        self.node_counts = node_counts
        for i, neuron_count in enumerate(node_counts):
            if i == 0:
                setattr(self, f"fc{i}", nn.Linear(input_dim, neuron_count))
            else:
                setattr(self, f"fc{i}", nn.Linear(node_counts[i - 1], neuron_count))

    def forward(self, x):
        for i in range(len(self.node_counts)):
            fc = getattr(self, f"fc{i}")
            batch_norm = nn.LazyBatchNorm1d()
            x = relu(fc(batch_norm(x)))
        x = self.out(x)
        return x


class DelaySpreadEstimationModel(nn.Module):
    def __init__(self, criterion, node_counts):
        super().__init__()
        self.criterion = criterion
        self.node_counts = node_counts
        for layer_name, params in node_counts.items():
            layer = get_layer_type(layer_name, params)
            setattr(self, layer_name, layer)

    def forward(self, x):
        for layer_name in self.node_counts.keys():
            layer = getattr(self, layer_name)
            x = layer(x)
        return x


class ConvChannelEstimationModel(nn.Module):
    def __init__(self, criterion, input_dim=726, output_dim=18, node_counts=None):
        super().__init__()
        self.output_scaler = None
        self.input_scaler = None
        self.criterion = criterion
        self.out = nn.Linear(node_counts[-1], output_dim)
        if node_counts is None:
            node_counts = [50, 50, 50]
        self.node_counts = node_counts
        for i, neuron_count in enumerate(node_counts):
            if i == 0:
                setattr(self, f"fc{i}", nn.Linear(input_dim, neuron_count))
            else:
                setattr(self, f"fc{i}", nn.Linear(node_counts[i - 1], neuron_count))

    def forward(self, x):
        for i in range(len(self.node_counts)):
            fc = getattr(self, f"fc{i}")
            batch_norm = nn.LazyBatchNorm1d()
            x = relu(fc(batch_norm(x)))
        x = self.out(x)
        return x


class AutoEncoderModel(nn.Module):
    def __init__(self, criterion, input_dim=726, output_dim=18, node_counts=None):
        super().__init__()
        self.output_scaler = None
        self.input_scaler = None
        self.criterion = criterion
        self.out = nn.Linear(node_counts[-1], output_dim)
        if node_counts is None:
            node_counts = [50, 50, 50]
        self.node_counts = node_counts
        for i, neuron_count in enumerate(node_counts):
            if i == 0:
                setattr(self, f"fc{i}", nn.Linear(input_dim, neuron_count))
            else:
                setattr(self, f"fc{i}", nn.Linear(node_counts[i - 1], neuron_count))

    def forward(self, x):
        for i in range(len(self.node_counts)):
            fc = getattr(self, f"fc{i}")
            batch_norm = nn.LazyBatchNorm1d()
            x = relu(fc(batch_norm(x)))
        x = self.out(x)
        return x


class SmootherEstimationModel(nn.Module):
    def __init__(self, criterion, node_counts, ref_sequence):
        super().__init__()
        self.criterion = criterion
        self.sequence = ref_sequence
        for layer_name, params in node_counts.items():
            layer = get_layer_type(layer_name, params)
            setattr(self, layer_name, layer)

    def forward(self, x, h_ls):
        for layer_name in self.node_counts.keys():
            layer = getattr(self, layer_name)
            x = layer(x)
        # x is the smoothing filter of size (batch_size, 2, M)
        conv_out = nn.Conv1d(2, 2, kernel_size=x.shape[1], padding="same", groups=2)
        weights = torch.mean(x, 0)  # size (2, M)
        conv_out.weight = nn.Parameter(weights, requires_grad=False)
        h = conv_out(h_ls)
        return h
