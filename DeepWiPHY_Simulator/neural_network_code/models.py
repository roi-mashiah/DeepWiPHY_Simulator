import torch.nn as nn
from torch import relu
import torch


def get_layer_type(layer_name, values):
    if "conv" in layer_name:
        return torch.nn.Conv1d(
            values["input_channels"], values["output_channels"], values["kernel_size"]
        )
    elif "fc" in layer_name:
        return torch.nn.Linear(values["input_dim"], values["output_dim"])
    elif "bn" in layer_name:
        return torch.nn.BatchNorm1d(values["num_features"])
    elif "aFunc" in layer_name:
        return torch.relu if values == "relu" else None
    elif "pool" in layer_name:
        return torch.nn.MaxPool1d(values["kernel_size"], values["stride"])
    elif "cTrans" in layer_name:
        return torch.nn.ConvTranspose1d(
            values["input_channels"],
            values["output_channels"],
            values["kernel_size"],
            values["stride"],
        )
    elif "view" in layer_name:
        return lambda x: x.view(*values["args"])


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
    def __init__(self, criterion, node_counts, ref_sequence):
        super().__init__()
        self.reference_sequence = ref_sequence
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


class AutoEncoderModel(nn.Module):
    def __init__(self, criterion, node_counts):
        super().__init__()
        self.criterion = criterion
        self.node_counts = node_counts
        self.scale_factor = nn.Parameter(torch.tensor(1.0))
        for layer_name, params in node_counts.items():
            layer = get_layer_type(layer_name, params)
            setattr(self, layer_name, layer)

    def forward(self, x):
        for layer_name in self.node_counts.keys():
            layer = getattr(self, layer_name)
            x = layer(x)
        return x * self.scale_factor


class SmootherEstimationModel(nn.Module):
    def __init__(self, criterion, node_counts, ref_sequence):
        super().__init__()
        self.criterion = criterion
        self.sequence = ref_sequence
        for layer_name, params in node_counts.items():
            layer = get_layer_type(layer_name, params)
            setattr(self, layer_name, layer)

    def forward(self, x):
        h_ls = x / self.sequence
        for layer_name in self.node_counts.keys():
            layer = getattr(self, layer_name)
            x = layer(x)
        # x is the smoothing filter of size (batch_size, 2, M)
        conv_out = nn.Conv1d(2, 2, kernel_size=x.shape[1], padding="same", groups=2)
        weights = torch.mean(x, 0)  # size (2, M)
        place_holder = torch.zeros_like(conv_out.weight)
        place_holder[:, 1, :] = weights
        conv_out.weight = nn.Parameter(place_holder, requires_grad=False)
        h = conv_out(h_ls)
        return h
