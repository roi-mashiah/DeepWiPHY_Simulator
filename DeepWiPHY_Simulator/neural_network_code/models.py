import torch.nn as nn
from torch import relu
from torch.nn.functional import conv1d, log_softmax
import torch
from torch.fft import fftshift
from enum import Enum
from configuration import Configuration


class ModelType(Enum):
    delaySpreadEst = 1
    autoEncoder = 2
    smootherEst = 3
    channelConvNetEst = 4
    channelClassifier = 5


class ModelUtils:
    @staticmethod
    def get_model_from_config(configuration: Configuration, reference_seq):
        model = ModelType[configuration.model_type]
        criterion = nn.MSELoss()
        nll_criterion = nn.NLLLoss()
        nn_architecture = configuration.node_counts
        if model == ModelType.autoEncoder:
            return AutoEncoderModel(criterion, nn_architecture)
        elif model == ModelType.channelConvNetEst:
            return ConvChannelEstimationModel(criterion, nn_architecture)
        elif model == ModelType.delaySpreadEst:
            return DelaySpreadEstimationModel(criterion, nn_architecture, reference_seq)
        elif model == ModelType.smootherEst:
            return SmootherEstimationModel(criterion, nn_architecture, reference_seq)
        elif model == ModelType.channelClassifier:
            return ChannelClassifierModel(nll_criterion, nn_architecture)
        else:
            raise TypeError(f"Unexpected model name - {configuration.model_type}, unknown")

    @staticmethod
    def get_layer_type(layer_name, values: dict):
        if "conv" in layer_name:
            return torch.nn.Conv1d(
                values["input_channels"],
                values["output_channels"],
                values["kernel_size"],
                stride=values.get("stride", 1),
                padding=values.get("padding", 0)
            )
        elif "fc" in layer_name:
            return torch.nn.Linear(values["input_dim"], values["output_dim"])
        elif "bn" in layer_name:
            return torch.nn.BatchNorm1d(values["num_features"])
        elif "aFunc" in layer_name:
            return eval(f"torch.{values}")
        elif "pool" in layer_name:
            return torch.nn.MaxPool1d(values["kernel_size"], values["stride"])
        elif "dropout" in layer_name:
            return torch.nn.Dropout(values["p"])
        elif "cTrans" in layer_name:
            return torch.nn.ConvTranspose1d(
                values["input_channels"],
                values["output_channels"],
                values["kernel_size"],
                values["stride"],
                values.get("padding", 0),
                values.get("output_padding", 0)
            )
        elif "view" in layer_name:
            return lambda x: x.view(*values["args"])
        elif "reshape" in layer_name:
            return lambda x: x.reshape(*values["args"])


class ChannelClassifierModel(nn.Module):
    def __init__(self, criterion, node_counts):
        super().__init__()
        self.criterion = criterion
        self.node_counts = node_counts
        for layer_name, params in node_counts.items():
            layer = ModelUtils.get_layer_type(layer_name, params)
            setattr(self, layer_name, layer)
        self.double()

    def forward(self, x):
        for layer_name in self.node_counts.keys():
            layer = getattr(self, layer_name)
            x = layer(x)
        return log_softmax(x, dim=1)


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
            layer = ModelUtils.get_layer_type(layer_name, params)
            setattr(self, layer_name, layer)
        self.double()

    def forward(self, x):
        for layer_name in self.node_counts.keys():
            layer = getattr(self, layer_name)
            x = layer(x)
        return x.view(-1)


class ConvChannelEstimationModel(nn.Module):
    def __init__(self, criterion, node_counts):
        super().__init__()
        self.criterion = criterion
        self.node_counts = node_counts
        for layer_name, params in node_counts.items():
            layer = ModelUtils.get_layer_type(layer_name, params)
            setattr(self, layer_name, layer)
        self.double()

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
            layer = ModelUtils.get_layer_type(layer_name, params)
            setattr(self, layer_name, layer)
        self.double()

    def forward(self, x):
        for layer_name in self.node_counts.keys():
            layer = getattr(self, layer_name)
            x = layer(x)
        return x * self.scale_factor


class SmootherEstimationModel(nn.Module):
    def __init__(self, criterion, node_counts, ref_sequence):
        super().__init__()
        self.criterion = criterion
        self.node_counts = node_counts
        self.sequence = ref_sequence
        self.scale_factor = nn.Parameter(torch.tensor(10.0))
        for layer_name, params in node_counts.items():
            layer = ModelUtils.get_layer_type(layer_name, params)
            setattr(self, layer_name, layer)
        self.double()

    def forward(self, x):
        h_ls = fftshift(x.cpu() / self.sequence)
        for layer_name in self.node_counts.keys():
            layer = getattr(self, layer_name)
            x = layer(x)
        # x is the smoothing filter of size (batch_size, 2, M)
        filters = torch.mean(x, 0).view(1, 1, x.shape[-1])  # output channels = 1, input channels = 1, kernel size
        h = torch.zeros_like(h_ls)
        batch_size, in_channels, iW = h_ls.shape
        for i in range(in_channels):
            input_signal = h_ls[:, i, :].view(batch_size, 1, iW)
            filtered = conv1d(input_signal, filters.cpu(), padding="same")
            h[:, i, :] = filtered.view(-1, batch_size, iW)
        return h * self.scale_factor.cpu()
