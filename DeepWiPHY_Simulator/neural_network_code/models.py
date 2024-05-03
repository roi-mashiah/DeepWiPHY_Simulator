import torch.nn as nn
from torch import relu
import torch


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
    def __init__(self, criterion, ref_sequence, kernel_sizes=[6, 6]):
        super().__init__()
        self.criterion = criterion
        self.sequence = ref_sequence

        self.conv1 = nn.Conv1d(2, 2, kernel_sizes[0])
        self.fc1 = nn.Linear(in_features=8, out_features=256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu1 = relu
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = relu

    def forward(self, x):
        h_ls = (x[:242] + 1j * x[243:]) / self.sequence
        x = self.fc1(x)
        h_hat = torch.conv1d(h_ls, x)
        x = torch.concatenate((torch.real(h_hat), torch.imag(h_hat)))
        return x
