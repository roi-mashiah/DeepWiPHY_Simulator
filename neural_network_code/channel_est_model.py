import torch.nn as nn
from torch import relu


class ChannelEstimationModel(nn.Module):
    def __init__(self, criterion, input_dim=484, output_dim=18, node_counts=None):
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
            x = relu(fc(x))
        x = self.out(x)
        return x
