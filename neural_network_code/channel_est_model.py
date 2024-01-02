import torch.nn as nn
from torch import relu, sigmoid


class ChannelEstimationModel(nn.Module):
    def __init__(self, group_size, criterion, h1=20, h2=20, h3=20):
        super().__init__()
        self.criterion = criterion
        self.fc1 = nn.Linear(group_size, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.fc3 = nn.Linear(h2, h3)
        self.bn3 = nn.BatchNorm1d(h3)
        self.out = nn.Linear(h3, group_size)

    def forward(self, x):
        x = sigmoid(self.bn1(self.fc1(x)))
        x = sigmoid(self.bn2(self.fc2(x)))
        x = relu(self.bn3(self.fc3(x)))
        x = self.out(x)
        return x
