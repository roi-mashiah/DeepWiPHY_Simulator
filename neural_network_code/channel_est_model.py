import torch.nn as nn
from torch import relu, sigmoid


class ChannelEstimationModel(nn.Module):
    def __init__(self, group_size, h1=60, h2=30, h3=15):
        super().__init__()
        self.fc1 = nn.Linear(group_size, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, group_size)

    def forward(self, x):
        x = sigmoid(self.fc1(x))
        x = sigmoid(self.fc2(x))
        x = relu(self.fc3(x))
        x = self.out(x)
        return x


if __name__ == '__main__':
    pass
