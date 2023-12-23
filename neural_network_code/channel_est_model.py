import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from helpers.data_utils import concat_all_csv_files, reshape_vectors_to_matrices
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class ChannelEstimationModel(nn.Module):
    def __init__(self, in_features=18, h1=20, h2=25, h3=15, out_features=18):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, out_features)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x


if __name__ == '__main__':
    pass
