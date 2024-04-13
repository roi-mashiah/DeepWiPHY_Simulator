import numpy as np
import torch.nn as nn
from torch import relu


class SmootherEstimationModel(nn.Module):
    def __init__(self, criterion, input_dim=484, output_dim=484, node_counts=None):
        super().__init__()
        self.criterion = criterion
        

    def forward(self, x):
        
        h_tilde_hat = x[:242] + 1j*x[243:483]
        h_sm = x[484:]
        h_hat = np.convolve(h_tilde_hat, h_sm, "same")
        x = np.concatenate((np.real(h_hat), np.imag(h_hat)))
        return x
