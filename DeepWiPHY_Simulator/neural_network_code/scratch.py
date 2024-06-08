import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder: 3 Conv1d layers and 2 MaxPool1d layers
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, stride=2),
            # Output: (batchsize, 16, 121)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Output: (batchsize, 16, 60)
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            # Output: (batchsize, 32, 30)
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Output: (batchsize, 32, 15)
            # nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            # Output: (batchsize, 64, 15)
            # nn.ReLU()
        )

        # Decoder: 3 ConvTranspose1d layers
        self.decoder = nn.Sequential(
            # nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            # Output: (batchsize, 32, 15)
            # nn.ReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=1),
            # Output: (batchsize, 16, 30)
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=2, kernel_size=5, stride=2, padding=1),
            # Output: (batchsize, 2, 60)
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=2, out_channels=2, kernel_size=5, stride=2, padding=1, output_padding=0),
            # Output: (batchsize, 2, 121)
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=2, out_channels=2, kernel_size=9, stride=2, padding=2, output_padding=1)
            # Output: (batchsize, 2, 242)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Create an instance of the autoencoder
autoencoder = Autoencoder()

# Example input tensor with shape (batchsize, 2, 242)
input_tensor = torch.randn(8, 2, 242)  # batchsize = 8

# Forward pass
output = autoencoder(input_tensor)
print(autoencoder)
print(output.shape)  # Should print torch.Size([8, 2, 242])
