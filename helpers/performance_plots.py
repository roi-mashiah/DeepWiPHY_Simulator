import os
from os import path

import matplotlib.pyplot as plt


def plot_loss(epochs, losses, title_str):
    plt.plot(range(epochs), losses)
    plt.grid()
    plt.xlabel("Number of training iterations")
    plt.ylabel("Loss")
    plt.title(f"Loss as a function of epochs {title_str}")
    plt.show()


def plot_channel_reconstruction(gt, estimation, metadata):
    out_fig_folder = r".\figures"
    if not os.path.exists(out_fig_folder):
        os.makedirs(out_fig_folder)
    for i in range(gt.size()[0]):
        plt.figure()
        plt.stem(gt[i, :estimation.size()[-1]].numpy(), linefmt='g', markerfmt='go', label='truth')
        plt.stem(estimation[i, :].numpy(), linefmt='r', markerfmt='rd', label='estimation')
        plt.grid()
        plt.legend()
        key = float(gt[i, -1])
        title_str = '_'.join(f"{key.upper()}: {value}"
                             for key, value in metadata[key].iloc[0].to_dict().items())
        plt.title(title_str)
        plt.savefig(path.join(out_fig_folder, f"{title_str.replace(': ', '-')}_{i}.png"))


if __name__ == '__main__':
    pass
