import os
from os import path

from torch import FloatTensor
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


def plot_loss(epochs, losses, title_str):
    plt.plot(range(epochs), losses)
    plt.grid()
    plt.xlabel("Number of training iterations")
    plt.ylabel("Loss")
    plt.title(f"Loss as a function of epochs {title_str}")
    plt.show()


def plot_channel_reconstruction(gt: FloatTensor, estimation: FloatTensor, metadata, writer):
    for i in range(np.shape(gt.numpy())[0]):
        curr_gt = gt.numpy()[i, :]
        curr_est = estimation.numpy()[i, :]
        mse = np.mean((curr_gt - curr_est) ** 2)
        gt_abs = np.sqrt(np.sum(np.power(curr_gt.reshape([2, curr_est.shape[0] // 2]), 2), 0))
        estimation_abs = np.sqrt(np.sum(np.power(curr_est.reshape([2, curr_gt.shape[0] // 2]), 2), 0))
        f = plt.figure()
        plt.stem(gt_abs, linefmt='g', markerfmt='go', label='truth')
        plt.stem(estimation_abs, linefmt='r', markerfmt='rd', label='estimation')
        plt.grid()
        plt.legend()
        plt.title(f"{metadata[i]}_mse_{mse}")
        writer.add_figure("Perf Plots", f, i)
        plt.close(f)
    return


if __name__ == '__main__':
    pass
