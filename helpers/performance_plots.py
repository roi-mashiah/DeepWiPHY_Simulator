import matplotlib.pyplot as plt
import numpy as np
from torch import FloatTensor


def plot_loss(epochs, losses, title_str):
    plt.plot(range(epochs), losses)
    plt.grid()
    plt.xlabel("Number of training iterations")
    plt.ylabel("Loss")
    plt.title(f"Loss as a function of epochs {title_str}")
    plt.show()


def plot_performance(results_df, args):
    writer, config_name = args
    for snr, snr_sub_df in results_df.groupby('snr'):
        i = 0
        f = plt.figure(figsize=(10, 10))
        for channel, ch_sub_df in snr_sub_df.groupby('ch'):
            title = f"Channel {channel}, N={ch_sub_df.shape[0]}"
            plt.subplot(2, 3, i + 1, title=title)
            plt.ecdf(ch_sub_df.nn_loss, label="neural net")
            plt.ecdf(ch_sub_df.bl_loss, label="baseline")
            plt.grid()
            plt.legend()
            i += 1
        writer.add_figure(f"Perf Plots - SNR {int(snr)}, Configuration {config_name}",
                          figure=f,
                          close=True)


def plot_channel_reconstruction(gt: FloatTensor, estimation: FloatTensor, baseline_channel_est, metadata, writer):
    for i in range(np.shape(gt.numpy())[0]):
        curr_gt = gt.numpy()[i, :]
        curr_est = estimation.numpy()[i, :]
        curr_bl = baseline_channel_est.numpy()[i, :]
        gt_abs = np.sqrt(np.sum(np.power(curr_gt.reshape([2, curr_est.shape[0] // 2]), 2), 0))
        estimation_abs = np.sqrt(np.sum(np.power(curr_est.reshape([2, curr_gt.shape[0] // 2]), 2), 0))
        baseline_estimation_abs = np.sqrt(np.sum(np.power(curr_bl.reshape([2, curr_bl.shape[0] // 2]), 2), 0))
        mse = np.round(np.mean((gt_abs - estimation_abs) ** 2), 2)
        bl_mse = np.round(np.mean((gt_abs - baseline_estimation_abs) ** 2), 2)
        # f = plt.figure(i)
        # plt.stem(gt_abs, linefmt='g', markerfmt='go', label='truth')
        # plt.stem(estimation_abs, linefmt='r', markerfmt='rd', label='estimation')
        # plt.stem(baseline_estimation_abs, linefmt='m', markerfmt='mv', label='classic method')
        # plt.grid()
        # plt.legend()
        # plt.title(f"{metadata[i]} MSE(NN,BL): {mse},{bl_mse}")
        # writer.add_figure("Perf Plots", figure=f, global_step=i, close=True)
        writer.add_scalars("Performance - Classic VS NN", {"NeuralNet": mse, "Classic": bl_mse}, i)
        writer.flush()
    return


if __name__ == '__main__':
    pass
