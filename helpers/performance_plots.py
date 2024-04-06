import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import FloatTensor


def plot_loss(epochs, losses, title_str):
    plt.plot(range(epochs), losses)
    plt.grid()
    plt.xlabel("Number of training iterations")
    plt.ylabel("Loss")
    plt.title(f"Loss as a function of epochs {title_str}")
    plt.show()


def get_ccdf(data: np.ndarray, N=10000):
    # % This function plots the CCDF of the columns of data
    ccdfs = []
    for column_index in range(data.shape[-1]):
        h, x = np.histogram(data[:, column_index], N)
        my_pdf = h / data.shape[0]
        my_cdf = np.cumsum(my_pdf)
        my_ccdf = 1 - my_cdf
        ccdfs.append({'ccdf': my_ccdf, 'x': 10 * np.log10(x[1:])})
        # semilogy(x,my_ccdf)
        # addFigNames('CCDF','alpha','P(x>alpha)',1,legendCell)
    return ccdfs if len(ccdfs) > 1 else ccdfs[0]


def plot_performance(results_df, args):
    writer, config_name = args
    for snr, snr_sub_df in results_df.groupby('snr'):
        i = 0
        f = plt.figure(figsize=(15, 12))
        for channel, ch_sub_df in snr_sub_df.groupby('ch'):
            nn_ccdf = get_ccdf((ch_sub_df.nn_loss / ch_sub_df.bl_loss).values.reshape(ch_sub_df.shape[0], 1))
            change_index = np.argwhere(nn_ccdf['x'] > 0)[0]
            losing_probability = nn_ccdf['ccdf'][change_index]
            title = f"Channel {channel} N={ch_sub_df.shape[0]} Pr(BL<NN) = {round(losing_probability[0], 3)}"
            plt.subplot(2, 3, i + 1, title=title)
            plt.semilogy(nn_ccdf['x'], nn_ccdf['ccdf'], label="DB(NN loss/BL Loss)")
            plt.ylim([10e-4, 1])
            plt.grid()
            plt.legend()
            i += 1
        writer.add_figure(f"Perf Plots - SNR {int(snr)}, Configuration {config_name}",
                          figure=f,
                          close=True)
        # f.suptitle(f"SNR: {snr}")
        # f.savefig(f"performance_figs\Configuration_M_SNR_{int(snr)}.png")
    plt.show()


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
    res = pd.read_csv(r"..\helpers\ch_est_results_config_M.csv")
    plot_performance(res, [None, ""])
    pass
