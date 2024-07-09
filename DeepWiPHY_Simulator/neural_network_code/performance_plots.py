import os

import matplotlib.pyplot as plt
import pandas as pd
from torch import FloatTensor
from os import path
from decimal import Decimal
from utils import *


def plot_loss_curves(epochs, train_losses, test_losses, title_str, writer):
    f = plt.figure(figsize=(15, 12))
    if "ds" in title_str:
        plt.semilogy(range(epochs), train_losses)
        plt.semilogy(range(epochs), test_losses)
    else:
        plt.plot(range(epochs), train_losses)
        plt.plot(range(epochs), test_losses)
    plt.grid()
    plt.xlabel("Number of training iterations")
    plt.ylabel("Loss")
    plt.title(f"Loss as a function of epochs {title_str}")
    plt.legend(["Training", "Validation"])
    writer.add_figure(
        title_str,
        figure=f,
        close=True,
    )


def get_ccdf(data: np.ndarray, N=10000):
    # % This function plots the CCDF of the columns of data
    ccdfs = []
    for column_index in range(data.shape[-1]):
        h, x = np.histogram(data[:, column_index], N)
        my_pdf = h / data.shape[0]
        my_cdf = np.cumsum(my_pdf)
        my_ccdf = 1 - my_cdf
        ccdfs.append({"ccdf": my_ccdf, "x": 10 * np.log10(x[1:])})
        # semilogy(x,my_ccdf)
        # addFigNames('CCDF','alpha','P(x>alpha)',1,legendCell)
    return ccdfs if len(ccdfs) > 1 else ccdfs[0]


def plot_performance(results_df, args):
    writer, config_name, f = args
    for snr, snr_sub_df in results_df.groupby("snr"):
        i = 0
        f = plt.figure(figsize=(15, 12))
        for channel, ch_sub_df in snr_sub_df.groupby("ch"):
            nn_ccdf = get_ccdf(
                (ch_sub_df.nn_loss / ch_sub_df.bl_loss).values.reshape(
                    ch_sub_df.shape[0], 1
                )
            )
            change_index = np.argwhere(nn_ccdf["x"] > 0)[0]
            losing_probability = nn_ccdf["ccdf"][change_index]
            title = f"Channel {channel} N={ch_sub_df.shape[0]} Pr(BL<NN) = {round(losing_probability[0], 3)}"
            plt.subplot(2, 3, i + 1, title=title)
            plt.semilogy(nn_ccdf["x"], nn_ccdf["ccdf"], label="DB(NN loss/BL Loss)")
            plt.ylim([10e-4, 1])
            plt.grid()
            plt.legend()
            i += 1
        writer.add_figure(
            f"Perf Plots - SNR {int(snr)}, Configuration {config_name}",
            figure=f,
            close=True,
        )
        # f.suptitle(f"SNR: {snr}")
        # f.savefig(f"performance_figs\Configuration_M_SNR_{int(snr)}.png")
    plt.show()


def plot_performance_all(results_df):
    for snr, snr_sub_df in results_df.groupby("snr"):
        i = 0
        f = plt.figure(figsize=(15, 12))
        for channel, ch_sub_df in snr_sub_df.groupby("ch"):
            plt.subplot(2, 3, i + 1, title=f"Channel {channel}")
            for config_name, config_df in ch_sub_df.groupby("config_name"):
                nn_ccdf = get_ccdf(
                    (config_df.nn_loss / config_df.bl_loss).values.reshape(
                        config_df.shape[0], 1
                    )
                )
                change_index = np.argwhere(nn_ccdf["x"] > 0)
                if len(change_index) == 0:
                    # we always win
                    losing_probability = [0]
                else:
                    change_index = change_index[0]
                    losing_probability = nn_ccdf["ccdf"][change_index]
                free_text = f"{config_name}\nN={config_df.shape[0]}\nPr(BL<NN)={round(losing_probability[0], 3)}"
                plt.semilogy(nn_ccdf["x"], nn_ccdf["ccdf"], label=free_text)
                plt.ylim([10e-4, 1])
            plt.grid()
            plt.legend()
            i += 1
        f.suptitle(f"SNR: {snr}")
        f.savefig(f"all_results_snr_{int(snr)}.png")
    plt.show()


def plot_channel_reconstruction(
        gt: FloatTensor, estimation: FloatTensor, baseline_channel_est, metadata, writer, config_name
):
    for i in range(min(np.shape(gt.numpy())[0], 11)):
        curr_gt = gt.numpy()[i, :]
        curr_est = estimation.numpy()[i, :]
        curr_bl = baseline_channel_est.numpy()[i, :]
        gt_abs = calculate_absolute_value(curr_gt)
        estimation_abs = calculate_absolute_value(curr_est)
        baseline_estimation_abs = calculate_absolute_value(curr_bl)
        nn_mse = np.mean(np.power(estimation_abs - gt_abs, 2))
        bl_mse = np.mean(np.power(baseline_estimation_abs - gt_abs, 2))
        f, (ax1, ax2) = plt.subplots(2, 1)
        ax1.plot(gt_abs, '*-', label='truth')
        ax1.plot(estimation_abs, '.--', label='neural net')
        ax1.set_ylim(0, 1.5)
        ax1.grid()
        ax1.legend()
        ax2.plot(gt_abs, '*-', label='truth')
        ax2.plot(baseline_estimation_abs, '.--', label='classic method')
        ax2.set_ylim(0, 1.5)
        ax2.grid()
        ax2.legend()
        title_str = f"{config_name} Channel: {metadata['ch'][i]}, SNR: {metadata['snr'][i]}"
        f.suptitle("%s MSE(NN,BL): %.2E,%.2E" % (title_str, Decimal(nn_mse), Decimal(bl_mse)))
        writer.add_figure("Perf Plots - {}".format(config_name), figure=f, global_step=i, close=True)
        plt.close(f)
        writer.flush()
    return


if __name__ == "__main__":
    base_results_dir = "/home/tauproj3/Documents/DeepWiPHY_Simulator/DeepWiPHY_Simulator/results"
    postfix = "_results.csv"
    best_results = [f.replace(postfix,"") for f in os.listdir(base_results_dir) if "channel_est" in f and postfix in f]
    dfs = []
    for res_name in best_results:
        res_path = path.join(base_results_dir, res_name + postfix)
        res = pd.read_csv(res_path)
        res["config_name"] = res_name
        dfs.append(res)
    all_results = pd.concat(dfs, ignore_index=True)
    plot_performance_all(all_results)
    pass
