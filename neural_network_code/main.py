import os
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import utils
from channel_est_model import ChannelEstimationModel
from configuration import Configuration
from dataset import WiPhyDataset
from helpers import performance_plots as perf_plots

writer = SummaryWriter(f"runs\\{int(datetime.now().timestamp())}", flush_secs=5)


def training_loop(data_loader, model, optimizer):
    model.train()
    losses = []
    for batch, (X, y, _, _) in enumerate(data_loader):
        y_predicted = model(X)  # get predicted results
        loss = model.criterion(y_predicted, y)  # predicted values vs y_train
        losses.append(loss.detach().numpy())
        writer.add_scalar(f"Training Loss - {config_name}", loss, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    log.info("Done training")
    return losses


def testing_loop(dataloader, model, plot=False, save=True):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0
    results_dfs = []

    with torch.no_grad():
        for X, y, baseline_ch_est, packet_info in dataloader:
            pred = model(X)
            curr_loss = model.criterion(pred, y).item()
            test_loss += curr_loss
            metadata_dict = utils.calculate_performance(y, pred, baseline_ch_est, packet_info)
            results_dfs.append(pd.DataFrame(metadata_dict, index=metadata_dict['packet']))
    if plot:
        perf_plots.plot_performance(pd.concat(results_dfs), (writer, config_name))
    if save:
        pd.concat(results_dfs).to_csv(
            fr"C:\Projects\DeepWiPHY\DeepWiPHY_Simulator\helpers\ch_est_results_{config_name.split('.')[0]}.csv")

    test_loss /= num_batches
    return test_loss


def train_test_ch_est_model(train_data_loader, test_data_loader, configuration: Configuration):
    model = ChannelEstimationModel(criterion=nn.MSELoss(),
                                   output_dim=configuration.group_size * 2,
                                   node_counts=configuration.node_counts)
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration.mu, weight_decay=1e-5)
    train_loss_over_epochs = []
    test_loss_over_epochs = []
    with tqdm(total=config.training_iterations, desc='Current losses (train,test)') as pbar:
        for t in range(config.training_iterations):
            curr_tr_loss = training_loop(train_data_loader, model, optimizer)
            curr_test_loss = testing_loop(test_data_loader, model, plot=True)
            pbar.set_description(
                f"Current losses (train,test): {sum(curr_tr_loss) / len(curr_tr_loss)},{curr_test_loss}")
            train_loss_over_epochs.append(curr_tr_loss)
            test_loss_over_epochs.append(curr_test_loss)
            pbar.update(1)
    return model, train_loss_over_epochs, test_loss_over_epochs


if __name__ == '__main__':
    log = utils.init_logger()
    log.info("Starting session...")
    config_dir = r"C:\Projects\DeepWiPHY\DeepWiPHY_Simulator\neural_network_code\configs"
    configs = [os.path.join(config_dir, f) for f in os.listdir(config_dir)]
    for config_path in configs:
        config = utils.load_config(config_path, log)
        config_name = os.path.split(config_path)[-1]
        train_dataset = WiPhyDataset(config, is_train=True)
        test_dataset = WiPhyDataset(config, is_train=False)
        log.info(
            f"Train data size after filter (number of scenarios): {len(train_dataset)}")
        log.info(
            f"Test data size after filter (number of scenarios): {len(test_dataset)}")
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
        torch.manual_seed(config.manual_seed)
        log.info("Start training...")
        ch_est_model, train_loss, test_loss = train_test_ch_est_model(
            train_loader, test_loader, config)
    writer.close()
