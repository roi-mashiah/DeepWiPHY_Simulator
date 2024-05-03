import os
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

import utils
from models import *
from configuration import Configuration
from dataset import WiPhyDataset
from utils import ModelType
from performance_plots import *


def training_loop(data_loader, model, optimizer):
    model.train()
    losses = 0
    for X, y, _, _ in data_loader:
        y_predicted = model(X)  # get predicted results
        loss = model.criterion(y_predicted, y)  # predicted values vs y_train
        losses += loss.detach().numpy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses /= len(data_loader)
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
            metadata_dict = utils.calculate_performance(
                y, pred, baseline_ch_est, packet_info
            )
            results_dfs.append(
                pd.DataFrame(metadata_dict, index=metadata_dict["packet"])
            )
    if plot:
        plot_performance(pd.concat(results_dfs), (writer, config_name))
    if save:
        pd.concat(results_dfs).to_csv(
            rf"C:\Projects\DeepWiPHY\DeepWiPHY_Simulator\helpers\ch_est_results_{config_name.split('.')[0]}.csv"
        )

    test_loss /= num_batches
    return test_loss


def get_model_from_config(configuration: Configuration):
    model = ModelType(configuration.model_type)
    if model == ModelType.autoEncoder:
        return AutoEncoderModel(criterion=nn.MSELoss())
    elif model == ModelType.channelConvNetEst:
        return ConvChannelEstimationModel(criterion=nn.MSELoss())
    elif model == ModelType.delaySpreadEst:
        return DelaySpreadEstimationModel(criterion=nn.MSELoss())
    elif model == ModelType.smootherEst:
        return SmootherEstimationModel(criterion=nn.MSELoss())
    else:
        raise TypeError(f"Unexpected model name - {configuration.model_type}, unknown")


def train_test_ch_est_model(
    train_data_loader,
    test_data_loader,
    configuration: Configuration,
    reference_sequence,
):
    model = get_model_from_config(configuration)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=configuration.mu, weight_decay=configuration.w_decay
    )
    train_loss_over_epochs = []
    test_loss_over_epochs = []
    for t in range(config.training_iterations):
        curr_tr_loss = training_loop(train_data_loader, model, optimizer)
        curr_test_loss = testing_loop(test_data_loader, model)
        train_loss_over_epochs.append(curr_tr_loss)
        test_loss_over_epochs.append(curr_test_loss)

    return model, train_loss_over_epochs, test_loss_over_epochs


def create_train_test_subsets(full_dataset, subset_size, test_perc):
    # Create subset
    subset_indices = torch.randperm(len(full_dataset))[:subset_size]
    test_indices = subset_indices[: int(test_perc * subset_size)]
    train_indices = subset_indices[int(test_perc * subset_size) :]
    train_wiphy_datasubset = Subset(full_dataset, train_indices)
    test_wiphy_datasubset = Subset(full_dataset, test_indices)
    return train_wiphy_datasubset, test_wiphy_datasubset


if __name__ == "__main__":
    writer = SummaryWriter(f"runs/{int(datetime.now().timestamp())}", flush_secs=5)
    log = utils.init_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}\nStarting session...")
    config_dir = (
        "/home/tauproj3/Documents/DeepWiPHY_Simulator/neural_network_code/configs"
    )
    configs = [os.path.join(config_dir, f) for f in os.listdir(config_dir)]
    sub_size = 50e3
    test_percentage = 0.2
    for config_path in configs:
        config_name = os.path.split(config_path)[-1].replace(".json", "")
        torch.manual_seed(config.manual_seed)
        config = utils.load_config(config_path, log)
        config.test_perc = 1
        wiphy_dataset = WiPhyDataset(config, is_train=False)
        train_dataset, test_dataset = create_train_test_subsets(
            wiphy_dataset, sub_size, test_percentage
        )
        train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
        )
        test_loader = DataLoader(
            test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
        )
        log.info("Start training...")
        ch_est_model, train_loss, test_loss = train_test_ch_est_model(
            train_loader, test_loader, config, wiphy_dataset.ref_seq
        )
        plot_loss_curves(
            config.training_iterations, train_loss, test_loss, config_name, writer
        )
    writer.close()
