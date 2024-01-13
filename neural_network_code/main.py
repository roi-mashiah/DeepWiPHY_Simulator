import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import utils
from channel_est_model import ChannelEstimationModel
from configuration import Configuration
from dataset import WiPhyDataset
from helpers import performance_plots as perf_plots


def training_loop(data_loader, model, optimizer):
    model.train()
    losses = []
    for batch, (X, y) in enumerate(data_loader):
        y_predicted = model(X)  # get predicted results
        loss = model.criterion(y_predicted, y)  # predicted values vs y_train
        losses.append(loss.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses


def testing_loop(dataloader, model):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += model.criterion(pred, y).item()

    test_loss /= num_batches
    return test_loss


def train_test_ch_est_model(train_data_loader, test_data_loader, configuration: Configuration):
    model = ChannelEstimationModel(criterion=nn.MSELoss(),
                                   output_dim=configuration.group_size,
                                   node_counts=configuration.node_counts)
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration.mu)
    train_loss_over_epochs = []
    test_loss_over_epochs = []
    with tqdm(total=config.training_iterations, desc='Current losses (train,test)') as pbar:
        for t in range(config.training_iterations):
            curr_tr_loss = training_loop(train_data_loader, model, optimizer)
            curr_test_loss = testing_loop(test_data_loader, model)
            pbar.set_description(f"Current losses (train,test): {sum(curr_tr_loss)/len(curr_tr_loss)},{curr_test_loss}")
            train_loss_over_epochs.append(curr_tr_loss)
            test_loss_over_epochs.append(curr_test_loss)
            pbar.update(1)
    return model, train_loss_over_epochs, test_loss_over_epochs


if __name__ == '__main__':
    log = utils.init_logger()
    log.info("Starting session...")
    config = utils.load_config(r".\config.json", log)
    train_dataset = WiPhyDataset(config, is_train=True)
    test_dataset = WiPhyDataset(config, is_train=False)
    log.info(f"Train data size after filter (number of scenarios): {len(train_dataset)}")
    log.info(f"Test data size after filter (number of scenarios): {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    torch.manual_seed(config.manual_seed)
    log.info("Start training...")
    ch_est_model, train_loss, test_loss = train_test_ch_est_model(train_loader, test_loader, config)

    perf_plots.plot_loss(config.training_iterations, train_loss, "Training")
    perf_plots.plot_loss(config.training_iterations, test_loss, "Testing")

    # loss = evaluate_model(x_test, y_test, input_data, ch_est_model)
    # log.info(f"Mean Loss: {loss}")
