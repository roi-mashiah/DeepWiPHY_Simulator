import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from channel_est_model import ChannelEstimationModel
from helpers.data_utils import concat_all_csv_files, reshape_vectors_to_matrices
import json
import logging
import colorlog
from configuration import Configuration, asdict


def setup_train_test_data(group, group_size, test_size=0.2, seed=41):
    x_1_flat = group['HE-LTF'].values
    y_1_flat = group['channel_taps'].values
    x_1, y_1 = reshape_vectors_to_matrices(x_1_flat, y_1_flat, group_size)
    train_test_data = train_test_split(x_1, y_1, test_size=test_size, random_state=seed)
    return [torch.FloatTensor(v) for v in train_test_data]


def train_ch_est_model(x_train, y_train, group_size, lr, epochs):
    model = ChannelEstimationModel(group_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in tqdm(range(epochs)):
        y_predicted = model.forward(x_train)  # get predicted results
        loss = criterion(y_predicted, y_train)  # predicted values vs y_train
        losses.append(loss.detach().numpy())
        # if i % 10 == 0:
        #     print((f'Epoch: {i} amd loss: {loss}'))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model, criterion, losses


def plot_loss(epochs, losses, title_str):
    plt.plot(range(epochs), losses)
    plt.grid()
    plt.xlabel("Number of training iterations")
    plt.ylabel("Loss")
    plt.title(f"Loss as a function of epochs {title_str}")
    plt.show()


def evaluate_model(x_test, y_test, model, criterion):
    with torch.no_grad():
        y_eval = model.forward(x_test)
        # calculate MSE for each scenario then take the mean - returns a scalar
        loss = criterion(y_eval, y_test)
    return loss


def load_config(p: str) -> Configuration:
    with open(p, 'r') as r:
        config_json = json.load(r)
    config = Configuration.from_dict(config_json)
    log.info("Loaded Configuration")
    for field_name, field_value in asdict(config).items():
        log.info(f"{field_name}: {field_value}")
    return config


def init_logger():
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = colorlog.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = colorlog.ColoredFormatter("%(log_color)s%(levelname)s: %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


if __name__ == '__main__':
    log = init_logger()
    log.info("Starting session...")
    config = load_config(r".\config.json")
    input_data = concat_all_csv_files(config)
    log.info(f"Data size after filter (number of scenarios): {input_data.shape[0] / 242}")
    g_1 = input_data[input_data['group'] == 1]
    x_train, x_test, y_train, y_test = setup_train_test_data(g_1, config.group_size, config.test_perc,
                                                             config.manual_seed)

    torch.manual_seed(config.group_size)
    log.info("Start training...")
    ch_est_model, criterion, losses = train_ch_est_model(x_train, y_train, config.group_size, lr=config.mu,
                                                         epochs=config.training_iterations)

    plot_loss(config.training_iterations, losses, "Training")

    loss = evaluate_model(x_test, y_test, ch_est_model, criterion)
    log.info(f"Mean Loss: {loss}")
