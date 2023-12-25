import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from channel_est_model import ChannelEstimationModel
from helpers.data_utils import concat_all_csv_files, reshape_vectors_to_matrices
from enum import Enum


class ChannelType(Enum):
    A = "ch_A"
    B = "ch_B"
    C = "ch_B"
    ALL = None


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


if __name__ == '__main__':
    manual_seed = 41
    mu = 0.1
    training_iterations = 1500
    group_size = 96
    test_perc = 0.2
    data_path = r"..\data"
    ch_type = ChannelType.C.value
    input_data = concat_all_csv_files(data_path, group_size, ch_type)
    g_1 = input_data[input_data['group'] == 1]
    x_train, x_test, y_train, y_test = setup_train_test_data(g_1, group_size, test_perc, manual_seed)

    torch.manual_seed(manual_seed)
    ch_est_model, criterion, losses = train_ch_est_model(x_train, y_train, group_size, lr=mu,
                                                         epochs=training_iterations)

    plot_loss(training_iterations, losses, "Training")

    loss = evaluate_model(x_test, y_test, ch_est_model, criterion)
    print(f"Mean Loss: {loss}")
