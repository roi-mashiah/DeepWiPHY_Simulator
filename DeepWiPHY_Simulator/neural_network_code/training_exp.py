import os
from glob import glob
from datetime import datetime

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import WiPhyDataset
from utils import *
from performance_plots import *


def training_loop(data_loader, model, optimizer):
    """
    This function loops on batches (per one EPOCH)
    """
    model.train()
    losses = 0
    for X, y, _, _ in data_loader:
        X = X.to(device)
        y = y.to(device)
        y_predicted = model(X)  # get predicted results
        loss = model.criterion(y_predicted, y)  # predicted values vs y_train
        losses += loss.detach().cpu().numpy()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    losses /= len(data_loader)
    return losses


def validation_loop(dataloader, model, model_type: ModelType, plot=False, save=True):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0
    results_dfs = []

    with torch.no_grad():
        for X, y, baseline_ch_est, packet_info in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            curr_loss = model.criterion(pred, y).item()
            test_loss += curr_loss
            if model_type == ModelType.delaySpreadEst:
                # baseline_ch_est is the gt CIR, X is HE-LTF, y is gt RMS DS
                h_ls = X.cpu() / model.reference_sequence # baseline estimation
                metadata_dict = calculate_ds_performance(baseline_ch_est, pred.cpu(), h_ls, packet_info)
            else:
                metadata_dict = calculate_performance(y.cpu(), pred.cpu(), baseline_ch_est, packet_info)
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


def train_test_ch_est_model(
        train_data_loader,
        test_data_loader,
        configuration: Configuration,
        reference_sequence,
):
    model = get_model_from_config(configuration, reference_sequence)
    model_type = ModelType[configuration.model_type]
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=configuration.mu, weight_decay=configuration.w_decay
    )
    train_loss_over_epochs = []
    test_loss_over_epochs = []
    for t in range(configuration.training_iterations):
        curr_tr_loss = training_loop(train_data_loader, model, optimizer)
        curr_test_loss = validation_loop(test_data_loader, model, model_type)
        train_loss_over_epochs.append(curr_tr_loss)
        test_loss_over_epochs.append(curr_test_loss)

    return model, train_loss_over_epochs, test_loss_over_epochs


def main_loop(config_path):
    config = load_config(config_path, log)
    torch.manual_seed(config.manual_seed)
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


if __name__ == "__main__":
    writer = SummaryWriter(f"runs/{int(datetime.now().timestamp())}", flush_secs=5)
    log = init_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}\nStarting session...")
    config_dir = (
        "/home/tauproj3/Documents/DeepWiPHY_Simulator/neural_network_code/configs"
    )
    configs = [
        f
        for f in glob(f"{config_dir}/**/*.json", recursive=True)
        if not "older" in f and f.endswith(".json")
    ]
    sub_size = int(1e3)
    test_percentage = 0.2
    for config_path in configs:
        config_name = os.path.split(config_path)[-1].replace(".json", "")
        main_loop(config_path)
    writer.close()
