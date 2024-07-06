import os
from glob import glob
from datetime import datetime

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from torch.nn.functional import nll_loss

from dataset import WiPhyDataset
from performance_plots import *
from models import ModelUtils, ModelType


def calc_accuracy(truth, prediction):
    absolute_perc_err = torch.abs(truth - prediction) / truth
    zero_value_mask = absolute_perc_err == torch.inf
    absolute_perc_err[zero_value_mask] = torch.abs(prediction[zero_value_mask])
    mean_abs_perc_err = torch.mean(absolute_perc_err)
    return mean_abs_perc_err


def training_loop(data_loader: DataLoader, model, optimizer):
    """
    This function loops on batches (per one EPOCH)
    """
    model.train()
    train_acc = 0
    losses = 0
    for X, y, _, _ in data_loader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_predicted = model(X).to(device)  # get predicted results
        loss = model.criterion(y_predicted, y)  # predicted values vs y_train
        if model.model_type == ModelType.delaySpreadEst:
            train_acc += calc_accuracy(y, y_predicted).item()
        losses += loss.detach().cpu().numpy()
        loss.backward()
        optimizer.step()
    train_acc /= len(data_loader)
    losses /= len(data_loader)
    return losses, train_acc


def validation_loop(dataloader, model, model_type: ModelType, plot=False, save=True):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0
    results_dfs = []
    test_acc = 0

    with torch.no_grad():
        for i, item in enumerate(dataloader):
            X, y, baseline_ch_est, packet_info = item
            X = X.to(device)
            y = y.to(device)
            pred = model(X).to(device)
            curr_loss = model.criterion(pred, y).item()
            test_loss += curr_loss
            if plot and i == num_batches - 1:
                plot_channel_reconstruction(y.cpu(), pred.cpu(), baseline_ch_est, packet_info, writer, config_name)
            if model_type == ModelType.delaySpreadEst:
                test_acc += calc_accuracy(y, pred).item()
                # baseline_ch_est is the gt CIR, X is HE-LTF, y is gt RMS DS
                h_ls = X.cpu() / model.reference_sequence  # baseline estimation
                metadata_dict = calculate_ds_performance(baseline_ch_est, X.cpu(), pred.cpu(), h_ls, packet_info)
            else:
                metadata_dict = calculate_performance(y.cpu(), pred.cpu(), baseline_ch_est, packet_info)
            results_dfs.append(
                pd.DataFrame(metadata_dict, index=metadata_dict["packet"])
            )
    if save:
        pd.concat(results_dfs).to_csv(
            rf"/home/tauproj3/Documents/DeepWiPHY_Simulator/DeepWiPHY_Simulator/results/{config_name.split('.')[0]}_results.csv"
        )
    test_acc /= num_batches
    test_loss /= num_batches
    return test_loss, test_acc


def train(model, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    train_correct = 0
    for batch_idx, (data, target, _, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data).to(device)
        loss = model.criterion(output, target)
        loss.backward()
        optimizer.step()

        # log performance
        train_loss += nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability [[.2, .4,.1,.3],[],[]]
        train_correct += pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % 100 == 0:
            log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader.dataset)
    train_accuracy = round(100. * train_correct / len(train_loader.dataset), 4)

    return train_loss, train_accuracy


def validation(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, _, _ in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data).to(device)
            test_loss += nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    log.info(
        f'\nTest set: Average loss: {round(test_loss, 4)}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_accuracy}%)\n')

    return test_loss, test_accuracy


def train_test_ch_est_model(
        train_data_loader,
        test_data_loader,
        configuration: Configuration,
        reference_sequence,
):
    model = ModelUtils.get_model_from_config(configuration, reference_sequence)
    model_type = ModelType[configuration.model_type]
    model.model_type = model_type
    model = model.to(device)
    log.info(summary(model, (2, 242)))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=configuration.mu, weight_decay=configuration.w_decay
    )
    scheduler = ExponentialLR(optimizer, 0.9)
    early_stopper = EarlyStopping()
    for t in range(configuration.training_iterations):
        curr_tr_loss, train_acc = train(model, train_data_loader, optimizer,
                                        t + 1) if model_type == ModelType.channelClassifier else training_loop(
            train_data_loader,
            model, optimizer)
        visualize_res = False # if (t + 1) % 40 == 0 or t == configuration.training_iterations - 1 else False
        curr_test_loss, test_acc = validation(model,
                                              test_data_loader) if model_type == ModelType.channelClassifier else validation_loop(
            test_data_loader, model, model_type, plot=visualize_res)

        writer.add_scalars(f"Loss Graph - {config_name}",
                           {
                               "train set": curr_tr_loss,
                               "validation set": curr_test_loss
                           },
                           t)
        if model_type == ModelType.channelClassifier or model_type == ModelType.delaySpreadEst:
            writer.add_scalars(f"Accuracy Graph - {config_name}",
                               {
                                   "train set": train_acc,
                                   "validation set": test_acc
                               },
                               t)
        early_stopper(curr_tr_loss, curr_test_loss)
        if early_stopper.early_stop:
            log.info("reached early stopping...\nexiting training loop")
            break
        previous_lr = scheduler.get_lr()
        scheduler.step()
        log.info(f"LR changed from {previous_lr} to {scheduler.get_lr()}")

    return model


def get_data_loaders(config: Configuration):
    if sub_size == -1:
        train_dataset = WiPhyDataset(config)
        test_dataset = WiPhyDataset(config, is_train=False)
        ref_seq = train_dataset.ref_seq
    else:
        config.test_perc = 1
        wiphy_dataset = WiPhyDataset(config, is_train=False)
        train_dataset, test_dataset = create_train_test_subsets(
            wiphy_dataset, sub_size, test_percentage
        )
        ref_seq = wiphy_dataset.ref_seq
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4
    )
    return train_loader, test_loader, ref_seq


def save_model(model):
    log.info(f"Saving model - {config_name}")
    output_filename = config_name.split('.')[0]
    model_output_path = os.path.join(results_dir, f"{output_filename}.pt")
    torch.save(model.state_dict(), model_output_path)


def inference(config_path):
    config = load_config(config_path, log)
    torch.manual_seed(config.manual_seed)
    state_path = path.join(results_dir, config_name + ".pt")
    train_loader, test_loader, sequence = get_data_loaders(config)
    model = ModelUtils.get_model_from_config(config, sequence)
    model.load_state_dict(torch.load(state_path))
    model = model.to(device)
    model_type = ModelType[config.model_type]
    validation_loop(test_loader, model, model_type, plot=True, save=False)


def main_loop(config_path):
    config = load_config(config_path, log)
    torch.manual_seed(config.manual_seed)
    train_loader, test_loader, sequence = get_data_loaders(config)
    log.info(f"Start training model {config_name}...")
    ch_est_model = train_test_ch_est_model(
        train_loader, test_loader, config, sequence
    )
    save_model(ch_est_model)


if __name__ == "__main__":
    tb_log_dir = f"/home/tauproj3/Documents/DeepWiPHY_Simulator/DeepWiPHY_Simulator/runs/{int(datetime.now().timestamp())}"
    results_dir = rf"/home/tauproj3/Documents/DeepWiPHY_Simulator/DeepWiPHY_Simulator/results"
    writer = SummaryWriter(tb_log_dir, flush_secs=5)
    log = init_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}\nStarting session...")
    config_dir = (
        "/home/tauproj3/Documents/DeepWiPHY_Simulator/DeepWiPHY_Simulator/neural_network_code/configs/delay_spread_configs"
    )
    configs = [
        f
        for f in glob(f"{config_dir}/**/*.json", recursive=True)
        if f.endswith(".json")  # and "channel_est" in f
    ]
    sub_size = int(150e3)
    test_percentage = 0.2
    for config_path in configs:
        config_name = os.path.split(config_path)[-1].replace(".json", "")
        # if config_name + '.pt' in os.listdir(results_dir):
        #     log.info(f"Skipping {config_name}, result exists...")
        #     continue
        try:
            # inference(config_path)
            main_loop(config_path)
        except Exception as ex:
            log.error(f"error: {config_name}\n{ex}")
        finally:
            log.info(f"Finished processing {config_name}...")

    writer.close()
