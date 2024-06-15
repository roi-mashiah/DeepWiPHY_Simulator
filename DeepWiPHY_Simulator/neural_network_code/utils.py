import json
import logging
import colorlog
from configuration import Configuration, asdict
import torch
from torch.utils.data import Subset
from sklearn.preprocessing import StandardScaler
import numpy as np
from enum import Enum
from models import ConvChannelEstimationModel, AutoEncoderModel, SmootherEstimationModel, DelaySpreadEstimationModel


class ModelType(Enum):
    delaySpreadEst = 1
    autoEncoder = 2
    smootherEst = 3
    channelConvNetEst = 4


def create_train_test_subsets(full_dataset, subset_size, test_perc):
    # Create subset
    subset_indices = torch.randperm(len(full_dataset))[:subset_size]
    test_indices = subset_indices[: int(test_perc * subset_size)]
    train_indices = subset_indices[int(test_perc * subset_size):]
    train_wiphy_datasubset = Subset(full_dataset, train_indices)
    test_wiphy_datasubset = Subset(full_dataset, test_indices)
    return train_wiphy_datasubset, test_wiphy_datasubset


def get_model_from_config(configuration: Configuration, reference_seq):
    model = ModelType[configuration.model_type]
    criterion = torch.nn.MSELoss()
    nn_architecture = configuration.node_counts
    if model == ModelType.autoEncoder:
        return AutoEncoderModel(criterion, nn_architecture)
    elif model == ModelType.channelConvNetEst:
        return ConvChannelEstimationModel(criterion, nn_architecture)
    elif model == ModelType.delaySpreadEst:
        return DelaySpreadEstimationModel(criterion, nn_architecture, reference_seq)
    elif model == ModelType.smootherEst:
        return SmootherEstimationModel(criterion, nn_architecture, reference_seq)
    else:
        raise TypeError(f"Unexpected model name - {configuration.model_type}, unknown")


def scale_vector(v):
    # created scaler
    scaler = StandardScaler()
    # fit scaler on training dataset
    scaler.fit(v)
    # transform training dataset
    return scaler.transform(v), scaler


def rmse(output, target):
    loss = torch.mean((output - target) ** 2) / (torch.linalg.norm(target) ** 2)
    return loss


def load_config(p: str, log) -> Configuration:
    with open(p, "r") as r:
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


def calculate_performance(gt, estimation, baseline_channel_est, metadata_dict):
    batch_size = gt.shape[0]
    metadata_dict["nn_loss"] = list(range(batch_size))
    metadata_dict["bl_loss"] = list(range(batch_size))
    metadata_dict["snr"] = metadata_dict["snr"].numpy()
    metadata_dict["packet"] = metadata_dict["packet"].numpy()
    for i in range(batch_size):
        curr_gt = gt.numpy()[i, :]
        curr_est = estimation.numpy()[i, :]
        curr_bl = baseline_channel_est.numpy()[i, :]
        gt_abs = calculate_absolute_value(curr_gt)
        estimation_abs = calculate_absolute_value(curr_est)
        baseline_estimation_abs = calculate_absolute_value(curr_bl)
        metadata_dict["nn_loss"][i] = calculate_mse(gt_abs, estimation_abs)
        metadata_dict["bl_loss"][i] = calculate_mse(gt_abs, baseline_estimation_abs)
    return metadata_dict


def calculate_absolute_value(vector):
    return np.sqrt(np.sum(np.power(vector, 2), 0))


def calculate_mse(x, y):
    return np.round(np.mean((x - y) ** 2), 5)


def calculate_ds_performance(gt_cir, he_ltf, rms_ds, baseline_channel_est, metadata_dict):
    batch_size = gt_cir.shape[0]
    metadata_dict["nn_loss"] = list(range(batch_size))
    metadata_dict["bl_loss"] = list(range(batch_size))
    metadata_dict["snr"] = metadata_dict["snr"].numpy()
    metadata_dict["packet"] = metadata_dict["packet"].numpy()
    for i in range(batch_size):
        curr_gt = gt_cir.numpy()[i, :]
        curr_rms_ds_est = rms_ds.numpy()[i, :]
        curr_bl = baseline_channel_est.numpy()[i, :]
        gt_abs = calculate_absolute_value(curr_gt)
        baseline_estimation_abs = calculate_absolute_value(curr_bl)
        # use rms ds estimation to decide smoother
        estimation_abs = smoothing_filter(he_ltf, curr_rms_ds_est)
        metadata_dict["nn_loss"][i] = calculate_mse(gt_abs, estimation_abs)
        metadata_dict["bl_loss"][i] = calculate_mse(gt_abs, baseline_estimation_abs)
    return metadata_dict


def smoothing_filter(he_ltf, rms_ds, snr):
    delta_f = 20e6 / 256  # sub-carrier spacing
    m = 5  # number of taps
    m_range = torch.arange(-(m - 1) / 2, (m - 1) / 2)
    sigma_squared = 1 / (10 ** (snr / 20))  # noise power assuming signal power is normalized
    r_hh = torch.sinc(m_range * delta_f * rms_ds)

