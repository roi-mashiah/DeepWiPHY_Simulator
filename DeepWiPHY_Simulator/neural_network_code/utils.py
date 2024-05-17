import json
import logging
import colorlog
from configuration import Configuration, asdict
import torch
from torch.utils.data import Subset
from sklearn.preprocessing import StandardScaler
import numpy as np
from enum import Enum
from models import *


class ModelType(Enum):
    delaySpreadEst = 1
    autoEncoder = 2
    smootherEst = 3
    channelConvNetEst = 4


def create_train_test_subsets(full_dataset, subset_size, test_perc):
    # Create subset
    subset_indices = torch.randperm(len(full_dataset))[:subset_size]
    test_indices = subset_indices[: int(test_perc * subset_size)]
    train_indices = subset_indices[int(test_perc * subset_size) :]
    train_wiphy_datasubset = Subset(full_dataset, train_indices)
    test_wiphy_datasubset = Subset(full_dataset, test_indices)
    return train_wiphy_datasubset, test_wiphy_datasubset


def get_model_from_config(configuration: Configuration):
    model = ModelType(configuration.model_type)
    if model == ModelType.autoEncoder:
        return AutoEncoderModel(criterion=nn.MSELoss())
    elif model == ModelType.channelConvNetEst:
        return ConvChannelEstimationModel(criterion=nn.MSELoss())
    elif model == ModelType.delaySpreadEst:
        return DelaySpreadEstimationModel(nn.MSELoss(), configuration.node_counts)
    elif model == ModelType.smootherEst:
        return SmootherEstimationModel(criterion=nn.MSELoss())
    else:
        raise TypeError(f"Unexpected model name - {configuration.model_type}, unknown")


def get_layer_type(layer_name, values):
    if "conv" in layer_name:
        return torch.nn.Conv1d(
            values["in_channels"], values["out_channels"], values["kernel_size"]
        )
    elif "fc" in layer_name:
        return torch.nn.Linear(values["input_dim"], values["output_dim"])
    elif "bn" in layer_name:
        return torch.nn.BatchNorm1d(values["num_features"])
    elif "aFunc" in layer_name:
        return torch.relu if values == "relu" else None


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
    metadata_dict["nn_loss"] = list(range(np.shape(gt.numpy())[0]))
    metadata_dict["bl_loss"] = list(range(np.shape(gt.numpy())[0]))
    metadata_dict["snr"] = metadata_dict["snr"].numpy()
    metadata_dict["packet"] = metadata_dict["packet"].numpy()
    for i in range(np.shape(gt.numpy())[0]):
        curr_gt = gt.numpy()[i, :]
        curr_est = estimation.numpy()[i, :]
        curr_bl = baseline_channel_est.numpy()[i, :]
        gt_abs = np.sqrt(
            np.sum(np.power(curr_gt.reshape([2, curr_est.shape[0] // 2]), 2), 0)
        )
        estimation_abs = np.sqrt(
            np.sum(np.power(curr_est.reshape([2, curr_gt.shape[0] // 2]), 2), 0)
        )
        baseline_estimation_abs = np.sqrt(
            np.sum(np.power(curr_bl.reshape([2, curr_bl.shape[0] // 2]), 2), 0)
        )
        mse = np.round(np.mean((gt_abs - estimation_abs) ** 2), 5)
        bl_mse = np.round(np.mean((gt_abs - baseline_estimation_abs) ** 2), 5)
        metadata_dict["nn_loss"][i] = mse
        metadata_dict["bl_loss"][i] = bl_mse
    return metadata_dict
