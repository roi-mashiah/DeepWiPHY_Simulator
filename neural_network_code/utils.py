import json
import logging
import colorlog
from configuration import Configuration, asdict
import torch
from sklearn.preprocessing import StandardScaler


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
