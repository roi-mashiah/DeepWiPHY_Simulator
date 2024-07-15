import torch
import torch.nn as nn
from torch.jit import trace
from models import ConvChannelEstimationModel, ChannelClassifierModel, DelaySpreadEstimationModel
import utils
import os

result_dir = "/home/tauproj3/Documents/DeepWiPHY_Simulator/DeepWiPHY_Simulator/results"
traced_dir = os.path.join(result_dir,"traced")
if not os.path.exists(traced_dir):
    os.mkdir(traced_dir)
configs_path = "/home/tauproj3/Documents/DeepWiPHY_Simulator/DeepWiPHY_Simulator/neural_network_code/configs/hybrid_model_configs"
configs = [os.path.join(configs_path, c) for c in os.listdir(configs_path)]
log = utils.init_logger()
for config_path in configs:
    criterion = nn.MSELoss()
    config = utils.load_config(config_path, log)
    model_name = os.path.split(config_path)[-1].replace(".json", ".pt")
    state = torch.load(os.path.join(result_dir, model_name))
    model = ConvChannelEstimationModel(criterion, config.node_counts) if "channel_est" in config_path else ChannelClassifierModel(criterion, config.node_counts)
    # model = DelaySpreadEstimationModel(criterion, config.node_counts)
    model.load_state_dict(state)
    model.eval()
    example_input = torch.rand(2, 242, dtype=torch.double)
    module = trace(model.forward, example_input)
    traced_filename = os.path.join(traced_dir, model_name)
    module.save(traced_filename)
