import pandas as pd
import os
import numpy as np
import re
from neural_network_code.configuration import Configuration


def parse_filename(filename):
    # snipped from ChatGPT
    # Define a regular expression pattern to extract information from the filename
    pattern = re.compile(r'snr_(\d+(\.\d+)?)_ch_([A-Za-z]*)_packet_(\d+)_(\w+).csv')
    # Use the pattern to match against the filename
    match = pattern.search(filename)
    if not match:
        return
    # Extract matched groups and create the dictionary
    snr = float(match.group(1))
    ch = match.group(3)
    packet_number = int(match.group(4))
    part = match.group(5)
    result_dict = {
        'snr': snr,
        'ch': ch,
        'part': part,
        'path': filename
    }
    return result_dict


def filter_data(all_runs: pd.DataFrame, model_type: str, snr: int, part: str) -> pd.DataFrame:
    model_mask = all_runs['ch'] == model_type \
        if model_type else [True] * all_runs.shape[0]
    snr_mask = all_runs['snr'] >= snr \
        if snr else [True] * all_runs.shape[0]
    part_mask = all_runs['part'] == part \
        if part else [True] * all_runs.shape[0]
    final_filter = model_mask & snr_mask & part_mask
    return all_runs.loc[final_filter, :]


def concat_all_csv_files(configuration: Configuration):
    data_path = configuration.data_path
    all_runs = [parse_filename(os.path.join(data_path, f))
                for f in os.listdir(data_path)
                if f.endswith('.csv')]
    all_runs_df = pd.DataFrame(all_runs, index=range(len(all_runs)))
    filtered_data = filter_data(all_runs_df, configuration.ch_type, configuration.snr_value, configuration.part)
    filtered_data.reset_index(drop=True, inplace=True)
    all_data_list = []
    for i, scenario_row in filtered_data.iterrows():
        scenario = pd.read_csv(scenario_row['path'])
        scenario[list(scenario_row.keys())] = scenario_row
        scenario['sc_index'] = i
        scenario['group'] = (scenario.index // configuration.group_size) + 1
        all_data_list.append(scenario)

    all_data_concatenated = pd.concat(all_data_list, ignore_index=True)
    return all_data_concatenated


def reshape_vectors_to_matrices(x, y, group_size):
    x_r = np.reshape(x, [len(x) // group_size, group_size])
    y_r = np.reshape(y, [len(y) // group_size, group_size])
    return x_r, y_r


if __name__ == '__main__':
    pass
