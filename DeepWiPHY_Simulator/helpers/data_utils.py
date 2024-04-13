from concurrent.futures import ThreadPoolExecutor
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
        'path': filename,
        'packet': packet_number
    }
    return result_dict


def filter_data(all_runs: pd.DataFrame, model_type: str, snr: int, part: str) -> pd.DataFrame:
    model_mask = all_runs['ch'] == model_type \
        if model_type else pd.Series([True] * all_runs.shape[0])
    snr_mask = all_runs['snr'] >= snr \
        if snr else pd.Series([True] * all_runs.shape[0])
    part_mask = all_runs['part'] == part \
        if part else pd.Series([True] * all_runs.shape[0])
    final_filter = model_mask & snr_mask & part_mask
    return all_runs.loc[final_filter, :]


def read_one_data_file(data_path_obj):
    index, scenario_row, config = data_path_obj
    scenario = pd.read_csv(scenario_row['path'])
    scenario[list(scenario_row.keys())] = scenario_row
    scenario['sc_index'] = index
    scenario['group'] = (scenario.index // config.group_size) + 1
    return scenario


def concat_all_csv_files(configuration: Configuration):
    data_path = configuration.data_path
    all_runs = [parse_filename(os.path.join(data_path, f))
                for f in os.listdir(data_path)
                if f.endswith('.csv')]
    all_runs_df = pd.DataFrame(all_runs, index=range(len(all_runs)))
    filtered_data = filter_data(all_runs_df, configuration.ch_type, configuration.snr_value, configuration.part)
    filtered_data.reset_index(drop=True, inplace=True)
    list_of_all_data = [(i, scenario_row, configuration) for i, scenario_row in filtered_data.iterrows()]
    with ThreadPoolExecutor(max_workers=4) as ex:
        mapping = ex.map(read_one_data_file, list_of_all_data)
    all_data_list = list(mapping)
    return pd.concat(all_data_list, ignore_index=True)


def reshape_vectors_to_matrices(x, y, group_size):
    x_r = np.reshape(x, [len(x) // 242, 242])
    y_r = np.reshape(y, [len(y) // group_size, group_size])
    return x_r, y_r


if __name__ == '__main__':
    pass
