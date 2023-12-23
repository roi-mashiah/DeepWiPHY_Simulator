import pandas as pd
import os
import numpy as np


def concat_all_csv_files(data_path, group_size):
    all_csv_file_paths = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.csv')]
    all_data_list = []
    for file in all_csv_file_paths:
        scenario = pd.read_csv(file)
        scenario['group'] = (scenario.index // group_size) + 1
        all_data_list.append(scenario)

    all_data_concatenated = pd.concat(all_data_list, ignore_index=True)
    return all_data_concatenated


def reshape_vectors_to_matrices(x, y, group_size):
    x_r = np.reshape(x, [len(x) // group_size, group_size])
    y_r = np.reshape(y, [len(y) // group_size, group_size])
    return x_r, y_r


# if __name__ == '__main__':
#     data_path = r"C:\Users\97252\PycharmProjects\CleaningData\data"
#     input_data = concat_all_csv_files(data_path, 18)
#
#     group_1_samples = input_data[input_data['group'] == 1]
#     print(group_1_samples.head(20))
