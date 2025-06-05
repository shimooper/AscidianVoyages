import os
import pandas as pd
import logging
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt


METRICS_NAME_TO_AVERAGE_METRIC_NAME = {
    'val_mcc': 'mean_test_mcc',
    'val_auprc': 'mean_test_auprc',
    'val_f1': 'mean_test_f1',
    'train_mcc': 'mean_train_mcc',
    'train_auprc': 'mean_train_auprc',
    'train_f1': 'mean_train_f1',
}

DAYS_DESCRIPTIONS = {0: "Current Day", 1: "1 Day Ago", 2: "2 Days Ago", 3: "3 Days Ago"}

SURVIVAL_COLORS = {
    'Alive': '#0072B2',  # blue
    'Dead': '#E69F00',   # orange
}


def variable_equals_value(variable, value):
    if pd.isna(value):
        return pd.isna(variable)

    return pd.notna(variable) and variable == value


def setup_logger(log_file, logger_name, level=logging.INFO):
    if os.path.exists(log_file):
        os.remove(log_file)

    # Create a logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    # Create a formatter and set it for the file handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger


def convert_pascal_to_snake_case(name):
    # Convert PascalCase to snake_case
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()


def get_column_groups_sorted(df):
    lived_columns = sorted([col for col in df.columns if 'Lived' in col], key=lambda x: int(x.split()[1]))
    temperature_columns = sorted([col for col in df.columns if 'Temp' in col], key=lambda x: int(x.split()[1]))
    salinity_columns = sorted([col for col in df.columns if 'Salinity' in col], key=lambda x: int(x.split()[1]))

    return lived_columns, temperature_columns, salinity_columns


def convert_columns_to_int(df, columns_to_skip=None):
    if columns_to_skip is None:
        columns_to_skip = []
    columns_to_convert = [col for col in df.columns if col not in columns_to_skip]
    for col in columns_to_convert:
        df[col] = df[col].astype(int)


def get_lived_columns_to_consider(row, day, number_of_future_days_to_consider_death):
    lived_cols_to_consider = [f'Lived {day + offset}'
                              for offset in range(0, number_of_future_days_to_consider_death + 1)
                              if f'Lived {day + offset}' in row.index and pd.notna(row[f'Lived {day + offset}'])]
    return lived_cols_to_consider


def merge_dicts_average(dict_list):
    sum_dict = defaultdict(float)
    count_dict = defaultdict(int)

    # Sum values and count occurrences
    for d in dict_list:
        for key, value in d.items():
            sum_dict[key] += value
            count_dict[key] += 1

    # Compute the average
    return {METRICS_NAME_TO_AVERAGE_METRIC_NAME[key]: sum_dict[key] / count_dict[key] for key in sum_dict}


def convert_data_to_tensor_for_rnn(X_df, device):
    # Reshape DataFrame into (n_samples, number_of_days, 3)

    n_samples = X_df.shape[0]
    number_of_days = int(len(X_df.columns) / 3)

    # Extract temperature and salinity values
    temperature = X_df[[f"Temperature {i}" for i in range(1, number_of_days + 1)]].values
    salinity = X_df[[f"Salinity {i}" for i in range(1, number_of_days + 1)]].values
    time = X_df[[f"Time {i}" for i in range(1, number_of_days + 1)]].values

    # Stack them to get (n_samples, number_of_days, 3)
    tensor_data = np.stack([temperature, salinity, time], axis=-1)  # Shape (n_samples, number_of_days, 3)

    # Convert to PyTorch tensor
    tensor_data = torch.tensor(tensor_data, dtype=torch.float32, device=device)
    return tensor_data


def plot_models_comparison(results_df, outputs_dir: Path, title):
    all_models_test_results_df_melted = results_df.melt(id_vars='model_name', var_name='metric', value_name='value')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=all_models_test_results_df_melted, x='metric', y='value', hue='model_name', palette='viridis')
    plt.title(title, fontsize=16)
    plt.ylabel('Score', fontsize=14)
    plt.xlabel('Metric', fontsize=14)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0., fontsize=12)
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig(outputs_dir / 'all_models_comparison.png', dpi=600)
    plt.close()

    results_df.set_index('model_name', inplace=True)
    max_indices = results_df.idxmax()
    results_df.loc['best_model'] = max_indices
    results_df.to_csv(outputs_dir / 'all_models_comparison.csv')
