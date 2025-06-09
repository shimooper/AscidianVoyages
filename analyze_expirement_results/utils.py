import pandas as pd
import logging
import re
from pathlib import Path
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests


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


def get_logger(log_file_path, logger_name, verbose: bool):
    logger = logging.getLogger(logger_name)
    file_handler = logging.FileHandler(log_file_path, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

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


def plot_models_comparison(results_df, outputs_dir: Path, metric):
    results_filtered_df = results_df[['mean_test_f1', 'mean_test_auprc', 'mean_test_mcc']].copy()
    results_filtered_df.rename(columns={'mean_test_f1': 'F1', 'mean_test_auprc': 'AUPRC', 'mean_test_mcc': 'MCC'}, inplace=True)
    results_filtered_melted_df = results_filtered_df.reset_index().melt(id_vars='model_name', var_name='metric', value_name='score')

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid", context="paper")
    palette = sns.color_palette("Set2", n_colors=len(results_filtered_df.index))

    sns.barplot(data=results_filtered_melted_df, x='metric', y='score', hue='model_name', palette=palette)
    plt.ylabel('Score', fontsize=14)
    plt.xlabel('Metric', fontsize=14)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0., title_fontsize=14, fontsize=12)
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.savefig(outputs_dir / 'models_comparison.png', dpi=600)
    plt.close()

    max_indices = results_filtered_df.idxmax()
    results_filtered_df.loc['best_model'] = max_indices
    results_filtered_df.to_csv(outputs_dir / 'models_comparison.csv')

    if len(results_df.index) == 1:  # There is only one model, no comparison needed
        return

    best_model_name = results_df[f'mean_test_{metric}'].idxmax()
    best_model_fold_scores = results_df.loc[best_model_name, [f'split0_test_{metric}', f'split1_test_{metric}',
                                                              f'split2_test_{metric}', f'split3_test_{metric}',
                                                              f'split4_test_{metric}']].values
    statistics_results = []
    for model in results_df.index:
        if model == best_model_name:
            continue
        other_f1_scores = results_df.loc[
            model, [f'split0_test_{metric}', f'split1_test_{metric}', f'split2_test_{metric}', f'split3_test_{metric}',
                    f'split4_test_{metric}']].values

        # Paired t-test
        t_stat, p_val_ttest = ttest_rel(best_model_fold_scores, other_f1_scores)

        # Wilcoxon signed-rank test
        try:
            stat_wilcox, p_val_wilcox, _ = wilcoxon(best_model_fold_scores, other_f1_scores)
        except ValueError:
            p_val_wilcox = 1.0  # fallback if no difference

        statistics_results.append({
            'model_compared': model,
            'p_val_ttest': p_val_ttest,
            'p_val_wilcox': p_val_wilcox,
            'mean_diff': best_model_fold_scores.mean() - other_f1_scores.mean()
        })

    statistics_results_df = pd.DataFrame(statistics_results)

    # FDR correction for t-test p-values
    reject_ttest, pvals_ttest_fdr, _, _ = multipletests(statistics_results_df['p_val_ttest'], method='fdr_bh')
    statistics_results_df['p_val_ttest_fdr_corrected'] = pvals_ttest_fdr
    statistics_results_df['significant_ttest_fdr'] = reject_ttest

    # FDR correction for Wilcoxon p-values
    reject_wilcox, pvals_wilcox_fdr, _, _ = multipletests(statistics_results_df['p_val_wilcox'], method='fdr_bh')
    statistics_results_df['p_val_wilcox_fdr_corrected'] = pvals_wilcox_fdr
    statistics_results_df['significant_wilcox_fdr'] = reject_wilcox

    statistics_results_df.to_csv(outputs_dir / 'model_comparison_statistics.csv', index=False)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise TypeError("Boolean value expected.")
