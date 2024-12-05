import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import TRAIN_INCLUDE_SUSPECTED_PATH, TEST_INCLUDE_SUSPECTED_PATH, OUTPUTS_INCLUDE_SUSPECTED_DIR,\
    TRAIN_EXCLUDE_SUSPECTED_PATH, TEST_EXCLUDE_SUSPECTED_PATH, OUTPUTS_EXCLUDE_SUSPECTED_DIR
from classifiers_train_and_test import fit_on_train_data, test_on_test_data


def convert_routes_to_one_day_data(df):
    lived_columns = sorted([col for col in df.columns if 'Lived' in col], key=lambda x: int(x.split()[1]))
    temperature_columns = sorted([col for col in df.columns if 'Temp' in col], key=lambda x: int(x.split()[1]))
    salinity_columns = sorted([col for col in df.columns if 'Salinity' in col], key=lambda x: int(x.split()[1]))

    one_day_data = []
    for idx, row in df.iterrows():
        for lived_col, temp_col, salinity_col in zip(lived_columns, temperature_columns, salinity_columns):
            one_day_data.append({
                'temperature': row[temp_col],
                'salinity': row[salinity_col],
                'death': row[lived_col],
            })

    one_day_df = pd.DataFrame(one_day_data)
    for col in one_day_df.columns:
        one_day_df[col] = one_day_df[col].astype(int)

    return one_day_df


def create_one_day_data(train_path, test_path, data_dir):
    train_df = pd.read_csv(train_path)
    one_day_train_df = convert_routes_to_one_day_data(train_df)
    one_day_train_df.to_csv(data_dir / 'train.csv', index=False)

    test_df = pd.read_csv(test_path)
    one_day_test_df = convert_routes_to_one_day_data(test_df)
    one_day_test_df.to_csv(data_dir / 'test.csv', index=False)

    return one_day_train_df, one_day_test_df


def plot_one_day_data(one_day_df, data_dir):
    one_day_df['death_label'] = one_day_df['death'].map({1: 'Dead', 0: 'Alive'})
    sns.scatterplot(x='temperature', y='salinity', hue='death_label', data=one_day_df, alpha=0.7, palette={'Dead': 'red', 'Alive': 'green'})
    plt.xlabel("Temperature (celsius)")
    plt.ylabel("Salinity (ppt)")
    plt.title("One Day Model")
    plt.legend(title="Status")
    plt.grid(alpha=0.3)
    plt.savefig(data_dir / "scatter_plot.png", dpi=300, bbox_inches='tight')


def run_analysis(train_path, test_path, outputs_dir):
    os.makedirs(outputs_dir, exist_ok=True)
    model_data_dir = outputs_dir / 'data'
    os.makedirs(model_data_dir, exist_ok=True)
    model_train_dir = outputs_dir / 'train_outputs'
    os.makedirs(model_train_dir, exist_ok=True)
    model_test_dir = outputs_dir / 'test_outputs'
    os.makedirs(model_test_dir, exist_ok=True)

    one_day_train_df, one_day_test_df = create_one_day_data(train_path, test_path, model_data_dir)
    one_day_full_df = pd.concat([one_day_train_df, one_day_test_df], axis=0)
    plot_one_day_data(one_day_full_df, model_data_dir)

    fit_on_train_data(one_day_train_df.drop(columns=['death']), one_day_train_df['death'],
                      model_train_dir, -1, 'ONE_DAY_MODEL_TRAIN')
    test_on_test_data(model_train_dir / 'best_model.pkl', one_day_test_df.drop(columns=['death']),
                      one_day_test_df['death'], model_test_dir, 'ONE_DAY_MODEL_TEST')


def main():
    run_analysis(TRAIN_INCLUDE_SUSPECTED_PATH, TEST_INCLUDE_SUSPECTED_PATH, OUTPUTS_INCLUDE_SUSPECTED_DIR / 'one_day_model')
    run_analysis(TRAIN_EXCLUDE_SUSPECTED_PATH, TEST_EXCLUDE_SUSPECTED_PATH, OUTPUTS_EXCLUDE_SUSPECTED_DIR / 'one_day_model')


if __name__ == '__main__':
    main()
