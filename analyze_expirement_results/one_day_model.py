import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import TRAIN_PATH, TEST_PATH, OUTPUTS_DIR
from classifiers_train_and_test import fit_on_train_data, test_on_test_data

ONE_DAY_MODEL_DIR = OUTPUTS_DIR / 'one_day_model'
ONE_DAY_MODEL_DATA_DIR = ONE_DAY_MODEL_DIR / 'data'
ONE_DAY_MODEL_TRAIN_OUTPUTS_DIR = ONE_DAY_MODEL_DIR / 'train_outputs'
ONE_DAY_MODEL_TEST_OUTPUTS_DIR = ONE_DAY_MODEL_DIR / 'test_outputs'


def convert_routes_to_one_day_data(df):
    lived_columns = sorted([col for col in df.columns if 'Lived' in col], key=lambda x: int(x.split()[1]))
    temperature_columns = sorted([col for col in df.columns if 'Temp' in col], key=lambda x: int(x.split()[1]))
    salinity_columns = sorted([col for col in df.columns if 'Salinity' in col], key=lambda x: int(x.split()[1]))

    one_day_data = []
    for idx, row in df.iterrows():
        for lived_col, temp_col, salinity_col in zip(lived_columns, temperature_columns, salinity_columns):
            if pd.isna(row[lived_col]) or pd.isna(row[temp_col]) or pd.isna(row[salinity_col]):
                if pd.isna(row[lived_col]) and pd.isna(row[temp_col]) and pd.isna(row[salinity_col]):
                    continue
                else:
                    raise ValueError(f'Row {idx} contains NaN values in some but not all of the columns: '
                                     f'{lived_col}, {temp_col}, {salinity_col}')

            one_day_data.append({
                'temperature': row[temp_col],
                'salinity': row[salinity_col],
                'death': row[lived_col],
            })

    one_day_df = pd.DataFrame(one_day_data)
    for col in one_day_df.columns:
        one_day_df[col] = one_day_df[col].astype(int)

    return one_day_df


def create_one_day_data():
    train_df = pd.read_csv(TRAIN_PATH)
    one_day_train_df = convert_routes_to_one_day_data(train_df)
    one_day_train_df.to_csv(ONE_DAY_MODEL_DATA_DIR / 'train.csv', index=False)

    test_df = pd.read_csv(TEST_PATH)
    one_day_test_df = convert_routes_to_one_day_data(test_df)
    one_day_test_df.to_csv(ONE_DAY_MODEL_DATA_DIR / 'test.csv', index=False)

    return one_day_train_df, one_day_test_df


def plot_one_day_data(one_day_df):
    one_day_df['death_label'] = one_day_df['death'].map({1: 'Dead', 0: 'Alive'})
    sns.scatterplot(x='temperature', y='salinity', hue='death_label', data=one_day_df, alpha=0.7, palette={'Dead': 'red', 'Alive': 'green'})
    plt.xlabel("Temperature (celsius)")
    plt.ylabel("Salinity (ppt)")
    plt.title("One Day Model")
    plt.legend(title="Status")
    plt.grid(alpha=0.3)
    plt.savefig(ONE_DAY_MODEL_DATA_DIR / "scatter_plot.png", dpi=300, bbox_inches='tight')


def main():
    os.makedirs(ONE_DAY_MODEL_DIR, exist_ok=True)
    os.makedirs(ONE_DAY_MODEL_DATA_DIR, exist_ok=True)
    os.makedirs(ONE_DAY_MODEL_TRAIN_OUTPUTS_DIR, exist_ok=True)
    os.makedirs(ONE_DAY_MODEL_TEST_OUTPUTS_DIR, exist_ok=True)

    one_day_train_df, one_day_test_df = create_one_day_data()
    one_day_full_df = pd.concat([one_day_train_df, one_day_test_df], axis=0)
    plot_one_day_data(one_day_full_df)

    fit_on_train_data(one_day_train_df[['temperature', 'salinity']], one_day_train_df['death'],
                      ONE_DAY_MODEL_TRAIN_OUTPUTS_DIR, -1, 'ONE_DAY_MODEL_TRAIN')
    test_on_test_data(ONE_DAY_MODEL_TRAIN_OUTPUTS_DIR / 'best_model.pkl', one_day_test_df[['temperature', 'salinity']],
                      one_day_test_df['death'], ONE_DAY_MODEL_TEST_OUTPUTS_DIR, 'ONE_DAY_MODEL_TEST')


if __name__ == '__main__':
    main()
