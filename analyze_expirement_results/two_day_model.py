import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import TRAIN_PATH, TEST_PATH, OUTPUTS_DIR
from classifiers_train_and_test import fit_on_train_data, test_on_test_data

TWO_DAY_MODEL_DIR = OUTPUTS_DIR / 'two_day_model'
TWO_DAY_MODEL_DATA_DIR = TWO_DAY_MODEL_DIR / 'data'
TWO_DAY_MODEL_TRAIN_OUTPUTS_DIR = TWO_DAY_MODEL_DIR / 'train_outputs'
TWO_DAY_MODEL_TEST_OUTPUTS_DIR = TWO_DAY_MODEL_DIR / 'test_outputs'


def convert_routes_to_two_day_data(df):
    temperature_columns = sorted([col for col in df.columns if 'Temp' in col], key=lambda x: int(x.split()[1]))

    two_day_data = []
    for index, row in df.iterrows():
        for col in temperature_columns[:-1]:
            col_day = int(col.split(' ')[1])
            if pd.isna(row[f'Temp {col_day + 1}']):  # Row (route) is done
                break
            new_row = {
                'previous day temperature': row[f'Temp {col_day}'],
                'current day temperature': row[f'Temp {col_day + 1}'],
                'previous day salinity': row[f'Salinity {col_day}'],
                'current day salinity': row[f'Salinity {col_day + 1}'],
                'death': row[f'Lived {col_day + 1}']
            }
            two_day_data.append(new_row)

    two_day_df = pd.DataFrame(two_day_data)
    for col in two_day_df.columns:
        two_day_df[col] = two_day_df[col].astype(int)

    return two_day_df


def create_two_day_data():
    train_df = pd.read_csv(TRAIN_PATH)
    two_day_train_df = convert_routes_to_two_day_data(train_df)
    two_day_train_df.to_csv(TWO_DAY_MODEL_DATA_DIR / 'train.csv', index=False)

    test_df = pd.read_csv(TEST_PATH)
    two_day_test_df = convert_routes_to_two_day_data(test_df)
    two_day_test_df.to_csv(TWO_DAY_MODEL_DATA_DIR / 'test.csv', index=False)

    return two_day_train_df, two_day_test_df


def main():
    os.makedirs(TWO_DAY_MODEL_DIR, exist_ok=True)
    os.makedirs(TWO_DAY_MODEL_DATA_DIR, exist_ok=True)
    os.makedirs(TWO_DAY_MODEL_TRAIN_OUTPUTS_DIR, exist_ok=True)
    os.makedirs(TWO_DAY_MODEL_TEST_OUTPUTS_DIR, exist_ok=True)

    two_day_train_df, two_day_test_df = create_two_day_data()

    fit_on_train_data(two_day_train_df.drop(columns=['death']), two_day_train_df['death'],
                      TWO_DAY_MODEL_TRAIN_OUTPUTS_DIR, -1, 'TWO_DAY_MODEL_TRAIN')
    test_on_test_data(TWO_DAY_MODEL_TRAIN_OUTPUTS_DIR / 'best_model.pkl', two_day_test_df.drop(columns=['death']),
                      two_day_test_df['death'], TWO_DAY_MODEL_TEST_OUTPUTS_DIR, 'TWO_DAY_MODEL_TEST')


if __name__ == '__main__':
    main()
