import os
import pandas as pd

from utils import TRAIN_INCLUDE_SUSPECTED_PATH, TEST_INCLUDE_SUSPECTED_PATH, OUTPUTS_INCLUDE_SUSPECTED_DIR,\
    TRAIN_EXCLUDE_SUSPECTED_PATH, TEST_EXCLUDE_SUSPECTED_PATH, OUTPUTS_EXCLUDE_SUSPECTED_DIR
from classifiers_train_and_test import fit_on_train_data, test_on_test_data


def convert_routes_to_two_day_data(df):
    temperature_columns = sorted([col for col in df.columns if 'Temp' in col], key=lambda x: int(x.split()[1]))

    two_day_data = []
    for index, row in df.iterrows():
        for col in temperature_columns[-1:0:-1]:
            col_day = int(col.split(' ')[1])
            if pd.isna(row[f'Temp {col_day}']):
                continue
            new_row = {
                'current day temperature': row[f'Temp {col_day}'],
                'previous day temperature': row[f'Temp {col_day - 1}'],
                'current day salinity': row[f'Salinity {col_day}'],
                'previous day salinity': row[f'Salinity {col_day - 1}'],
                'death': row[f'Lived {col_day}']
            }
            two_day_data.append(new_row)

    two_day_df = pd.DataFrame(two_day_data)
    for col in two_day_df.columns:
        two_day_df[col] = two_day_df[col].astype(int)

    return two_day_df


def create_two_day_data(train_path, test_path, model_data_dir):
    train_df = pd.read_csv(train_path)
    two_day_train_df = convert_routes_to_two_day_data(train_df)
    two_day_train_df.to_csv(model_data_dir / 'train.csv', index=False)

    test_df = pd.read_csv(test_path)
    two_day_test_df = convert_routes_to_two_day_data(test_df)
    two_day_test_df.to_csv(model_data_dir / 'test.csv', index=False)

    return two_day_train_df, two_day_test_df


def run_analysis(train_path, test_path, outputs_dir):
    os.makedirs(outputs_dir, exist_ok=True)
    model_data_dir = outputs_dir / 'data'
    os.makedirs(model_data_dir, exist_ok=True)
    model_train_dir = outputs_dir / 'train_outputs'
    os.makedirs(model_train_dir, exist_ok=True)
    model_test_dir = outputs_dir / 'test_outputs'
    os.makedirs(model_test_dir, exist_ok=True)

    two_day_train_df, two_day_test_df = create_two_day_data(train_path, test_path, model_data_dir)

    fit_on_train_data(two_day_train_df.drop(columns=['death']), two_day_train_df['death'],
                      model_train_dir, -1, 'TWO_DAY_MODEL_TRAIN')
    test_on_test_data(model_train_dir / 'best_model.pkl', two_day_test_df.drop(columns=['death']),
                      two_day_test_df['death'], model_test_dir, 'TWO_DAY_MODEL_TEST')


def main():
    run_analysis(TRAIN_INCLUDE_SUSPECTED_PATH, TEST_INCLUDE_SUSPECTED_PATH, OUTPUTS_INCLUDE_SUSPECTED_DIR / 'two_day_model')
    run_analysis(TRAIN_EXCLUDE_SUSPECTED_PATH, TEST_EXCLUDE_SUSPECTED_PATH, OUTPUTS_EXCLUDE_SUSPECTED_DIR / 'two_day_model')


if __name__ == '__main__':
    main()
