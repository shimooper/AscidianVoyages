import os
import pandas as pd

from utils import TRAIN_INCLUDE_SUSPECTED_PATH, TEST_INCLUDE_SUSPECTED_PATH, OUTPUTS_INCLUDE_SUSPECTED_DIR,\
    TRAIN_EXCLUDE_SUSPECTED_PATH, TEST_EXCLUDE_SUSPECTED_PATH, OUTPUTS_EXCLUDE_SUSPECTED_DIR
from classifiers_train_and_test import fit_on_train_data, test_on_test_data


def convert_routes_to_one_week_data(df):
    temperature_columns = sorted([col for col in df.columns if 'Temp' in col], key=lambda x: int(x.split()[1]))

    one_week_data = []
    for index, row in df.iterrows():
        for col in temperature_columns[-1:5:-1]:
            col_day = int(col.split(' ')[1])
            if pd.isna(row[f'Temp {col_day}']):
                continue
            week_temperature_column = [f'Temp {col_day - i}' for i in range(7)]
            week_salinity_column = [f'Salinity {col_day - i}' for i in range(7)]
            new_row = {
                'current day temperature': row[f'Temp {col_day}'],
                'previous day temperature': row[f'Temp {col_day - 1}'],
                'current day salinity': row[f'Salinity {col_day}'],
                'previous day salinity': row[f'Salinity {col_day -1}'],
                'weekly average temperature': row[week_temperature_column].mean(),
                'weekly average salinity': row[week_salinity_column].mean(),
                'weekly max temperature': row[week_temperature_column].max(),
                'weekly min temperature': row[week_temperature_column].min(),
                'weekly max salinity': row[week_salinity_column].max(),
                'weekly min salinity': row[week_salinity_column].min(),
                'death': row[f'Lived {col_day}']
            }
            one_week_data.append(new_row)

    one_week_df = pd.DataFrame(one_week_data)

    for col in [col for col in one_week_df.columns if col not in ['weekly average temperature', 'weekly average salinity']]:
        one_week_df[col] = one_week_df[col].astype(int)

    return one_week_df


def create_one_week_data(train_path, test_path, model_data_dir):
    train_df = pd.read_csv(train_path)
    one_week_train_df = convert_routes_to_one_week_data(train_df)
    one_week_train_df.to_csv(model_data_dir / 'train.csv', index=False)

    test_df = pd.read_csv(test_path)
    one_week_test_df = convert_routes_to_one_week_data(test_df)
    one_week_test_df.to_csv(model_data_dir / 'test.csv', index=False)

    return one_week_train_df, one_week_test_df


def run_analysis(train_path, test_path, outputs_dir):
    os.makedirs(outputs_dir, exist_ok=True)
    model_data_dir = outputs_dir / 'data'
    os.makedirs(model_data_dir, exist_ok=True)
    model_train_dir = outputs_dir / 'train_outputs'
    os.makedirs(model_train_dir, exist_ok=True)
    model_test_dir = outputs_dir / 'test_outputs'
    os.makedirs(model_test_dir, exist_ok=True)

    one_week_train_df, one_week_test_df = create_one_week_data(train_path, test_path, model_data_dir)

    fit_on_train_data(one_week_train_df.drop(columns=['death']), one_week_train_df['death'],
                      model_train_dir, -1, 'ONE_WEEK_MODEL_TRAIN')
    test_on_test_data(model_train_dir / 'best_model.pkl', one_week_test_df.drop(columns=['death']),
                      one_week_test_df['death'], model_test_dir, 'ONE_WEEK_MODEL_TEST')


def main():
    run_analysis(TRAIN_INCLUDE_SUSPECTED_PATH, TEST_INCLUDE_SUSPECTED_PATH, OUTPUTS_INCLUDE_SUSPECTED_DIR / 'one_week_model')
    run_analysis(TRAIN_EXCLUDE_SUSPECTED_PATH, TEST_EXCLUDE_SUSPECTED_PATH, OUTPUTS_EXCLUDE_SUSPECTED_DIR / 'one_week_model')


if __name__ == '__main__':
    main()
