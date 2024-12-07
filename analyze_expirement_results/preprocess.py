import os
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import (variable_equals_value, setup_logger, DATA_PATH, OUTPUTS_DIR,
                   OUTPUTS_INCLUDE_SUSPECTED_DIR, OUTPUTS_EXCLUDE_SUSPECTED_DIR, RANDOM_STATE,
                   STRATIFY_TRAIN_TEST_SPLIT)


OUTPUTS_PREPROCESS_DIR = OUTPUTS_DIR / 'preprocess'
DATA_PROCESSED_PATH = OUTPUTS_PREPROCESS_DIR / 'Final_Data_Voyages_Processed.csv'
DATA_PROCESSED_EXCLUDE_CONTROL_PATH = OUTPUTS_PREPROCESS_DIR / 'Final_Data_Voyages_Processed_Exclude_Control.csv'
TEST_SET_SIZE = 0.25


def preprocess_data():
    os.makedirs(OUTPUTS_PREPROCESS_DIR, exist_ok=True)
    logger = setup_logger(OUTPUTS_PREPROCESS_DIR / 'preprocess.log', 'PREPROCESS')

    df = pd.read_excel(DATA_PATH, sheet_name='final_data')

    # Convert Temp, Salinity, Lived columns to integers
    df.replace("\\", pd.NA, inplace=True)
    lived_columns = sorted([col for col in df.columns if 'Lived' in col], key=lambda x: int(x.split()[1]))
    temperature_columns = sorted([col for col in df.columns if 'Temp' in col], key=lambda x: int(x.split()[1]))
    salinity_columns = sorted([col for col in df.columns if 'Salinity' in col], key=lambda x: int(x.split()[1]))

    for col in lived_columns + temperature_columns + salinity_columns + ['Suspected from time point']:
        df[col] = pd.to_numeric(df[col], errors='raise').astype("Int64")

    replace_lived_indicators(df, lived_columns)
    add_dying_day(df, lived_columns, temperature_columns, salinity_columns)

    df['stratify_group'] = df['Season'] + '_' + df['dying_day'].notna().astype(int).astype(str)

    df.to_csv(DATA_PROCESSED_PATH, index=False)

    df_exclude_control = df[df['Name'] != 'CONTROL']
    df_exclude_control.to_csv(DATA_PROCESSED_EXCLUDE_CONTROL_PATH, index=False)
    logger.info(f"Originally there were {len(df)} rows in the dataset. "
                f"After removing the CONTROL group, there are {len(df_exclude_control)} rows. "
                f"Of which, {df_exclude_control['dying_day'].notna().sum()} ended with death "
                f"and {df_exclude_control['dying_day'].isna().sum()} did not.")

    # From now on, I refer only to the data excluding the CONTROL group
    df_exclude_control.to_csv(OUTPUTS_INCLUDE_SUSPECTED_DIR / 'full.csv', index=False)
    create_train_test_splits(logger, df_exclude_control, OUTPUTS_INCLUDE_SUSPECTED_DIR)
    remove_suspected_routes_parts(df_exclude_control, lived_columns, temperature_columns, salinity_columns,
                                  OUTPUTS_EXCLUDE_SUSPECTED_DIR / 'full.csv')
    create_train_test_splits(logger, df_exclude_control, OUTPUTS_EXCLUDE_SUSPECTED_DIR)


def replace_lived_indicators(df, lived_columns):
    # In Lived columns: replace 1 with 0, 0 with 1, keeping NaN as is
    for col in lived_columns:
        df[col] = df[col].apply(lambda x: 1 if variable_equals_value(x, 0) else (0 if variable_equals_value(x, 1) else x))

    # Convert all values in Lived columns to integers again
    for col in lived_columns:
        df[col] = pd.to_numeric(df[col], errors='raise').astype("Int64")


def add_dying_day(df, lived_columns, temperature_columns, salinity_columns):
    # Add dying_day column and convert all values after death to nan
    dying_day = {}
    # Iterate over each row
    for idx, row in df.iterrows():
        # Find the last column of survival (containing 0)
        last_col_of_survival = None
        for col in lived_columns:
            if variable_equals_value(row[col], 0):
                last_col_of_survival = col

        # Fill all previous columns with 0
        last_col_of_survival_index = lived_columns.index(last_col_of_survival)
        df.loc[idx, lived_columns[:last_col_of_survival_index + 1]] = 0

        first_col_of_death_index = last_col_of_survival_index + 1
        if first_col_of_death_index < len(lived_columns) and variable_equals_value(df.loc[idx, lived_columns[first_col_of_death_index]], 1):
            # Assign nan to all following columns
            df.loc[idx, lived_columns[first_col_of_death_index + 1:]] = pd.NA
            df.loc[idx, temperature_columns[first_col_of_death_index + 1:]] = pd.NA
            df.loc[idx, salinity_columns[first_col_of_death_index + 1:]] = pd.NA

            dying_day[idx] = first_col_of_death_index
        else:  # The animal lived until the end of the experiment
            dying_day[idx] = pd.NA

    df['dying_day'] = df.index.map(dying_day)

    # validate that the nan values are consistent in all columns
    for idx, row in df.iterrows():
        for lived_col, temp_col, salinity_col in zip(lived_columns, temperature_columns, salinity_columns):
            if pd.isna(row[lived_col]) or pd.isna(row[temp_col]) or pd.isna(row[salinity_col]):
                if pd.isna(row[lived_col]) and pd.isna(row[temp_col]) and pd.isna(row[salinity_col]):
                    continue
                else:
                    raise ValueError(f'Row {idx} contains NaN values in some but not all of the columns: '
                                     f'{lived_col}, {temp_col}, {salinity_col}')


def create_train_test_splits(logger, df, output_dir):
    if STRATIFY_TRAIN_TEST_SPLIT:
        stratify = df['stratify_group']
    else:
        stratify = None

    train_df, test_df = train_test_split(df, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE, stratify=stratify)
    train_df.to_csv(output_dir / 'train.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)
    logger.info(f"Train set has {len(train_df)} routes and test set has {len(test_df)} routes.")


def remove_suspected_routes_parts(df, lived_columns, temperature_columns, salinity_columns, output_path):
    for idx, row in df.iterrows():
        if pd.isna(row['Suspected from time point']):
            continue

        if row['Suspected from time point'] != row['dying_day']:
            raise ValueError(f"Row {idx} has different values in 'Suspected from time point' and 'dying_day' columns.")

        for col in lived_columns + temperature_columns + salinity_columns:
            if int(col.split()[1]) >= row['Suspected from time point']:
                df.loc[idx, col] = pd.NA

    df.to_csv(output_path, index=False)
