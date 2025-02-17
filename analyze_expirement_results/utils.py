import os
import pandas as pd
import logging
import re


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
