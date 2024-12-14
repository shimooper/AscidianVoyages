import os
from pathlib import Path
import pandas as pd
import logging
import re

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'data'
DATA_PATH = DATA_DIR / 'Final_Data_Voyages.xlsx'
OUTPUTS_DIR = SCRIPT_DIR / 'outputs'

INCLUDE_CONTROL_ROUTES = [
    True,
    # False
]
INCLUDE_SUSPECTED_ROUTES_PARTS = [
    # True,
    False
]
STRATIFY_TRAIN_TEST_SPLIT = [
    # True,
    False
]
RANDOM_STATE = [
    42,
    # 0
]
METRIC_TO_CHOOSE_BEST_MODEL_PARAMS = [
    'mcc',
    # 'f1'
]
TEST_SET_SIZE = 0.25
N_JOBS = -1  # Use all available CPUs


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
