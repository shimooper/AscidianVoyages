import os
from pathlib import Path
import pandas as pd
import logging

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'data'
DATA_PATH = DATA_DIR / 'Final_Data_Voyages.xlsx'


# The data I refer to from now on is always excluding the CONTROL routes
DATA_PROCESSED_INCLUDE_SUSPECTED_DIR = SCRIPT_DIR / 'data' / 'include_suspected'
os.makedirs(DATA_PROCESSED_INCLUDE_SUSPECTED_DIR, exist_ok=True)
FULL_INCLUDE_SUSPECTED_PATH = DATA_PROCESSED_INCLUDE_SUSPECTED_DIR / 'full.csv'
TRAIN_INCLUDE_SUSPECTED_PATH = DATA_PROCESSED_INCLUDE_SUSPECTED_DIR / 'train.csv'
TEST_INCLUDE_SUSPECTED_PATH = DATA_PROCESSED_INCLUDE_SUSPECTED_DIR / 'test.csv'

DATA_PROCESSED_EXCLUDE_SUSPECTED_DIR = SCRIPT_DIR / 'data' / 'exclude_suspected'
os.makedirs(DATA_PROCESSED_EXCLUDE_SUSPECTED_DIR, exist_ok=True)
FULL_EXCLUDE_SUSPECTED_PATH = DATA_PROCESSED_EXCLUDE_SUSPECTED_DIR / 'full.csv'
TRAIN_EXCLUDE_SUSPECTED_PATH = DATA_PROCESSED_EXCLUDE_SUSPECTED_DIR / 'train.csv'
TEST_EXCLUDE_SUSPECTED_PATH = DATA_PROCESSED_EXCLUDE_SUSPECTED_DIR / 'test.csv'

OUTPUTS_DIR = SCRIPT_DIR / 'outputs'
OUTPUTS_INCLUDE_SUSPECTED_DIR = OUTPUTS_DIR / 'include_suspected'
OUTPUTS_EXCLUDE_SUSPECTED_DIR = OUTPUTS_DIR / 'exclude_suspected'

RANDOM_STATE = 42


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
