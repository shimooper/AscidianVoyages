import os
from pathlib import Path
import pandas as pd
import logging
import re

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / 'data' / 'Final_Data_Voyages.xlsx'
OUTPUTS_DIR = SCRIPT_DIR / 'outputs'

STRATIFY_TRAIN_TEST_SPLIT = [True, False]
RANDOM_STATE = [42, 0]
METRIC_TO_CHOOSE_BEST_MODEL_PARAMS = ['mcc', 'f1']
TEST_SET_SIZE = 0.25


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
