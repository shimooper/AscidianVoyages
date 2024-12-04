from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / 'data' / 'Final_Data_Voyages.xlsx'
DATA_PROCESSED_PATH = SCRIPT_DIR / 'data' / 'Final_Data_Voyages_Processed.csv'
DATA_PROCESSED_NO_CONTROL_PATH = SCRIPT_DIR / 'data' / 'Final_Data_Voyages_Processed_No_Control.csv'
TRAIN_PATH = SCRIPT_DIR / 'data' / 'train.csv'
TEST_PATH = SCRIPT_DIR / 'data' / 'test.csv'

OUTPUTS_DIR = SCRIPT_DIR / 'outputs'

RANDOM_STATE = 42
TEST_SET_SIZE = 0.25


def variable_equals_value(variable, value):
    if pd.isna(value):
        return pd.isna(variable)

    return pd.notna(variable) and variable == value
