from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / 'data' / 'Final_Data_Voyages.xlsx'
DATA_PROCESSED_PATH = SCRIPT_DIR / 'data' / 'Final_Data_Voyages_Processed.csv'

OUTPUTS_DIR = SCRIPT_DIR / 'outputs'


def variable_equals_value(variable, value):
    if pd.isna(value):
        return pd.isna(variable)

    return pd.notna(variable) and variable == value
