import pandas as pd

from utils import DATA_PROCESSED_PATH


def main():
    df = pd.read_csv(DATA_PROCESSED_PATH)
    