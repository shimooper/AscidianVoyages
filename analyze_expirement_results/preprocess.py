import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / '..' / 'data' / 'Final_Data_Voyages.xlsx'


def main():
    df = pd.read_excel(DATA_PATH)



if __name__ == '__main__':
    main()
