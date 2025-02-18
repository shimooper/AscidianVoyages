import pandas as pd
import os
from pathlib import Path
import sys

sys.path.append(r"C:\repos\GoogleShips\analyze_expirement_results")
from analyze_expirement_results.main import plot_models_comparison

PATH = r"C:\repos\GoogleShips\analyze_expirement_results\outputs_optuna\all_models_comparison.csv"


def main():
    results_df = pd.read_csv(PATH)
    outputs_dir = Path(os.getcwd())
    plot_models_comparison(results_df, outputs_dir)


if __name__ == '__main__':
    main()
