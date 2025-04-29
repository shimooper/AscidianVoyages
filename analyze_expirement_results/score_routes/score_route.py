from pathlib import Path
import pandas as pd
import numpy as np
import argparse


def prepare_data(actual_experiment_data, planned_experiment_data, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    actual_df = pd.read_csv(actual_experiment_data)
    planned_df = pd.read_csv(planned_experiment_data)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data for scoring routes')
    parser.add_argument('--actual_experiment_data', type=Path, required=True, help='Path to the actual experiment data')
    parser.add_argument('--planned_experiment_data', type=Path, required=True, help='Path to the planned experiment data')
    parser.add_argument('--output_dir', type=Path, required=True, help='Directory to save the prepared data')
    parser.add_argument('--model_dir', type=Path, required=True, help='Directory of the trained 4-day model')
    args = parser.parse_args()

    prepare_data(args.actual_experiment_data, args.planned_experiment_data, args.output_dir)
