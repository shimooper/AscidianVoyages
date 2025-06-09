from collections import defaultdict
import pandas as pd
import argparse
from pathlib import Path

from configuration import Config, NUMBER_OF_DAYS_TO_CONSIDER, DEBUG_MODE, PROCESSED_DATA_PATH
from preprocess import preprocess_data_by_config
from routes_visualization import plot_timelines
from model import ScikitModel
from utils import plot_models_comparison
from q_submitter_power import run_step


def run_analysis_of_one_config(logger, config: Config):
    config.data_dir_path.mkdir(exist_ok=True, parents=True)
    preprocess_data_by_config(logger, config, PROCESSED_DATA_PATH)

    if not DEBUG_MODE:
        plot_timelines(config)

    model_id = 0
    for number_of_days_to_consider in NUMBER_OF_DAYS_TO_CONSIDER:
        model_instance = ScikitModel(config, model_id, number_of_days_to_consider)
        model_instance.run_analysis(logger)
        model_id += 1

    aggregate_validation_metrics_of_one_configuration(config)


def aggregate_validation_metrics_of_one_configuration(config: Config):
    classifier_name_to_validation_results = defaultdict(dict)

    for model_dir_path in config.models_dir_path.iterdir():
        if not model_dir_path.is_dir() or not model_dir_path.name.startswith('days_to_consider_'):
            continue

        model_results_path = model_dir_path / 'train_outputs' / 'best_classifier' / 'best_classifier_from_each_class.csv'
        model_results_df = pd.read_csv(model_results_path, index_col='model_name')
        for classifier_name in model_results_df.index.values:
            classifier_name_to_validation_results[classifier_name][model_dir_path.name] = model_results_df.loc[classifier_name]

    for classifier_name, days_to_results_dict in classifier_name_to_validation_results.items():
        days_comparison_df = pd.DataFrame.from_dict(days_to_results_dict, orient='index')
        days_comparison_df.index.name = 'model_name'

        classifier_comparison_dir = config.models_dir_path / 'classifier_comparison' / classifier_name
        classifier_comparison_dir.mkdir(exist_ok=True, parents=True)
        plot_models_comparison(days_comparison_df, classifier_comparison_dir, config.metric)


def main():
    parser = argparse.ArgumentParser(description="Run analysis of configurations")
    parser.add_argument('config_path', type=Path, help='Path to the configuration CSV file')
    args = parser.parse_args()

    config = Config.from_csv(args.config_path)
    run_step(config.outputs_dir_path, f'config_{config.configuration_id}', config.error_file_path, run_analysis_of_one_config, config)


if __name__ == "__main__":
    main()
