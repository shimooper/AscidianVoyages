import pandas as pd
import argparse
import itertools
from collections import defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

from configuration import SCRIPT_DIR, STRATIFY_TRAIN_TEST_SPLIT, RANDOM_STATE, METRIC_TO_CHOOSE_BEST_MODEL_HYPER_PARAMS, \
    INCLUDE_SUSPECTED_ROUTES_PARTS, INCLUDE_CONTROL_ROUTES, NUMBER_OF_FUTURE_DAYS_TO_CONSIDER_DEATH, Config, str_to_bool, \
    DEBUG_MODE, BALANCE_CLASSES_IN_TRAINING, NUMBER_OF_DAYS_TO_CONSIDER
from preprocess import get_preprocessed_data, preprocess_data_by_config
from routes_visualization import plot_timelines
from model import ScikitModel
from utils import plot_models_comparison


def create_config(flag_combination, configuration_id, cpus, outputs_dir, do_feature_selection):
    include_control_flag, include_suspected_flag, number_of_future_days, stratify_flag, random_state, metric, balance_classes = flag_combination

    outputs_dir_path = outputs_dir / f'configuration_{configuration_id}'
    outputs_dir_path.mkdir(exist_ok=True, parents=True)

    config = Config(
        include_control_routes=include_control_flag,
        include_suspected_routes=include_suspected_flag,
        number_of_future_days_to_consider_death=number_of_future_days,
        stratify=stratify_flag,
        random_state=random_state,
        metric=metric,
        configuration_id=configuration_id,
        cpus=cpus,
        outputs_dir_path=outputs_dir_path,
        data_dir_path=outputs_dir_path / 'data',
        models_dir_path=outputs_dir_path / 'models',
        do_feature_selection=do_feature_selection,
        balance_classes=balance_classes,
    )
    config.to_csv(outputs_dir_path / 'config.csv')

    return config


def run_analysis_of_one_config(config: Config):
    config.data_dir_path.mkdir(exist_ok=True, parents=True)

    processed_df = get_preprocessed_data()
    preprocess_data_by_config(config, processed_df)

    if not DEBUG_MODE:
        plot_timelines(config)

    model_id = 0
    for number_of_days_to_consider in NUMBER_OF_DAYS_TO_CONSIDER:
        model_instance = ScikitModel(config, model_id, number_of_days_to_consider)
        model_instance.run_analysis()
        model_id += 1

    aggregate_validation_metrics_of_one_configuration(config)


def main(outputs_dir, cpus, do_feature_selection, run_configurations_in_parallel):
    flags_combinations = list(itertools.product(
        INCLUDE_CONTROL_ROUTES, INCLUDE_SUSPECTED_ROUTES_PARTS, NUMBER_OF_FUTURE_DAYS_TO_CONSIDER_DEATH,
        STRATIFY_TRAIN_TEST_SPLIT, RANDOM_STATE, METRIC_TO_CHOOSE_BEST_MODEL_HYPER_PARAMS, BALANCE_CLASSES_IN_TRAINING))
    configuration_id = 0

    if run_configurations_in_parallel:
        with ProcessPoolExecutor(max_workers=cpus) as executor:
            for flags_combination in flags_combinations:
                config = create_config(flags_combination, configuration_id, max(1, cpus // len(flags_combinations)),
                                       outputs_dir, do_feature_selection)
                executor.submit(run_analysis_of_one_config, config)
                configuration_id += 1
    else:
        for flags_combination in flags_combinations:
            config = create_config(flags_combination, configuration_id, cpus, outputs_dir, do_feature_selection)
            run_analysis_of_one_config(config)
            configuration_id += 1

    print('All configurations have been processed.')


def aggregate_validation_metrics_of_one_configuration(config: Config):
    classifier_name_to_validation_results = defaultdict(dict)

    for model_dir_path in config.models_dir_path.iterdir():
        model_validation_results_path = model_dir_path / 'train_outputs' / 'best_classifier' / 'best_classifier_from_each_class.csv'
        model_validation_results_df = pd.read_csv(model_validation_results_path, index_col='model_name')[
            ['validation f1', 'validation auprc', 'validation mcc']]
        model_validation_results_df.rename(columns={
            'validation f1': 'F1',
            'validation auprc': 'AUPRC',
            'validation mcc': 'MCC'
        }, inplace=True)
        for classifier_name in model_validation_results_df.index.values:
            classifier_name_to_validation_results[classifier_name][model_dir_path.name] = model_validation_results_df.loc[classifier_name]

    for classifier_name, days_to_results_dict in classifier_name_to_validation_results.items():
        days_comparison_df = pd.DataFrame.from_dict(days_to_results_dict, orient='index')
        days_comparison_df.index.name = 'model_name'

        classifier_comparison_dir = config.models_dir_path / 'classifier_comparison' / classifier_name
        classifier_comparison_dir.mkdir(exist_ok=True, parents=True)
        plot_models_comparison(days_comparison_df.reset_index(), classifier_comparison_dir, f'Models comparison - validation set - {classifier_name}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the analysis of the experiment results')
    parser.add_argument('--outputs_dir_name', type=str, default='outputs_test_local', help='outputs dir name')
    parser.add_argument('--cpus', type=int, default=1, help='Number of CPUs to use')
    parser.add_argument('--do_feature_selection', type=str_to_bool, default=False)
    parser.add_argument('--run_configurations_in_parallel', type=str_to_bool, default=False)

    args = parser.parse_args()
    main(SCRIPT_DIR / args.outputs_dir_name, args.cpus, args.do_feature_selection, args.run_configurations_in_parallel)
