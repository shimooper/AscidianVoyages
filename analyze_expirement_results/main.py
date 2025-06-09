import argparse
import itertools

from configuration import ROOT_DIR, STRATIFY_TRAIN_TEST_SPLIT, RANDOM_STATE, METRIC_TO_CHOOSE_BEST_MODEL_HYPER_PARAMS, \
    INCLUDE_SUSPECTED_ROUTES_PARTS, INCLUDE_CONTROL_ROUTES, NUMBER_OF_FUTURE_DAYS_TO_CONSIDER_DEATH, Config, str_to_bool, \
    BALANCE_CLASSES_IN_TRAINING
from preprocess import preprocess_data
from run_analysis_of_config import run_analysis_of_one_config
from q_submitter_power import submit_mini_batch, wait_for_results, get_job_logger


def create_config(flag_combination, configuration_id, cpus, outputs_dir, do_feature_selection, run_lstm_configurations_in_parallel, error_file_path):
    include_control_flag, include_suspected_flag, number_of_future_days, stratify_flag, random_state, metric, balance_classes = flag_combination

    outputs_dir_path = outputs_dir / f'configuration_{configuration_id}'
    outputs_dir_path.mkdir(exist_ok=True, parents=True)

    config = Config(
        error_file_path=error_file_path,
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
        run_lstm_configurations_in_parallel=run_lstm_configurations_in_parallel
    )
    config_path = outputs_dir_path / 'config.csv'
    config.to_csv(config_path)

    return config, config_path


def main(outputs_dir, cpus, do_feature_selection, run_configurations_in_parallel, run_lstm_configurations_in_parallel):
    outputs_dir.mkdir(exist_ok=True, parents=True)
    main_logger = get_job_logger(outputs_dir, 'main')
    error_file_path = outputs_dir / 'error.txt'

    preprocess_data(main_logger)

    flags_combinations = list(itertools.product(
        INCLUDE_CONTROL_ROUTES, INCLUDE_SUSPECTED_ROUTES_PARTS, NUMBER_OF_FUTURE_DAYS_TO_CONSIDER_DEATH,
        STRATIFY_TRAIN_TEST_SPLIT, RANDOM_STATE, METRIC_TO_CHOOSE_BEST_MODEL_HYPER_PARAMS, BALANCE_CLASSES_IN_TRAINING))

    configuration_id = 0
    if run_configurations_in_parallel:
        script_path = ROOT_DIR / 'run_analysis_of_config.py'
        for flags_combination in flags_combinations:
            _, config_path = create_config(flags_combination, configuration_id, 8,
                                   outputs_dir, do_feature_selection, run_lstm_configurations_in_parallel, error_file_path)
            submit_mini_batch(main_logger, script_path, [[config_path]], outputs_dir, f'config_{configuration_id}', num_of_cpus=8)
            configuration_id += 1
        wait_for_results(main_logger, script_path, outputs_dir, configuration_id, error_file_path)
    else:
        for flags_combination in flags_combinations:
            config, _ = create_config(flags_combination, configuration_id, cpus, outputs_dir, do_feature_selection, run_lstm_configurations_in_parallel, error_file_path)
            run_analysis_of_one_config(main_logger, config)
            configuration_id += 1

    main_logger.info('All configurations have been processed.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the analysis of the experiment results')
    parser.add_argument('--outputs_dir_name', type=str, default='outputs_test_local', help='outputs dir name')
    parser.add_argument('--cpus', type=int, default=1, help='Number of CPUs to use')
    parser.add_argument('--do_feature_selection', type=str_to_bool, default=False)
    parser.add_argument('--run_configurations_in_parallel', type=str_to_bool, default=False)
    parser.add_argument('--run_lstm_configurations_in_parallel', type=str_to_bool, default=False)

    args = parser.parse_args()
    main(ROOT_DIR / args.outputs_dir_name, args.cpus, args.do_feature_selection, args.run_configurations_in_parallel,
         args.run_lstm_configurations_in_parallel)
