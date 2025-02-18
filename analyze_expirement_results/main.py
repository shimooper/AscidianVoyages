import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

from configuration import SCRIPT_DIR, STRATIFY_TRAIN_TEST_SPLIT, RANDOM_STATE, METRIC_TO_CHOOSE_BEST_MODEL_HYPER_PARAMS, \
    INCLUDE_SUSPECTED_ROUTES_PARTS, INCLUDE_CONTROL_ROUTES, NUMBER_OF_FUTURE_DAYS_TO_CONSIDER_DEATH, Config, str_to_bool, \
    DEFAULT_OPTUNA_NUMBER_OF_TRIALS, DEBUG_MODE
from preprocess import permanent_preprocess_data, preprocess_data_by_config
from routes_visualization import plot_timelines
from model import ScikitModel, OptunaModel


def run_analysis_of_one_config(config: Config, processed_df):
    config.data_dir_path.mkdir(exist_ok=True, parents=True)
    preprocess_data_by_config(config, processed_df)

    if not DEBUG_MODE:
        plot_timelines(config)

    model_id = 0
    for number_of_future_days in NUMBER_OF_FUTURE_DAYS_TO_CONSIDER_DEATH:
        model_class = OptunaModel if config.train_with_optuna else ScikitModel
        model_instance = model_class(config, model_id, number_of_future_days)
        model_instance.run_analysis()
        model_id += 1

    aggregate_test_metrics_of_one_configuration(config)


def main(outputs_dir, cpus, do_feature_selection, train_with_optuna, optuna_number_of_trials):
    processed_df = permanent_preprocess_data()

    if cpus == 1:
        pool_executor_class = ThreadPoolExecutor
    else:
        pool_executor_class = ProcessPoolExecutor

    with pool_executor_class(max_workers=cpus) as executor:
        futures = []

        flag_combinations = list(itertools.product(
            INCLUDE_CONTROL_ROUTES, INCLUDE_SUSPECTED_ROUTES_PARTS, STRATIFY_TRAIN_TEST_SPLIT, RANDOM_STATE,
            METRIC_TO_CHOOSE_BEST_MODEL_HYPER_PARAMS))
        configuration_id = 0
        for include_control_flag, include_suspected_flag, stratify_flag, random_state, metric in flag_combinations:
            outputs_dir_path = outputs_dir / f'configuration_{configuration_id}'
            outputs_dir_path.mkdir(exist_ok=True, parents=True)
            config = Config(
                include_control_routes=include_control_flag,
                include_suspected_routes=include_suspected_flag,
                stratify=stratify_flag,
                random_state=random_state,
                metric=metric,
                configuration_id=configuration_id,
                cpus=max(1, cpus // len(flag_combinations)),
                outputs_dir_path=outputs_dir_path,
                data_dir_path=outputs_dir_path / 'data',
                models_dir_path=outputs_dir_path / 'models',
                do_feature_selection=do_feature_selection,
                train_with_optuna=train_with_optuna,
                optuna_number_of_trials=optuna_number_of_trials
            )
            config.to_csv(outputs_dir_path / 'config.csv')

            futures.append(executor.submit(run_analysis_of_one_config, config, processed_df))
            configuration_id += 1

        for future in as_completed(futures):
            future.result()

    aggregate_all_configuration_results(outputs_dir)


def aggregate_test_metrics_of_one_configuration(config: Config):
    all_models_test_results = {}

    for model_dir_path in config.models_dir_path.iterdir():
        model_test_results_path = model_dir_path / 'test_outputs' / 'best_classifier_test_results.csv'
        model_test_results_df = pd.read_csv(model_test_results_path)
        all_models_test_results[model_dir_path.name] = model_test_results_df.loc[0]

    all_models_test_results_df = pd.DataFrame.from_dict(all_models_test_results, orient='index')
    all_models_test_results_df.index.name = 'model_name'
    all_models_test_results_df.reset_index(inplace=True)

    plot_models_comparison(all_models_test_results_df, config.outputs_dir_path)


def aggregate_all_configuration_results(outputs_dir: Path):
    all_config_best_results = []

    for config_output_dir in outputs_dir.iterdir():
        if not config_output_dir.is_dir():
            continue

        config = Config.from_csv(config_output_dir / 'config.csv')
        optimized_metric = config.metric

        models_results = pd.read_csv(config_output_dir / 'all_models_comparison.csv', index_col='model_name')
        best_model_name = models_results.loc['best_model', optimized_metric]
        best_model_metrics = models_results.loc[best_model_name]
        best_model_metrics['model_name'] = f"configId_{config.configuration_id}_{best_model_name}"
        all_config_best_results.append(best_model_metrics)

    all_config_best_results_df = pd.DataFrame(all_config_best_results)
    all_config_best_results_df.reset_index(inplace=True, drop=True)
    all_config_best_results_df['mcc'] = pd.to_numeric(all_config_best_results_df['mcc'])
    all_config_best_results_df['f1'] = pd.to_numeric(all_config_best_results_df['f1'])
    all_config_best_results_df['auprc'] = pd.to_numeric(all_config_best_results_df['auprc'])

    plot_models_comparison(all_config_best_results_df, outputs_dir)


def plot_models_comparison(results_df, outputs_dir: Path):
    all_models_test_results_df_melted = results_df.melt(id_vars='model_name', var_name='metric', value_name='value')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=all_models_test_results_df_melted, x='metric', y='value', hue='model_name', palette='viridis')
    plt.title('Comparison of models')
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(outputs_dir / 'all_models_comparison.png', dpi=300)
    plt.close()

    results_df.set_index('model_name', inplace=True)
    max_indices = results_df.idxmax()
    results_df.loc['best_model'] = max_indices
    results_df.to_csv(outputs_dir / 'all_models_comparison.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the analysis of the experiment results')
    parser.add_argument('--outputs_dir_name', type=str, default='outputs', help='outputs dir name')
    parser.add_argument('--cpus', type=int, default=8, help='Number of CPUs to use')
    parser.add_argument('--do_feature_selection', type=str_to_bool, default=True)
    parser.add_argument('--train_with_optuna', type=str_to_bool, default=True)
    parser.add_argument('--optuna_number_of_trials', type=int, default=DEFAULT_OPTUNA_NUMBER_OF_TRIALS)

    args = parser.parse_args()
    main(SCRIPT_DIR / args.outputs_dir_name, args.cpus, args.do_feature_selection, args.train_with_optuna,
         args.optuna_number_of_trials)
