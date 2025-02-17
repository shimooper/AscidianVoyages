import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

from configuration import OUTPUTS_DIR, STRATIFY_TRAIN_TEST_SPLIT, RANDOM_STATE, METRIC_TO_CHOOSE_BEST_MODEL_HYPER_PARAMS, \
    INCLUDE_SUSPECTED_ROUTES_PARTS, INCLUDE_CONTROL_ROUTES, NUMBER_OF_FUTURE_DAYS_TO_CONSIDER_DEATH, Config, DEBUG_MODE
from preprocess import permanent_preprocess_data, preprocess_data_by_config
from routes_visualization import plot_timelines
from model import Model


def run_analysis_of_one_config(config: Config, processed_df):
    config.data_dir_path.mkdir(exist_ok=True, parents=True)
    preprocess_data_by_config(config, processed_df)

    if not DEBUG_MODE:
        plot_timelines(config)

    model_id = 0
    for number_of_future_days in NUMBER_OF_FUTURE_DAYS_TO_CONSIDER_DEATH:
        model_instance = Model(config, model_id, number_of_future_days)
        model_instance.run_analysis()
        model_id += 1

    aggregate_test_metrics_of_one_configuration(config)


def main(cpus):
    processed_df = permanent_preprocess_data()

    if cpus == 1:
        pool_executor_class = ThreadPoolExecutor
    else:
        pool_executor_class = ProcessPoolExecutor

    with pool_executor_class(max_workers=cpus) as executor:
        futures = []
        configuration_id = 0
        for include_control_flag, include_suspected_flag, stratify_flag, random_state, metric in itertools.product(
                INCLUDE_CONTROL_ROUTES, INCLUDE_SUSPECTED_ROUTES_PARTS, STRATIFY_TRAIN_TEST_SPLIT, RANDOM_STATE,
                METRIC_TO_CHOOSE_BEST_MODEL_HYPER_PARAMS):

            outputs_dir_path = OUTPUTS_DIR / f'configuration_{configuration_id}'
            outputs_dir_path.mkdir(exist_ok=True, parents=True)
            config = Config(
                include_control_routes=include_control_flag,
                include_suspected_routes=include_suspected_flag,
                stratify=stratify_flag,
                random_state=random_state,
                metric=metric,
                configuration_id=configuration_id,
                cpus=cpus,
                outputs_dir_path=outputs_dir_path,
                data_dir_path=outputs_dir_path / 'data',
                models_dir_path=outputs_dir_path / 'models'
            )
            config.to_csv(outputs_dir_path / 'config.csv')

            futures.append(executor.submit(run_analysis_of_one_config, config, processed_df))
            configuration_id += 1


def aggregate_test_metrics_of_one_configuration(config: Config):
    all_models_test_results = {}

    for model_dir_path in config.models_dir_path.iterdir():
        model_test_results_path = model_dir_path / 'test_outputs' / 'best_classifier_test_results.csv'
        model_test_results_df = pd.read_csv(model_test_results_path)
        all_models_test_results[model_dir_path.name] = model_test_results_df.loc[0]

    all_models_test_results_df = pd.DataFrame.from_dict(all_models_test_results, orient='index')
    all_models_test_results_df.index.name = 'model_name'
    all_models_test_results_df.reset_index(inplace=True)

    all_models_test_results_df_melted = all_models_test_results_df.melt(id_vars='model_name', var_name='metric', value_name='value')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=all_models_test_results_df_melted, x='metric', y='value', hue='model_name', palette='viridis')
    plt.title('Comparison of models')
    plt.ylabel('Score')
    plt.xlabel('Metric')
    plt.legend(title='Model', bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(config.outputs_dir_path / 'all_models_comparison.png', dpi=300)
    plt.close()

    all_models_test_results_df.set_index('model_name', inplace=True)
    max_indices = all_models_test_results_df.idxmax()
    all_models_test_results_df.loc['best_model'] = max_indices
    all_models_test_results_df.to_csv(config.outputs_dir_path / 'all_models_comparison.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the analysis of the experiment results')
    parser.add_argument('--cpus', type=int, default=1, help='Number of CPUs to use')
    args = parser.parse_args()
    main(args.cpus)
