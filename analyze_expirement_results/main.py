import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

from utils import OUTPUTS_DIR, STRATIFY_TRAIN_TEST_SPLIT, RANDOM_STATE, METRIC_TO_CHOOSE_BEST_MODEL_HYPER_PARAMS, \
    INCLUDE_SUSPECTED_ROUTES_PARTS, INCLUDE_CONTROL_ROUTES, NUMBER_OF_FUTURE_DAYS_TO_CONSIDER_DEATH, N_JOBS
from preprocess import permanent_preprocess_data, preprocess_data
from routes_visualization import plot_timelines
from model import Model


def main(cpus):
    processed_df = permanent_preprocess_data()

    configuration_id = 0
    for include_control_flag, include_suspected_flag, stratify_flag, random_state, metric in itertools.product(
            INCLUDE_CONTROL_ROUTES, INCLUDE_SUSPECTED_ROUTES_PARTS, STRATIFY_TRAIN_TEST_SPLIT, RANDOM_STATE,
            METRIC_TO_CHOOSE_BEST_MODEL_HYPER_PARAMS):
        outputs_dir = OUTPUTS_DIR / f'configuration_{configuration_id}'
        outputs_dir.mkdir(exist_ok=True, parents=True)

        configuration = {
            'include_control_flag': include_control_flag,
            'include_suspected_flag': include_suspected_flag,
            'stratify_flag': stratify_flag,
            'random_state': random_state,
            'metric': metric
        }

        configuration_df = pd.DataFrame(list(configuration.items()), columns=['key', 'value'])
        configuration_df.to_csv(outputs_dir / 'configuration.csv', index=False)

        data_path = outputs_dir / 'data'
        data_path.mkdir(exist_ok=True, parents=True)
        preprocess_data(data_path, processed_df, include_control_flag, include_suspected_flag, stratify_flag,
                        random_state, configuration_id)
        plot_timelines(data_path)

        model_id = 0
        for number_of_future_days in NUMBER_OF_FUTURE_DAYS_TO_CONSIDER_DEATH:
            model_name = f'future_days_to_consider_death_{number_of_future_days}'
            model_instance = Model(f'{configuration_id}_{model_id}', model_name, data_path,
                                   outputs_dir / 'models', metric, random_state, number_of_future_days)
            model_instance.run_analysis(cpus)
            model_id += 1

        aggregate_test_metrics_of_one_configuration(outputs_dir)
        configuration_id += 1


def aggregate_test_metrics_of_one_configuration(outputs_dir):
    all_models_test_results = {}

    for model_dir_path in (outputs_dir / 'models').iterdir():
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
    plt.savefig(outputs_dir / 'all_models_comparison.png', dpi=300)
    plt.close()

    all_models_test_results_df.set_index('model_name', inplace=True)
    max_indices = all_models_test_results_df.idxmax()
    all_models_test_results_df.loc['best_model'] = max_indices
    all_models_test_results_df.to_csv(outputs_dir / 'all_test_results.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the analysis of the experiment results')
    parser.add_argument('--cpus', type=int, default=N_JOBS, help='Number of CPUs to use')
    args = parser.parse_args()
    main(args.cpus)
