import os
import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt

from utils import OUTPUTS_DIR, STRATIFY_TRAIN_TEST_SPLIT, RANDOM_STATE, METRIC_TO_CHOOSE_BEST_MODEL_HYPER_PARAMS, \
    INCLUDE_SUSPECTED_ROUTES_PARTS, INCLUDE_CONTROL_ROUTES, NUMBER_OF_FUTURE_DAYS_TO_CONSIDER_DEATH, N_JOBS
from preprocess import permanent_preprocess_data, preprocess_data
from routes_visualization import plot_timelines
from one_day_model import OneDayModel
from one_day_only_temperature_model import OneDayOnlyTemperatureModel
from one_day_only_salinity_model import OneDayOnlySalinityModel
from two_day_model import TwoDayModel
from two_day_only_temperature_model import TwoDayOnlyTemperatureModel
from two_day_only_salinity_model import TwoDayOnlySalinityModel
from four_days_model import FourDaysModel
from four_days_only_temperature_model import FourDaysOnlyTemperatureModel
from four_days_only_salinity_model import FourDaysOnlySalinityModel

MODEL_CLASSES = [
        OneDayModel,
        OneDayOnlyTemperatureModel,
        OneDayOnlySalinityModel,
        TwoDayModel,
        TwoDayOnlyTemperatureModel,
        TwoDayOnlySalinityModel,
        FourDaysModel,
        FourDaysOnlyTemperatureModel,
        FourDaysOnlySalinityModel
]


def main(cpus):
    processed_df = permanent_preprocess_data()

    flags = []
    for include_control_flag in INCLUDE_CONTROL_ROUTES:
        for include_suspected_flag in INCLUDE_SUSPECTED_ROUTES_PARTS:
            for stratify_flag in STRATIFY_TRAIN_TEST_SPLIT:
                for number_of_future_days in NUMBER_OF_FUTURE_DAYS_TO_CONSIDER_DEATH:
                    for random_state in RANDOM_STATE:
                        for metric in METRIC_TO_CHOOSE_BEST_MODEL_HYPER_PARAMS:
                            flags.append((include_control_flag, include_suspected_flag, stratify_flag,
                                          number_of_future_days, random_state, metric))

    configuration_id = 0
    for include_control_flag, include_suspected_flag, stratify_flag, number_of_future_days, random_state, metric in flags:
        outputs_dir = OUTPUTS_DIR / f'configuration_{configuration_id}'
        os.makedirs(outputs_dir, exist_ok=True)

        configuration_df = pd.DataFrame({
            'include_control_flag': [include_control_flag],
            'include_suspected_flag': [include_suspected_flag],
            'stratify_flag': [stratify_flag],
            'number_of_future_days': [number_of_future_days],
            'random_state': [random_state],
            'metric': [metric]
        })
        configuration_df.to_csv(outputs_dir / 'configuration.csv', index=False)

        preprocess_data(outputs_dir, processed_df, include_control_flag, include_suspected_flag, stratify_flag,
                        random_state, configuration_id)

        plot_timelines(outputs_dir)

        model_id = 0
        for model_class in MODEL_CLASSES:
            model_instance = model_class(outputs_dir, metric, random_state, number_of_future_days,
                                         f'{configuration_id}_{model_id}')
            model_instance.run_analysis(cpus)
            model_id += 1

        aggregate_test_metrics_of_one_configuration(outputs_dir)
        configuration_id += 1


def aggregate_test_metrics_of_one_configuration(outputs_dir):
    all_models_test_results = {}

    for model_dir_name in os.listdir(outputs_dir / 'models'):
        model_test_results_path = (outputs_dir / 'models' / model_dir_name / 'test_outputs' /
                                   'best_classifier_test_results.csv')
        model_test_results_df = pd.read_csv(model_test_results_path)
        all_models_test_results[model_dir_name] = model_test_results_df.loc[0]

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
