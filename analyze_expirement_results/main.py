import os
import pandas as pd

from utils import OUTPUTS_DIR, OUTPUTS_INCLUDE_SUSPECTED_DIR, OUTPUTS_EXCLUDE_SUSPECTED_DIR
from preprocess import preprocess_data
from routes_visualization import plot_timelines
from one_day_only_temperature_model import OneDayOnlyTemperatureModel
from one_day_only_salinity_model import OneDayOnlySalinityModel
from one_day_model import OneDayModel
from two_day_model import TwoDayModel
from one_week_model import OneWeekModel

MODEL_CLASSES = [
        OneDayOnlyTemperatureModel,
        OneDayOnlySalinityModel,
        OneDayModel,
        TwoDayModel,
        OneWeekModel
]


def main():
    preprocess_data()

    plot_timelines(OUTPUTS_INCLUDE_SUSPECTED_DIR)
    plot_timelines(OUTPUTS_EXCLUDE_SUSPECTED_DIR)

    model_instances = []
    for model_class in MODEL_CLASSES:
        model_include_suspected = model_class(OUTPUTS_INCLUDE_SUSPECTED_DIR)
        model_exclude_suspected = model_class(OUTPUTS_EXCLUDE_SUSPECTED_DIR)
        model_instances.extend([
            model_include_suspected,
            model_exclude_suspected
        ])

    for model_instance in model_instances:
        model_instance.run_analysis()

    aggregate_test_metrics()


def aggregate_test_metrics():
    all_models_test_results = {}

    for model_dir_name in os.listdir(OUTPUTS_INCLUDE_SUSPECTED_DIR / 'models'):
        model_test_results_path = (OUTPUTS_INCLUDE_SUSPECTED_DIR / 'models' / model_dir_name / 'test_outputs' /
                                   'best_classifier_test_results.csv')
        model_test_results_df = pd.read_csv(model_test_results_path)
        all_models_test_results[f'{model_dir_name}_include_suspected'] = model_test_results_df.loc[0]

    for model_dir_name in os.listdir(OUTPUTS_EXCLUDE_SUSPECTED_DIR / 'models'):
        model_test_results_path = (OUTPUTS_EXCLUDE_SUSPECTED_DIR / 'models' / model_dir_name / 'test_outputs' /
                                   'best_classifier_test_results.csv')
        model_test_results_df = pd.read_csv(model_test_results_path)
        all_models_test_results[f'{model_dir_name}_exclude_suspected'] = model_test_results_df.loc[0]

    all_models_test_results_df = pd.DataFrame.from_dict(all_models_test_results, orient='index')
    all_models_test_results_df.index.name = 'model_name'
    all_models_test_results_df.to_csv(OUTPUTS_DIR / 'all_test_results.csv')


if __name__ == "__main__":
    main()
