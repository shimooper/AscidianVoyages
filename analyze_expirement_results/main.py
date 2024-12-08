import os
import pandas as pd

from utils import OUTPUTS_DIR, STRATIFY_TRAIN_TEST_SPLIT, RANDOM_STATE, METRIC_TO_CHOOSE_BEST_MODEL_PARAMS
from preprocess import preprocess_data
from routes_visualization import plot_timelines
from one_day_only_temperature_model import OneDayOnlyTemperatureModel
from one_day_only_salinity_model import OneDayOnlySalinityModel
from one_day_model import OneDayModel
from two_day_model import TwoDayModel
from four_days_model import FourDaysModel
from one_week_model import OneWeekModel

MODEL_CLASSES = [
        OneDayOnlyTemperatureModel,
        OneDayOnlySalinityModel,
        OneDayModel,
        TwoDayModel,
        FourDaysModel,
        OneWeekModel
]


def main():
    flags = []
    for stratify_flag in STRATIFY_TRAIN_TEST_SPLIT:
        for random_state in RANDOM_STATE:
            for metric in METRIC_TO_CHOOSE_BEST_MODEL_PARAMS:
                flags.append((stratify_flag, random_state, metric))

    model_id = 0
    for stratify_flag, random_state, metric in flags:
        outputs_dir = OUTPUTS_DIR / f'outputs_stratify_{str(stratify_flag).lower()}_rs_{random_state}_metric_{metric}'
        outputs_include_suspected_dir = outputs_dir / 'include_suspected'
        outputs_exclude_suspected_dir = outputs_dir / 'exclude_suspected'
        os.makedirs(outputs_include_suspected_dir, exist_ok=True)
        os.makedirs(outputs_exclude_suspected_dir, exist_ok=True)

        preprocess_data(outputs_dir, outputs_include_suspected_dir, outputs_exclude_suspected_dir, stratify_flag, random_state)

        plot_timelines(outputs_include_suspected_dir)
        plot_timelines(outputs_exclude_suspected_dir)

        model_instances = []
        for model_class in MODEL_CLASSES:
            model_include_suspected = model_class(outputs_include_suspected_dir, metric, random_state, model_id)
            model_id += 1
            model_exclude_suspected = model_class(outputs_exclude_suspected_dir, metric, random_state, model_id)
            model_id += 1
            model_instances.extend([
                model_include_suspected,
                model_exclude_suspected
            ])

        for model_instance in model_instances:
            model_instance.run_analysis()

        aggregate_test_metrics(outputs_dir, outputs_include_suspected_dir, outputs_exclude_suspected_dir)


def aggregate_test_metrics(outputs_dir, outputs_include_suspected_dir, outputs_exclude_suspected_dir):
    all_models_test_results = {}

    for model_dir_name in os.listdir(outputs_include_suspected_dir / 'models'):
        model_test_results_path = (outputs_include_suspected_dir / 'models' / model_dir_name / 'test_outputs' /
                                   'best_classifier_test_results.csv')
        model_test_results_df = pd.read_csv(model_test_results_path)
        all_models_test_results[f'{model_dir_name}_include_suspected'] = model_test_results_df.loc[0]

    for model_dir_name in os.listdir(outputs_exclude_suspected_dir / 'models'):
        model_test_results_path = (outputs_exclude_suspected_dir / 'models' / model_dir_name / 'test_outputs' /
                                   'best_classifier_test_results.csv')
        model_test_results_df = pd.read_csv(model_test_results_path)
        all_models_test_results[f'{model_dir_name}_exclude_suspected'] = model_test_results_df.loc[0]

    all_models_test_results_df = pd.DataFrame.from_dict(all_models_test_results, orient='index')
    all_models_test_results_df.index.name = 'model_name'
    max_indices = all_models_test_results_df.idxmax()
    all_models_test_results_df.loc['best_model'] = max_indices
    all_models_test_results_df.to_csv(outputs_dir / 'all_test_results.csv')


if __name__ == "__main__":
    main()
