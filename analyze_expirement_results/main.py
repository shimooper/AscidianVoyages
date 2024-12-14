import os
import pandas as pd

from utils import OUTPUTS_DIR, STRATIFY_TRAIN_TEST_SPLIT, RANDOM_STATE, METRIC_TO_CHOOSE_BEST_MODEL_PARAMS, \
    INCLUDE_SUSPECTED_ROUTES_PARTS, INCLUDE_CONTROL_ROUTES
from preprocess import permanent_preprocess_data, preprocess_data
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
    processed_df = permanent_preprocess_data()

    flags = []
    for include_control_flag in INCLUDE_CONTROL_ROUTES:
        for include_suspected_flag in INCLUDE_SUSPECTED_ROUTES_PARTS:
            for stratify_flag in STRATIFY_TRAIN_TEST_SPLIT:
                for random_state in RANDOM_STATE:
                    for metric in METRIC_TO_CHOOSE_BEST_MODEL_PARAMS:
                        flags.append((include_control_flag, include_suspected_flag, stratify_flag, random_state, metric))

    model_id = 0
    for include_control_flag, include_suspected_flag, stratify_flag, random_state, metric in flags:
        outputs_dir = OUTPUTS_DIR / f'outputs_includeControl_{include_control_flag}_includeSuspected_{include_suspected_flag}_stratify_{stratify_flag}_rs_{random_state}_metric_{metric}'
        preprocess_data(outputs_dir, processed_df, include_control_flag, include_suspected_flag, stratify_flag, random_state)

        plot_timelines(outputs_dir)

        for model_class in MODEL_CLASSES:
            model_instance = model_class(outputs_dir, metric, random_state, model_id)
            model_instance.run_analysis()
            model_id += 1

        aggregate_test_metrics(outputs_dir)


def aggregate_test_metrics(outputs_dir):
    all_models_test_results = {}

    for model_dir_name in os.listdir(outputs_dir / 'models'):
        model_test_results_path = (outputs_dir / 'models' / model_dir_name / 'test_outputs' /
                                   'best_classifier_test_results.csv')
        model_test_results_df = pd.read_csv(model_test_results_path)
        all_models_test_results[model_dir_name] = model_test_results_df.loc[0]

    all_models_test_results_df = pd.DataFrame.from_dict(all_models_test_results, orient='index')
    all_models_test_results_df.index.name = 'model_name'
    max_indices = all_models_test_results_df.idxmax()
    all_models_test_results_df.loc['best_model'] = max_indices
    all_models_test_results_df.to_csv(outputs_dir / 'all_test_results.csv')


if __name__ == "__main__":
    main()
