
from preprocess import preprocess_data
from routes_visualization import plot_timelines
from one_day_only_temperature_model import OneDayOnlyTemperatureModel
from one_day_only_salinity_model import OneDayOnlySalinityModel
from one_day_model import OneDayModel
from two_day_model import TwoDayModel
from one_week_model import OneWeekModel
from utils import (
    FULL_INCLUDE_SUSPECTED_PATH, TRAIN_INCLUDE_SUSPECTED_PATH, TEST_INCLUDE_SUSPECTED_PATH, OUTPUTS_INCLUDE_SUSPECTED_DIR, \
    FULL_EXCLUDE_SUSPECTED_PATH, TRAIN_EXCLUDE_SUSPECTED_PATH, TEST_EXCLUDE_SUSPECTED_PATH, OUTPUTS_EXCLUDE_SUSPECTED_DIR
)

MODEL_CLASSES = [
        OneDayOnlyTemperatureModel,
        OneDayOnlySalinityModel,
        OneDayModel,
        TwoDayModel,
        OneWeekModel
]


def main():
    preprocess_data()

    plot_timelines(FULL_INCLUDE_SUSPECTED_PATH, OUTPUTS_INCLUDE_SUSPECTED_DIR)
    plot_timelines(FULL_EXCLUDE_SUSPECTED_PATH, OUTPUTS_EXCLUDE_SUSPECTED_DIR)

    model_instances = []
    for model_class in MODEL_CLASSES:
        model_include_suspected = model_class(TRAIN_INCLUDE_SUSPECTED_PATH, TEST_INCLUDE_SUSPECTED_PATH,
                                              OUTPUTS_INCLUDE_SUSPECTED_DIR)
        model_exclude_suspected = model_class(TRAIN_EXCLUDE_SUSPECTED_PATH, TEST_EXCLUDE_SUSPECTED_PATH,
                                              OUTPUTS_EXCLUDE_SUSPECTED_DIR)
        model_instances.extend([
            model_include_suspected,
            model_exclude_suspected
        ])

    for model_instance in model_instances:
        model_instance.run_analysis()


if __name__ == "__main__":
    main()
