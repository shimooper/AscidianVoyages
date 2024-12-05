
from utils import TRAIN_INCLUDE_SUSPECTED_PATH, TEST_INCLUDE_SUSPECTED_PATH, OUTPUTS_INCLUDE_SUSPECTED_DIR,\
    TRAIN_EXCLUDE_SUSPECTED_PATH, TEST_EXCLUDE_SUSPECTED_PATH, OUTPUTS_EXCLUDE_SUSPECTED_DIR
from one_week_model import OneWeekModel


def main():


    one_week_model_include_suspected = OneWeekModel(TRAIN_INCLUDE_SUSPECTED_PATH, TEST_INCLUDE_SUSPECTED_PATH,
                                                    OUTPUTS_INCLUDE_SUSPECTED_DIR)
    one_week_model_exclude_suspected = OneWeekModel(TRAIN_EXCLUDE_SUSPECTED_PATH, TEST_EXCLUDE_SUSPECTED_PATH,
                                                    OUTPUTS_EXCLUDE_SUSPECTED_DIR)

