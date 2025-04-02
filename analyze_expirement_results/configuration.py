from pathlib import Path
import argparse
from dataclasses import dataclass, asdict

from sklearn.metrics import make_scorer, matthews_corrcoef
import pandas as pd


DEBUG_MODE = False

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'data'
DATA_PATH = DATA_DIR / 'Final_Data_Voyages.xlsx'

INCLUDE_CONTROL_ROUTES = [True]
INCLUDE_SUSPECTED_ROUTES_PARTS = [False]
STRATIFY_TRAIN_TEST_SPLIT = [False]
RANDOM_STATE = [5]
NUMBER_OF_FUTURE_DAYS_TO_CONSIDER_DEATH = [0]
METRIC_TO_CHOOSE_BEST_MODEL_HYPER_PARAMS = ['mcc', 'f1', 'auprc'] if not DEBUG_MODE else ['mcc']
TEST_SET_SIZE = 0.25
VALIDATION_SET_SIZE = 0.2  # relevant only in case of neural networks
NUMBER_OF_DAYS_TO_CONSIDER = [1, 2, 3, 4]
DOWNSAMPLE_MAJORITY_CLASS = [False, True] if not DEBUG_MODE else [False]

METRIC_NAME_TO_SKLEARN_SCORER = {'mcc': make_scorer(matthews_corrcoef), 'f1': 'f1', 'auprc': 'average_precision'}

DEFAULT_OPTUNA_NUMBER_OF_TRIALS = 1000 if not DEBUG_MODE else 10


@dataclass
class Config:
    include_control_routes: bool
    include_suspected_routes: bool
    number_of_future_days_to_consider_death: int
    stratify: bool
    random_state: int
    metric: str
    configuration_id: int
    cpus: int
    outputs_dir_path: Path
    data_dir_path: Path
    models_dir_path: Path
    do_feature_selection: bool
    train_with_optuna: bool
    optuna_number_of_trials: int
    test_set_size: float = TEST_SET_SIZE
    nn_validation_set_size: float = VALIDATION_SET_SIZE
    nn_max_epochs: int = 100
    downsample_majority_class: bool = False

    def to_csv(self, path: Path):
        config_df = pd.DataFrame(list(asdict(self).items()), columns=['key', 'value'])
        config_df.to_csv(path, index=False)

    @classmethod
    def from_csv(cls, path: Path):
        config_df = pd.read_csv(path, na_filter=False)
        data_dict = dict(zip(config_df["key"], config_df["value"]))

        typed_data = {}
        for key, value in data_dict.items():
            if key in cls.__annotations__:
                expected_type = cls.__annotations__[key]

                if expected_type == bool:
                    typed_data[key] = str_to_bool(value)
                else:
                    typed_data[key] = expected_type(value)
            else:
                raise ValueError(f'key {key} not in {cls} annotations')

        return cls(**typed_data)


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
