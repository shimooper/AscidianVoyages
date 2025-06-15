from pathlib import Path
from dataclasses import dataclass, asdict
import platform

from sklearn.metrics import make_scorer, matthews_corrcoef
import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from analyze_expirement_results.utils import str_to_bool

if platform.system() == "Linux":
    DEBUG_MODE = False
elif platform.system() == "Windows":
    DEBUG_MODE = True
else:
    raise Exception(f"Unsupported platform: {platform.system()}")

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / 'data'
DATA_PATH = DATA_DIR / 'Final_Data_Voyages.xlsx'
PROCESSED_DATA_DIR = DATA_DIR / 'preprocess'
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / 'Final_Data_Voyages_Processed_30.csv'

INCLUDE_CONTROL_ROUTES = [True]
INCLUDE_SUSPECTED_ROUTES_PARTS = [False] if not DEBUG_MODE else [True, False]
STRATIFY_TRAIN_TEST_SPLIT = [True, False] if not DEBUG_MODE else [True]
# RANDOM_STATE = [0, 42, 123, 99, 2025] if not DEBUG_MODE else [0]
RANDOM_STATE = [0] if not DEBUG_MODE else [0]
NUMBER_OF_FUTURE_DAYS_TO_CONSIDER_DEATH = [0] if not DEBUG_MODE else [0]
# METRIC_TO_CHOOSE_BEST_MODEL_HYPER_PARAMS = ['f1', 'mcc', 'auprc'] if not DEBUG_MODE else ['f1']
METRIC_TO_CHOOSE_BEST_MODEL_HYPER_PARAMS = ['mcc'] if not DEBUG_MODE else ['f1']
TEST_SET_SIZE = 0.25
INTERVAL_LENGTH = [1, 2, 3, 4] if not DEBUG_MODE else [3, 4]
BALANCE_CLASSES_IN_TRAINING = [False, True] if not DEBUG_MODE else [False]
NN_MAX_EPOCHS = 15 if not DEBUG_MODE else 1
MAX_CLASS_RATIO = 0.25  # Relevant only if BALANCE_CLASSES_IN_TRAINING is True

METRIC_NAME_TO_SKLEARN_SCORER = {'mcc': make_scorer(matthews_corrcoef), 'f1': 'f1', 'auprc': 'average_precision'}

DEFAULT_OPTUNA_NUMBER_OF_TRIALS = 1000 if not DEBUG_MODE else 10


@dataclass
class Config:
    error_file_path: Path
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
    train_with_optuna: bool = False
    optuna_number_of_trials: int = DEFAULT_OPTUNA_NUMBER_OF_TRIALS
    test_set_size: float = TEST_SET_SIZE
    nn_max_epochs: int = NN_MAX_EPOCHS
    balance_classes: bool = False
    max_classes_ratio: float = MAX_CLASS_RATIO  # Relevant only if balance_classes is True
    run_lstm_configurations_in_parallel: bool = False

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
