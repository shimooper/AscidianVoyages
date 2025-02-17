from pathlib import Path
import argparse
from dataclasses import dataclass, asdict

from sklearn.metrics import make_scorer, matthews_corrcoef
import pandas as pd


DEBUG_MODE = False
DO_FEATURE_SELECTION = False

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'data'
DATA_PATH = DATA_DIR / 'Final_Data_Voyages.xlsx'

INCLUDE_CONTROL_ROUTES = [True]
INCLUDE_SUSPECTED_ROUTES_PARTS = [True, False] if not DEBUG_MODE else [False]
STRATIFY_TRAIN_TEST_SPLIT = [True, False] if not DEBUG_MODE else [False]
RANDOM_STATE = [42]
METRIC_TO_CHOOSE_BEST_MODEL_HYPER_PARAMS = ['mcc', 'f1'] if not DEBUG_MODE else ['mcc']
NUMBER_OF_FUTURE_DAYS_TO_CONSIDER_DEATH = [0, 1, 2, 3, 4]
TEST_SET_SIZE = 0.25

METRIC_NAME_TO_SKLEARN_SCORER = {'mcc': make_scorer(matthews_corrcoef), 'f1': 'f1', 'auprc': 'average_precision'}


@dataclass
class Config:
    include_control_routes: bool
    include_suspected_routes: bool
    stratify: bool
    random_state: int
    metric: str
    configuration_id: int
    cpus: int
    outputs_dir_path: Path
    data_dir_path: Path
    models_dir_path: Path
    test_set_size: float = TEST_SET_SIZE

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
