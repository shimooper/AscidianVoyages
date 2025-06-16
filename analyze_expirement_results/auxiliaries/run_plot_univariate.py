import pandas as pd
import sys
from pathlib import Path

from analyze_expirement_results.configuration import Config
from analyze_expirement_results.model import Model

OUTPUTS_PATH = Path(r"C:\repos\GoogleShips\analyze_expirement_results\outputs_cv_test_25\configuration_0")
DATA_PATH = OUTPUTS_PATH / 'models' / '4_day_interval' / 'data'

def main():
    config = Config.from_csv(OUTPUTS_PATH / 'config.csv')
    config.models_dir_path = OUTPUTS_PATH / 'models'
    model = Model(config, 4)

    train_df = pd.read_csv(DATA_PATH / 'train.csv')
    test_df = pd.read_csv(DATA_PATH / 'test.csv')
    train_features_df = pd.read_csv(DATA_PATH / 'Xs_train_features.csv')
    test_features_df = pd.read_csv(DATA_PATH / 'Xs_test_features.csv')

    model.plot_univariate_features_with_respect_to_label(train_features_df, test_features_df, train_df['death'], test_df['death'])


if __name__ == "__main__":
    main()
