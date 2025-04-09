from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_DIR))

from configuration import Config
from model import ScikitModel

CONFIG_PATH = Path("/home/ai_center/ai_users/yairshimony/ships/analyze_expirement_results/outputs_const_train_val_split/configuration_0/config.csv")
LSTM_MODEL_PATH = "/home/ai_center/ai_users/yairshimony/ships/analyze_expirement_results/outputs_const_train_val_split/configuration_0/models/days_to_consider_4/train_outputs/lstm_classifier/hidden_size_8_num_layers_1_lr_0.01_batch_size_16/checkpoints/best_model-epoch-epoch=15.ckpt"


def main():
    config = Config.from_csv(CONFIG_PATH)
    model_instance = ScikitModel(config, 3, 4)
    Xs_train, Ys_train, Xs_test, Ys_test = model_instance.create_model_data()
    model_instance.test_on_test_data(LSTM_MODEL_PATH, Xs_test, Ys_test)


if __name__ == "__main__":
    main()
