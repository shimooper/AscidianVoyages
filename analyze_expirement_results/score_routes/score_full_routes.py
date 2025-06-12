import math
from pathlib import Path
import pandas as pd
import argparse
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent

from analyze_expirement_results.utils import get_column_groups_sorted, convert_columns_to_int, convert_data_to_tensor_for_rnn
from analyze_expirement_results.model_lstm import LSTMModel
from analyze_expirement_results.model import ScikitModel
from analyze_expirement_results.configuration import Config

INTERVAL_LENGTH = 4
OUTPUTS_DIR_PATH = PROJECT_ROOT_DIR / 'analyze_expirement_results' / 'outputs_cv' / 'configuration_2'
CONFIG_PATH = OUTPUTS_DIR_PATH / 'config.csv'
MODELS_PATH = OUTPUTS_DIR_PATH / 'models' / f'{INTERVAL_LENGTH}_day_interval' / 'train_outputs'
RANDOM_FOREST_MODEL_PATH = MODELS_PATH / 'random_forest_classifier' / 'best_RandomForestClassifier.pkl'
LSTM_MODEL_PATH = MODELS_PATH / 'lstm_classifier' / 'final_train' / 'checkpoints' / 'best_model-epoch-epoch=2.ckpt'
LSTM_SCALER_PATH = MODELS_PATH / 'lstm_classifier' / 'final_train' / 'scaler.pkl'


def calc_aggregated_routes_score(routes_df, model_path):
    route_min_survival_probabilities = []
    route_max_survival_probabilities = []
    route_mean_survival_probabilities = []
    route_log_multiply_survival_probabilities = []
    route_died = []

    config = Config.from_csv(CONFIG_PATH)
    model = ScikitModel(config, INTERVAL_LENGTH)

    lived_columns, _, _ = get_column_groups_sorted(routes_df)

    for row_index, row in routes_df.iterrows():
        y = pd.notna(row['dying_day'])
        route_died.append(y)

        days_data = model.convert_route_to_intervals(row, lived_columns)
        days_df = pd.DataFrame(days_data, columns=[*[f'Temperature {i}' for i in range(1, model.interval_length + 1)],
                                                   *[f'Salinity {i}' for i in range(1, model.interval_length + 1)],
                                                   *[f'Time {i}' for i in range(1, model.interval_length + 1)],
                                                   'death']).drop(columns=['death'])
        convert_columns_to_int(days_df)

        y_survival_probabilities = run_model_inference(model, days_df, model_path)
        route_min_survival_probabilities.append(y_survival_probabilities.min())
        route_max_survival_probabilities.append(y_survival_probabilities.max())
        route_mean_survival_probabilities.append(y_survival_probabilities.mean())

        epsilon = 1e-10
        route_log_multiply_survival_probabilities.append(sum(math.log(p + epsilon) for p in y_survival_probabilities))

    routes_df['death'] = route_died
    routes_df['min'] = route_min_survival_probabilities
    routes_df['max'] = route_max_survival_probabilities
    routes_df['mean'] = route_mean_survival_probabilities
    routes_df['log_multiply'] = route_log_multiply_survival_probabilities

    return routes_df[['Season', 'Name', 'Replicate', 'death', 'min', 'max', 'mean', 'log_multiply']]


def run_model_inference(model, four_days_df, trained_model_path):
    if trained_model_path.suffix == '.pkl':
        four_days_features = model.convert_X_to_features(four_days_df)
        trained_classifier = joblib.load(trained_model_path)
        y_survival_probabilities = trained_classifier.predict_proba(four_days_features)[:, 0]
    elif trained_model_path.suffix == '.ckpt':
        device = torch.device('cpu')

        scaler = joblib.load(LSTM_SCALER_PATH)
        four_days_scaled = pd.DataFrame(scaler.transform(four_days_df), columns=four_days_df.columns, index=four_days_df.index)
        four_days_scaled_tensor = convert_data_to_tensor_for_rnn(four_days_scaled, device)
        dataset = TensorDataset(four_days_scaled_tensor)
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        final_trained_model = LSTMModel.load_from_checkpoint(trained_model_path)
        final_trained_model.to(device)
        final_trained_model.eval()
        with torch.no_grad():
            y_pred_probs = torch.cat([final_trained_model(x[0]) for x in data_loader]).cpu().numpy().flatten()

        y_survival_probabilities = 1 - y_pred_probs
    else:
        raise ValueError(f"Unsupported model file type: {trained_model_path.suffix}")

    return y_survival_probabilities


def score_routes(routes_train_path: Path, routes_test_path: Path, model_path: Path,  output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate aggregated routes scores
    routes_train_df = pd.read_csv(routes_train_path)
    routes_test_df = pd.read_csv(routes_test_path)

    routes_agg_train_df = calc_aggregated_routes_score(routes_train_df, model_path)
    routes_agg_test_df = calc_aggregated_routes_score(routes_test_df, model_path)

    routes_agg_train_df.to_csv(output_dir / 'routes_train_agg.csv', index=False)
    routes_agg_test_df.to_csv(output_dir / 'routes_test_agg.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data for scoring routes')
    parser.add_argument('--routes_train_path', type=Path,
                        help='Path to the experiment data', default=Path('full_routes') / 'actual_routes_augmented_train.csv')
    parser.add_argument('--routes_test_path', type=Path,
                        help='Path to the experiment data', default=Path('full_routes') / 'actual_routes_augmented_test.csv')
    parser.add_argument('--output_dir', type=Path, default=Path('full_routes_aggregated_scores'))
    args = parser.parse_args()

    score_routes(args.routes_train_path, args.routes_test_path, RANDOM_FOREST_MODEL_PATH, args.output_dir)
