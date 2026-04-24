import math
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from analyze_expirement_results.utils import get_column_groups_sorted, convert_columns_to_int, convert_data_to_tensor_for_rnn
from analyze_expirement_results.model_lstm import LSTMModel
from analyze_expirement_results.model import ScikitModel
from analyze_expirement_results.configuration import Config


def calc_aggregated_routes_score(config_path, interval_length, routes_df, model_inference_method, model_inference_args):
    route_min_survival_probabilities = []
    route_max_survival_probabilities = []
    route_mean_survival_probabilities = []
    route_log_multiply_survival_probabilities = []
    route_died = []

    config = Config.from_csv(config_path)
    model = ScikitModel(config, interval_length)

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

        y_survival_probabilities = model_inference_method(days_df, *model_inference_args)
        route_min_survival_probabilities.append(y_survival_probabilities.min())
        route_max_survival_probabilities.append(y_survival_probabilities.max())
        route_mean_survival_probabilities.append(y_survival_probabilities.mean())

        epsilon = 1e-10
        route_log_multiply_survival_probabilities.append(sum(math.log(p + epsilon) for p in y_survival_probabilities))

    routes_df['death'] = route_died
    routes_df['min'] = np.around(route_min_survival_probabilities, 4)
    routes_df['max'] = np.around(route_max_survival_probabilities, 4)
    routes_df['mean'] = np.around(route_mean_survival_probabilities, 4)
    routes_df['log_multiply'] = np.around(route_log_multiply_survival_probabilities, 4)

    return routes_df[['Season', 'Name', 'Replicate', 'death', 'min', 'max', 'mean', 'log_multiply']]


def run_sklearn_model_inference(data_df, trained_model_path):
    days_features = ScikitModel.convert_X_to_features(data_df)
    trained_classifier = joblib.load(trained_model_path)
    y_survival_probabilities = trained_classifier.predict_proba(days_features)[:, 0]
    return y_survival_probabilities


def run_nn_model_inference(data_df, trained_model_path, trained_scaler_path):
    device = torch.device('cpu')

    scaler = joblib.load(trained_scaler_path)
    four_days_scaled = pd.DataFrame(scaler.transform(data_df), columns=data_df.columns, index=data_df.index)
    four_days_scaled_tensor = convert_data_to_tensor_for_rnn(four_days_scaled, device)
    dataset = TensorDataset(four_days_scaled_tensor)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
    final_trained_model = LSTMModel.load_from_checkpoint(trained_model_path)
    final_trained_model.to(device)
    final_trained_model.eval()
    with torch.no_grad():
        y_pred_probs = torch.cat([final_trained_model(x[0]) for x in data_loader]).cpu().numpy().flatten()

    y_survival_probabilities = 1 - y_pred_probs
    return y_survival_probabilities


def score_routes(base_dir: Path, interval_length, model_inference_method, model_inference_args):
    routes_train_path = base_dir / 'score_routes' / 'full_routes' / 'actual_routes_augmented_train.csv'
    routes_test_path = base_dir / 'score_routes' / 'full_routes' / 'actual_routes_augmented_test.csv'
    routes_train_df = pd.read_csv(routes_train_path)
    routes_test_df = pd.read_csv(routes_test_path)

    # Calculate aggregated routes scores
    routes_agg_train_df = calc_aggregated_routes_score(base_dir / 'config.csv', interval_length, routes_train_df, model_inference_method, model_inference_args)
    routes_agg_test_df = calc_aggregated_routes_score(base_dir / 'config.csv', interval_length, routes_test_df, model_inference_method, model_inference_args)

    output_dir = base_dir / 'score_routes' / 'full_routes_aggregated_scores'
    output_dir.mkdir(parents=True, exist_ok=True)

    routes_agg_train_df.to_csv(output_dir / 'routes_train_agg.csv', index=False)
    routes_agg_test_df.to_csv(output_dir / 'routes_test_agg.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data for scoring routes')
    parser.add_argument('--base_dir', type=Path, required=True)
    parser.add_argument('--best_interval_length', type=int, default=3, help='Best interval length to use for scoring routes')
    parser.add_argument('--best_model', type=str, choices=['random_forest', 'lstm'], default='random_forest', help='Best model to use for scoring routes')
    parser.add_argument('--random_forest_model_path', type=Path, default=Path('random_forest_classifier') / 'best_RandomForestClassifier.pkl')
    parser.add_argument('--lstm_model_path', type=Path, default=Path('lstm_classifier') / 'final_train' / 'checkpoints' / 'best_model-epoch-epoch=2.ckpt')
    parser.add_argument('--lstm_scaler_path', type=Path, default=Path('lstm_classifier') / 'final_train' / 'scaler.pkl')
    args = parser.parse_args()

    models_path = args.base_dir / 'models' / f'{args.best_interval_length}_day_interval' / 'train_outputs'
    if args.best_model == 'random_forest':
        model_inference_method = run_sklearn_model_inference
        model_inference_args = [models_path / args.random_forest_model_path]
    elif args.best_model == 'lstm':
        model_inference_method = run_nn_model_inference
        model_inference_args = [models_path / args.lstm_model_path, models_path / args.lstm_scaler_path]
    else:
        raise ValueError(f"Unsupported model type: {args.best_model}")

    score_routes(args.base_dir, args.best_interval_length, model_inference_method, model_inference_args)
