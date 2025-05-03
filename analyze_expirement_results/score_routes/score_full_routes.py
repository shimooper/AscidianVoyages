import math
from pathlib import Path
import pandas as pd
import argparse
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent.parent

from analyze_expirement_results.utils import DAYS_DESCRIPTIONS, convert_columns_to_int, convert_features_df_to_tensor_for_rnn, setup_logger
from analyze_expirement_results.model_lstm import LSTMModel

NUMBER_OF_DAYS_TO_CONSIDER = 4


def calc_aggregated_routes_score(routes_df, model_path):
    route_min_survival_probabilities = []
    route_max_survival_probabilities = []
    route_mean_survival_probabilities = []
    route_multiply_survival_probabilities = []
    route_died = []

    for row_index, row in routes_df.iterrows():
        y = pd.notna(row['dying_day'])
        route_died.append(y)

        days_data = []
        for col_day in range(30, 0, -1):
            temperature_columns = {f'{DAYS_DESCRIPTIONS[i]} Temperature': row[f'Temp {col_day - i}'] for i in range(NUMBER_OF_DAYS_TO_CONSIDER)}
            salinity_columns = {f'{DAYS_DESCRIPTIONS[i]} Salinity': row[f'Salinity {col_day - i}'] for i in range(NUMBER_OF_DAYS_TO_CONSIDER)}

            new_row = {
                **temperature_columns, **salinity_columns,
            }
            days_data.append(new_row)

        days_df = pd.DataFrame(days_data)
        convert_columns_to_int(days_df)

        y_survival_probabilities = run_model_inference(days_df, model_path)
        route_min_survival_probabilities.append(y_survival_probabilities.min())
        route_max_survival_probabilities.append(y_survival_probabilities.max())
        route_mean_survival_probabilities.append(y_survival_probabilities.mean())

        epsilon = 1e-10
        route_multiply_survival_probabilities.append(sum(math.log(p + epsilon) for p in y_survival_probabilities))

    routes_df['death'] = route_died
    routes_df['min'] = route_min_survival_probabilities
    routes_df['max'] = route_max_survival_probabilities
    routes_df['mean'] = route_mean_survival_probabilities
    routes_df['multiply'] = route_multiply_survival_probabilities

    return routes_df[['Season', 'Name', 'death', 'min', 'max', 'mean', 'multiply']]


def run_model_inference(four_days_df, model_path):
    if model_path.suffix == '.pkl':
        model = joblib.load(model_path)
        y_survival_probabilities = model.predict_proba(four_days_df)[:, 0]
    elif model_path.suffix == '.ckpt':
        device = torch.device('cpu')
        X_tensor = convert_features_df_to_tensor_for_rnn(four_days_df, device)

        dataset = TensorDataset(X_tensor)
        data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
        best_model = LSTMModel.load_from_checkpoint(model_path)
        best_model.to(device)
        best_model.eval()
        with torch.no_grad():
            y_pred_probs = torch.cat([best_model(x[0]) for x in data_loader]).cpu().numpy().flatten()
        y_survival_probabilities = 1 - y_pred_probs
    else:
        raise ValueError(f"Unsupported model file type: {model_path.suffix}")

    return y_survival_probabilities


def score_routes(routes_train_path: Path, routes_test_path: Path, model_path: Path,  output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = output_dir / 'score_routes.log'
    logger = setup_logger(log_file_path, 'score_routes')

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
    # parser.add_argument('--model_path', type=Path, default=Path('actual_routes') / 'best_model-epoch-epoch=13.ckpt')
    parser.add_argument('--model_path', type=Path, default=Path('actual_routes') / 'best_RandomForestClassifier.pkl')
    parser.add_argument('--output_dir', type=Path, default=Path('full_routes_aggregated_scores'),)
    args = parser.parse_args()

    score_routes(args.routes_train_path, args.routes_test_path, args.model_path, args.output_dir)
