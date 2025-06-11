import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
import lightning as L

from analyze_expirement_results.model_lstm import train_lstm_with_hyperparameters_train_val
from analyze_expirement_results.configuration import Config
from analyze_expirement_results.q_submitter_power import run_step, add_default_step_args


def train_lstm_with_hyperparameters_cv(logger, config, classifier_output_dir, hidden_size, num_layers, lr, batch_size,
                                       X_train, y_train, device, cv_splitter):
    grid_combination_dir = classifier_output_dir / f'hs_{hidden_size}_nl_{num_layers}_lr_{lr}_bs_{batch_size}'
    grid_combination_dir.mkdir(exist_ok=True, parents=True)

    logger.info(
        f"Training LSTM with hidden_size={hidden_size}, num_layers={num_layers}, lr={lr}, batch_size={batch_size}, "
        f"using {cv_splitter.n_splits}-fold cross-validation.")

    results = {'hidden_size': hidden_size, 'num_layers': num_layers, 'lr': lr, 'batch_size': batch_size}
    for fold_id, (train_idx, test_idx) in enumerate(cv_splitter.split(X_train, y_train)):
        fold_dir = grid_combination_dir / f'fold_{fold_id}'
        fold_dir.mkdir(exist_ok=True, parents=True)

        X_train_fold = X_train.iloc[train_idx]
        y_train_fold = y_train.iloc[train_idx]
        X_val_fold = X_train.iloc[test_idx]
        y_val_fold = y_train.iloc[test_idx]

        train_mcc, train_f1, train_auprc, val_mcc, val_f1, val_auprc, best_model_path = \
            train_lstm_with_hyperparameters_train_val(logger, config, fold_dir, hidden_size, num_layers, lr, batch_size,
                                                           X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                                                           device)

        results[f'split{fold_id}_train_mcc'] = train_mcc
        results[f'split{fold_id}_train_f1'] = train_f1
        results[f'split{fold_id}_train_auprc'] = train_auprc
        results[f'split{fold_id}_test_mcc'] = val_mcc
        results[f'split{fold_id}_test_f1'] = val_f1
        results[f'split{fold_id}_test_auprc'] = val_auprc

    results[f'mean_train_mcc'] = np.mean(
        [results[f'split{fold_id}_train_mcc'] for fold_id in range(cv_splitter.n_splits)])
    results[f'mean_train_f1'] = np.mean(
        [results[f'split{fold_id}_train_f1'] for fold_id in range(cv_splitter.n_splits)])
    results[f'mean_train_auprc'] = np.mean(
        [results[f'split{fold_id}_train_auprc'] for fold_id in range(cv_splitter.n_splits)])
    results[f'mean_test_mcc'] = np.mean(
        [results[f'split{fold_id}_test_mcc'] for fold_id in range(cv_splitter.n_splits)])
    results[f'mean_test_f1'] = np.mean([results[f'split{fold_id}_test_f1'] for fold_id in range(cv_splitter.n_splits)])
    results[f'mean_test_auprc'] = np.mean(
        [results[f'split{fold_id}_test_auprc'] for fold_id in range(cv_splitter.n_splits)])

    results_df = pd.Series(results)
    results_df.to_csv(grid_combination_dir / 'results.csv')

    return results


def main():
    parser = argparse.ArgumentParser(description="Train LSTM with hyperparameters using cross-validation")
    parser.add_argument('config', type=Path, help='Path to the configuration file')
    parser.add_argument('classifier_output_dir', type=Path, help='Directory to save classifier output')
    parser.add_argument('hidden_size', type=int, help='Hidden size for LSTM')
    parser.add_argument('num_layers', type=int, help='Number of layers for LSTM')
    parser.add_argument('lr', type=float, help='Learning rate for training')
    parser.add_argument('batch_size', type=int, help='Batch size for training')
    parser.add_argument('train_path', type=Path, help='Path to training data')
    add_default_step_args(parser)
    args = parser.parse_args()

    config = Config.from_csv(args.config)

    train_df = pd.read_csv(args.train_path)
    X_train = train_df.drop(columns=['death'])
    y_train = train_df['death']

    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.random_state)
    device = torch.device('cpu')
    L.seed_everything(config.random_state, workers=True)

    run_step(args, train_lstm_with_hyperparameters_cv, config,
             args.classifier_output_dir, args.hidden_size, args.num_layers, args.lr, args.batch_size, X_train, y_train,
             device, cv_splitter)


if __name__ == "__main__":
    main()
