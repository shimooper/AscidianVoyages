"""
Learning curve analysis: trains RandomForest with the best-found hyperparameters on
increasing subsets of the training data, evaluating on the full test set each time.
"""
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, average_precision_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_expirement_results.model import ScikitModel

INTERVAL_LENGTH = 3
N_SIZES = 10
RANDOM_STATE = 10


def load_best_hyperparams(results_csv_path: Path) -> dict:
    series = pd.read_csv(results_csv_path, index_col=0, header=0).iloc[:, 0]
    params = {}
    for key, val in series.items():
        if not str(key).startswith('param_'):
            continue
        param_name = key[len('param_'):]
        if pd.isna(val):
            params[param_name] = None
        elif str(val) == 'True':
            params[param_name] = True
        elif str(val) == 'False':
            params[param_name] = False
        else:
            try:
                as_float = float(val)
                params[param_name] = int(as_float) if as_float == int(as_float) else as_float
            except (ValueError, TypeError):
                params[param_name] = val
    return params


def main(base_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    model_data_dir = base_dir / 'models' / f'{INTERVAL_LENGTH}_day_interval' / 'data'
    results_csv = (base_dir / 'models' / f'{INTERVAL_LENGTH}_day_interval' / 'train_outputs'
                   / 'random_forest_classifier' / 'best_RandomForestClassifier_results.csv')

    hyperparams = load_best_hyperparams(results_csv)
    print(f"Best hyperparams: {hyperparams}")

    train_intervals = pd.read_csv(model_data_dir / 'train.csv')
    test_intervals = pd.read_csv(model_data_dir / 'test.csv')

    X_train_full = ScikitModel.convert_X_to_features(train_intervals.drop(columns=['death']))
    y_train_full = train_intervals['death'].astype(int)
    X_test = ScikitModel.convert_X_to_features(test_intervals.drop(columns=['death']))
    y_test = test_intervals['death'].astype(int)

    n_total = len(X_train_full)
    sizes = sorted(set(max(2, int(round(f * n_total))) for f in np.linspace(0.1, 1.0, N_SIZES)))

    records = []
    for size in sizes:
        if size >= n_total:
            X_sub, y_sub = X_train_full, y_train_full
        else:
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=size, random_state=RANDOM_STATE)
            idx, _ = next(splitter.split(X_train_full, y_train_full))
            X_sub = X_train_full.iloc[idx]
            y_sub = y_train_full.iloc[idx]

        clf = RandomForestClassifier(**hyperparams)
        clf.fit(X_sub, y_sub)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        record = {
            'train_size': size,
            'train_pct': round(size / n_total * 100),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auprc': average_precision_score(y_test, y_prob),
        }
        records.append(record)
        print(f"train_size={size} ({record['train_pct']}%): MCC={record['mcc']:.3f}, F1={record['f1']:.3f}, AUPRC={record['auprc']:.3f}")

    results_df = pd.DataFrame(records)
    results_df.to_csv(output_dir / 'learning_curve_results.csv', index=False)

    sns.set(style='whitegrid', context='paper')
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric, label, color in [('mcc', 'MCC', 'steelblue'), ('f1', 'F1', 'darkorange'), ('auprc', 'AUPRC', 'forestgreen')]:
        ax.plot(results_df['train_pct'], results_df[metric], marker='o', label=label, color=color)
    ax.set_xticks(results_df['train_pct'])
    ax.set_xticklabels([f'{p}%' for p in results_df['train_pct']])
    ax.set_xlabel('Training set size (% of full train set)', fontsize=18)
    ax.set_ylabel('Score', fontsize=18)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curve.png', dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved results to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Learning curve: test-set metrics vs. training set size.')
    parser.add_argument('--base_dir', type=Path,
                        default=Path(__file__).resolve().parent / 'outputs_revision' / 'configuration_0')
    args = parser.parse_args()

    main(args.base_dir, args.base_dir / 'learning_curve')
