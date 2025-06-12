from pathlib import Path
import pandas as pd
import joblib
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import matthews_corrcoef, average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import matplotlib.pyplot as plt

from analyze_expirement_results.configuration import Config

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT_DIR / 'outputs_cv' / 'configuration_2' / 'config.csv'

DEBUG = False


def run_one_config(config, routes_train_path, routes_test_path, aggregation_type, route_classifier, output_dir):
    train_df = pd.read_csv(routes_train_path)
    test_df = pd.read_csv(routes_test_path)

    # Use the aggregated score to fit a classification model
    route_classifier.fit(train_df[[aggregation_type]], train_df['death'])
    joblib.dump(route_classifier, output_dir / 'route_risk_classifier.pkl')

    all_metrics = []
    for dataset_name, dataset_df in [('train', train_df), ('test', test_df)]:
        routes_death_predictions = route_classifier.predict(dataset_df[[aggregation_type]])
        routes_death_predictions_probabilities = route_classifier.predict_proba(dataset_df[[aggregation_type]])[:, 1]
        dataset_df['death_prediction'] = routes_death_predictions
        dataset_df['death_prediction_probability'] = routes_death_predictions_probabilities

        mcc = matthews_corrcoef(dataset_df['death'], routes_death_predictions)
        auprc = average_precision_score(dataset_df['death'], routes_death_predictions_probabilities)
        f1 = f1_score(dataset_df['death'], routes_death_predictions)

        if config.metric == 'mcc':
            metrics = {'split': dataset_name, 'MCC': mcc, 'AUPRC': auprc, 'F1': f1}
        elif config.metric == 'f1':
            metrics = {'split': dataset_name, 'F1': f1, 'AUPRC': auprc, 'MCC': mcc}
        elif config.metric == 'auprc':
            metrics = {'split': dataset_name, 'AUPRC': auprc, 'MCC': mcc, 'F1': f1}
        else:
            raise ValueError(f"Unsupported metric: {config.metric}. Supported metrics are 'f1', 'mcc', and 'auprc'.")
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(output_dir / 'routes_risk_results.csv', index=False)

    # Plot
    df_melted = metrics_df.melt(id_vars='split', var_name='metric', value_name='value')
    sns.set(style="whitegrid", context="paper")
    palette = sns.color_palette("Set2", n_colors=len(metrics_df.index))
    sns.barplot(data=df_melted, x='metric', y='value', hue='split', palette=palette)
    plt.xlabel('Metric', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.legend(title='Dataset', bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.,  title_fontsize=14, fontsize=12)
    plt.tick_params(axis='both', labelsize=12)
    plt.savefig(output_dir / 'routes_risk_results.png', dpi=600, bbox_inches='tight')
    plt.close()

    # Write to file routes scores
    all_routes_df = pd.concat([train_df, test_df], ignore_index=True)
    all_routes_df.sort_values(by=['death_prediction_probability'], inplace=True)
    all_routes_df.to_csv(output_dir / 'routes_risk.csv', index=False)

    # Group replicates
    routes_grouped_df = all_routes_df.groupby(by=['Season', 'Name']).agg({
        'death_prediction_probability': 'mean',
    }).reset_index()
    routes_grouped_df['death_prediction'] = routes_grouped_df['death_prediction_probability'].map(lambda x: x >= 0.5)
    routes_grouped_df.sort_values(by=['death_prediction_probability'], inplace=True)
    routes_grouped_df['NIS introduction risk'] = routes_grouped_df['death_prediction'].map(lambda x: 'LOW' if x else 'HIGH')
    routes_grouped_df.to_csv(output_dir / 'routes_risk_grouped.csv', index=False)


def main(routes_train_path, routes_test_path, config_path, output_dir):
    config = Config.from_csv(config_path)

    cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.random_state)
    classifiers = [
        ('calibrated_sigmoid_lr', CalibratedClassifierCV(estimator=LogisticRegression(random_state=config.random_state), method='sigmoid', cv=cv_splitter)),
        ('calibrated_sigmoid_isotonic', CalibratedClassifierCV(estimator=LogisticRegression(random_state=config.random_state), method='isotonic', cv=cv_splitter)),
        ('lr', LogisticRegression(random_state=config.random_state))
    ] if not DEBUG else [('lr', LogisticRegression(random_state=config.random_state))]
    aggregation_types = ['min', 'max', 'mean', 'log_multiply'] if not DEBUG else ['log_multiply']

    for classifier_name, classifier in classifiers:
        for aggregation_type in aggregation_types:
            config_output_dir = output_dir / f'{classifier_name}_{aggregation_type}'
            config_output_dir.mkdir(parents=True, exist_ok=True)
            run_one_config(config, routes_train_path, routes_test_path, aggregation_type, classifier, config_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data for scoring routes')
    parser.add_argument('--routes_train_path', type=Path,
                        default=Path('full_routes_aggregated_scores') / 'routes_train_agg.csv')
    parser.add_argument('--routes_test_path', type=Path,
                        default=Path('full_routes_aggregated_scores') / 'routes_test_agg.csv')
    parser.add_argument('--output_dir', type=Path, default=Path('full_routes_risk_score'))
    args = parser.parse_args()

    main(args.routes_train_path, args.routes_test_path, CONFIG_PATH, args.output_dir)
