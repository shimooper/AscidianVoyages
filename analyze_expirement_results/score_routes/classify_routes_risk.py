from pathlib import Path
import pandas as pd
import joblib
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import matthews_corrcoef, average_precision_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

from analyze_expirement_results.utils import setup_logger
from analyze_expirement_results.configuration import Config

PROJECT_ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT_DIR / 'outputs' / 'configuration_0' / 'config.csv'

DEBUG = False
CLASSIFIERS = lambda rs: [LogisticRegression(random_state=rs), DecisionTreeClassifier(random_state=rs)] if not DEBUG else [LogisticRegression(random_state=rs)]
AGGREGATION_TYPES = ['min', 'max', 'mean', 'multiply'] if not DEBUG else ['multiply']


def run_one_config(routes_train_path, routes_test_path, aggregation_type, route_classifier, output_dir):
    train_df = pd.read_csv(routes_train_path)
    test_df = pd.read_csv(routes_test_path)

    # Use the aggregated score to fit a classification model
    route_classifier.fit(train_df[[aggregation_type]], train_df['death'])
    joblib.dump(route_classifier, output_dir / 'route_risk_classifier.pkl')

    all_metrics = []
    for dataset_name, dataset_df in [('train', train_df), ('test', test_df)]:
        routes_predictions = route_classifier.predict(dataset_df[[aggregation_type]])
        routes_predictions_probabilities = route_classifier.predict_proba(dataset_df[[aggregation_type]])[:, 1]
        dataset_df['death_prediction'] = routes_predictions
        dataset_df['death_prediction_probability'] = routes_predictions_probabilities

        mcc = matthews_corrcoef(dataset_df['death'], routes_predictions)
        auprc = average_precision_score(dataset_df['death'], routes_predictions_probabilities)
        f1 = f1_score(dataset_df['death'], routes_predictions)

        metrics = {'split': dataset_name, 'mcc': mcc, 'auprc': auprc, 'f1': f1}
        all_metrics.append(metrics)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(output_dir / 'routes_risk_results.csv', index=False)

    # Plot
    df_melted = metrics_df.melt(id_vars='split', var_name='metric', value_name='value')
    sns.barplot(data=df_melted, x='metric', y='value', hue='split')
    plt.title("Routes Risk Classifier Performance")
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.legend(title='Dataset')
    plt.savefig(output_dir / 'routes_risk_classified.png', dpi=600, bbox_inches='tight')
    plt.close()

    # Write to file routes scores
    all_routes_df = pd.concat([train_df, test_df], ignore_index=True)
    all_routes_df.sort_values(by=['death_prediction_probability'], inplace=True)
    all_routes_df.to_csv(output_dir / 'routes_risk.csv', index=False)

    # Group replicates
    routes_grouped_df = all_routes_df.groupby(by=['Season', 'Name']).agg({
        'death_prediction': lambda x: x.mode()[0],
        'death_prediction_probability': 'mean',
    }).reset_index()
    routes_grouped_df.sort_values(by=['death_prediction_probability'], inplace=True)
    routes_grouped_df.to_csv(output_dir / 'routes_risk_grouped.csv', index=False)


def main(routes_train_path, routes_test_path, config_path, output_dir):
    config = Config.from_csv(config_path)
    for classifier in CLASSIFIERS(config.random_state):
        for aggregation_type in AGGREGATION_TYPES:
            config_output_dir = output_dir / f'{classifier.__class__.__name__}_{aggregation_type}'
            config_output_dir.mkdir(parents=True, exist_ok=True)
            run_one_config(routes_train_path, routes_test_path, aggregation_type, classifier, config_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data for scoring routes')
    parser.add_argument('--routes_train_path', type=Path,
                        default=Path('full_routes_aggregated_scores') / 'routes_train_agg.csv')
    parser.add_argument('--routes_test_path', type=Path,
                        default=Path('full_routes_aggregated_scores') / 'routes_test_agg.csv')
    parser.add_argument('--output_dir', type=Path, default=Path('full_routes_risk_score'))
    args = parser.parse_args()

    main(args.routes_train_path, args.routes_test_path, CONFIG_PATH, args.output_dir)
