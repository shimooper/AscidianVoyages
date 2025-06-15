import itertools
from collections import Counter
import joblib
import pandas as pd
import json
import re
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz

import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, PredefinedSplit
from sklearn.metrics import matthews_corrcoef, average_precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost
from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as IMBPipeline

from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from analyze_expirement_results.utils import convert_pascal_to_snake_case, get_column_groups_sorted, convert_columns_to_int, \
    get_lived_columns_to_consider, convert_data_to_tensor_for_rnn, \
    plot_models_comparison, SURVIVAL_COLORS
from analyze_expirement_results.configuration import METRIC_NAME_TO_SKLEARN_SCORER, DEBUG_MODE, Config, ROOT_DIR
from analyze_expirement_results.model_lstm import LSTMModel, train_lstm_with_hyperparameters_train_val
from analyze_expirement_results.train_lstm_cv import train_lstm_with_hyperparameters_cv
from analyze_expirement_results.q_submitter_power import submit_mini_batch, wait_for_results


class Model:
    def __init__(self, config: Config, interval_length):
        self.config = config
        self.interval_length = interval_length

        output_dir_path = config.models_dir_path / f'{interval_length}_day_interval'
        self.model_data_dir = output_dir_path / 'data'
        self.model_train_dir = output_dir_path / 'train_outputs'
        self.model_data_dir.mkdir(exist_ok=True, parents=True)
        self.model_train_dir.mkdir(exist_ok=True, parents=True)

        np.random.seed(self.config.random_state)

    def convert_routes_to_model_data(self, df):
        lived_columns, _, _ = get_column_groups_sorted(df)

        days_data = []
        for index, row in df.iterrows():
            route_days_data = self.convert_route_to_intervals(row, lived_columns)
            days_data.extend(route_days_data)

        days_df = pd.DataFrame(days_data, columns=[*[f'Temperature {i}' for i in range(1, self.interval_length + 1)],
                                                   *[f'Salinity {i}' for i in range(1, self.interval_length + 1)],
                                                   *[f'Time {i}' for i in range(1, self.interval_length + 1)],
                                                   'death'])
        convert_columns_to_int(days_df)

        return days_df

    def convert_route_to_intervals(self, route_row: pd.Series, lived_columns):
        days_data = []
        start_interval_min_day = self.config.max_interval_length - 2
        for col in lived_columns[-1:start_interval_min_day:-1]:
            col_day = int(col.split(' ')[1])
            if pd.isna(route_row[f'Temp {col_day}']):
                continue

            temperature_values = [route_row[f'Temp {col_day - i}'] for i in range(self.interval_length)][::-1]
            salinity_values = [route_row[f'Salinity {col_day - i}'] for i in range(self.interval_length)][::-1]
            time_values = [col_day - i for i in range(self.interval_length)][::-1]
            lived_cols_to_consider = get_lived_columns_to_consider(route_row, col_day,
                                                                   self.config.number_of_future_days_to_consider_death)

            new_row = [*temperature_values, *salinity_values, *time_values, any(route_row[lived_cols_to_consider])]
            days_data.append(new_row)

        return days_data

    def create_model_data(self, logger):
        train_df = pd.read_csv(self.config.data_dir_path / 'train.csv')
        model_train_df = self.convert_routes_to_model_data(train_df)
        model_train_df.to_csv(self.model_data_dir / 'train.csv', index=False)

        test_df = pd.read_csv(self.config.data_dir_path / 'test.csv')
        model_test_df = self.convert_routes_to_model_data(test_df)
        model_test_df.to_csv(self.model_data_dir / 'test.csv', index=False)

        logger.info(f"Created model data for training and testing with interval length {self.interval_length} days, "
                    f"and saved to {self.model_data_dir / 'train.csv'} and {self.model_data_dir / 'test.csv'}")

        Xs_train = model_train_df.drop(columns=['death'])
        Ys_train = model_train_df['death']
        Xs_test = model_test_df.drop(columns=['death'])
        Ys_test = model_test_df['death']

        return Xs_train, Ys_train, Xs_test, Ys_test

    def plot_univariate_features_with_respect_to_label(self, Xs_train, Xs_test, Ys_train, Ys_test):
        Xs_train = Xs_train.copy()
        Xs_test = Xs_test.copy()
        Xs_train['death'] = Ys_train
        Xs_test['death'] = Ys_test
        full_df = pd.concat([Xs_train, Xs_test], axis=0)

        full_df['death_label'] = full_df['death'].map({1: 'Dead', 0: 'Alive'}).astype('category')
        full_df.drop(columns=['death'], inplace=True)

        temp_cols = [col for col in full_df.columns if 'temp' in col.lower()]
        salinity_cols = [col for col in full_df.columns if 'sal' in col.lower()]

        df_temp = full_df[temp_cols + ['death_label']].melt(id_vars='death_label', var_name='feature',
                                                            value_name='value')
        df_salinity = full_df[salinity_cols + ['death_label']].melt(id_vars='death_label', var_name='feature',
                                                                    value_name='value')
        df_time = full_df[['days_passed', 'death_label']].melt(id_vars='death_label', var_name='feature',
                                                               value_name='value')

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        sns.stripplot(data=df_temp, x='feature', y='value', hue='death_label', palette=SURVIVAL_COLORS, jitter=True, alpha=0.7, ax=axes[0],
                      legend=False)
        axes[0].set_ylabel("Temperature (celsius)", fontsize=12)
        axes[0].set_xlabel(None)

        sns.stripplot(data=df_salinity, x='feature', y='value', hue='death_label', palette=SURVIVAL_COLORS, jitter=True, alpha=0.7, ax=axes[1],
                      legend=False)
        axes[1].set_ylabel("Salinity (ppt)", fontsize=12)
        axes[1].set_xlabel(None)

        sns.stripplot(data=df_time, x='feature', y='value', hue='death_label', palette=SURVIVAL_COLORS, jitter=True, alpha=0.7, ax=axes[2])
        axes[2].set_ylabel("Days passed", fontsize=12)
        axes[2].set_xlabel(None)
        axes[2].legend(title=None)

        plt.savefig(self.model_data_dir / "scatter_plot.png", dpi=600, bbox_inches='tight')
        plt.close()

    def run_analysis(self, logger):
        raise NotImplementedError

    @staticmethod
    def plot_feature_importance(classifier_name, feature_names, features_importance, output_dir):
        indices = np.argsort(features_importance)[::-1]
        plt.figure(figsize=(10, 6))
        plt.bar(list(range(len(feature_names))), features_importance[indices], align="center")
        plt.xticks(list(range(len(feature_names))), [feature_names[i] for i in indices], fontsize=12)
        plt.xlabel('Features', fontsize=14)
        plt.ylabel('Feature Importance', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f'{classifier_name}_feature_importance.png', dpi=600)
        plt.close()

    @staticmethod
    def plot_decision_tree(model, feature_names, output_dir):
        export_graphviz(
            model,
            out_file=str(output_dir / 'DecisionTreeClassifier_plot.dot'),
            feature_names=feature_names,
            class_names=['Alive', 'Dead'],
            max_depth=2,
            impurity=False,
            filled=True,  # Color nodes by class
            rounded=True,  # Rounded corners
            special_characters=True,
        )

        plt.figure(figsize=(24, 16))
        plot_tree(
            model,
            feature_names=feature_names,  # Custom feature names
            class_names=['Alive', 'Dead'],  # Class names
            max_depth=2,
            impurity=False,
            filled=True,  # Color nodes by class
            rounded=True,  # Rounded corners
            fontsize=16,  # Font size,
        )

        plt.savefig(output_dir / 'DecisionTreeClassifier_plot.png', dpi=600)
        plt.close()

    @staticmethod
    def plot_decision_functions_of_features_pairs(Xs_train, Ys_train, best_params, feature_importances, output_dir):
        n_classes = 2
        plot_colors = "gr"
        plot_step = 0.02

        # Create a DataFrame for feature importance and sort by importance
        importance_df = pd.DataFrame({
            'column': Xs_train.columns,
            'importance': feature_importances
        }).sort_values(by='importance', ascending=False)

        # Get the top 4 columns
        top_columns = importance_df['column'][:4].tolist()
        Xs_train = Xs_train[top_columns]

        for pairidx, pair in enumerate(itertools.combinations(Xs_train.columns, 2)):
            # We only take the two corresponding features
            X = Xs_train[list(pair)]
            # Train
            clf = DecisionTreeClassifier(**best_params).fit(X, Ys_train)

            # Plot the decision boundary
            ax = plt.subplot(2, 3, pairidx + 1)
            plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
            DecisionBoundaryDisplay.from_estimator(
                clf,
                X,
                cmap=plt.cm.RdYlBu,
                response_method="predict",
                ax=ax,
                xlabel=pair[0],
                ylabel=pair[1],
            )

            # Plot the training points
            for i, color in zip(range(n_classes), plot_colors):
                plt.scatter(
                    X.loc[Ys_train[Ys_train == i].index, X.columns[0]],
                    X.loc[Ys_train[Ys_train == i].index, X.columns[1]],
                    c=color,
                    label='Alive' if i == 0 else 'Death',
                    edgecolor="black",
                    s=15,
                )

        plt.suptitle("Decision surface of decision trees trained on pairs of features")
        plt.legend(loc="lower right", borderpad=0, handletextpad=0)
        _ = plt.axis("tight")
        plt.savefig(output_dir / 'decision_functions_on_features_pairs_plot.png', dpi=600)
        plt.close()

    def test_on_test_data(self, logger, Ys_test, y_pred_probs, y_pred, output_dir):
        mcc_on_test = matthews_corrcoef(Ys_test, y_pred)
        auprc_on_test = average_precision_score(Ys_test, y_pred_probs)
        f1_on_test = f1_score(Ys_test, y_pred)

        logger.info(
            f"Final estimator - MCC on test: {mcc_on_test}, AUPRC on test: {auprc_on_test}, F1 on test: {f1_on_test}")

        test_results = pd.DataFrame({'mcc': [mcc_on_test], 'auprc': [auprc_on_test], 'f1': [f1_on_test]})
        test_results.to_csv(output_dir / 'final_classifier_test_results.csv', index=False)


class ScikitModel(Model):
    def run_analysis(self, logger):
        Xs_train, Ys_train, Xs_test, Ys_test = self.create_model_data(logger)

        # Train


        # if self.config.do_feature_selection:
        #     selected_features_mask = self.feature_selection_on_train_data(train_logger, Xs_train, Ys_train)
        # else:
        #     selected_features_mask = [True] * len(Xs_train.columns)

        # Xs_train_selected = Xs_train.loc[:, selected_features_mask]
        # Xs_test_selected = Xs_test.loc[:, selected_features_mask]

        self.fit_and_test(logger, Xs_train, Ys_train, Xs_test, Ys_test)

    def feature_selection_on_train_data(self, logger, Xs_train, Ys_train):
        outputs_dir = self.model_train_dir / 'feature_selection'
        outputs_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"Performing feature selection with RFECV using the estimator RandomForestClassifier on the training data.")
        rf = RandomForestClassifier(n_estimators=5, max_depth=5, random_state=self.config.random_state, n_jobs=self.config.cpus)
        cv = StratifiedKFold(shuffle=True, random_state=self.config.random_state)

        rfecv = RFECV(estimator=rf, cv=cv, scoring=METRIC_NAME_TO_SKLEARN_SCORER[self.config.metric], n_jobs=self.config.cpus)
        rfecv.fit(Xs_train, Ys_train)  # Transform dataset to keep only selected features

        rfecv_results = pd.DataFrame.from_dict(rfecv.cv_results_)
        rfecv_results.to_csv(outputs_dir / f'rfecv_results.csv', index=False)
        features_ranking = pd.DataFrame(
            {'feature': rfecv.feature_names_in_, 'ranking': rfecv.ranking_, 'support': rfecv.support_})
        features_ranking.to_csv(outputs_dir / 'features_ranking.csv', index=False)
        logger.info(f'RFECV selected {rfecv.n_features_} features out of {rfecv.n_features_in_}.')

        return rfecv.support_

    def fit_and_test(self, logger, Xs_train, Ys_train, Xs_test, Ys_test):
        cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config.random_state)
        best_classifiers_metrics = {}
        self.train_tree_models(logger, Xs_train, Ys_train, Xs_test, Ys_test, cv_splitter, best_classifiers_metrics)
        self.train_lstm(logger, Xs_train, Xs_test, Ys_train, Ys_test, cv_splitter, best_classifiers_metrics)

        best_classifier_dir = self.model_train_dir / 'best_classifier'
        best_classifier_dir.mkdir(exist_ok=True, parents=True)

        # Find common indices across all Series objects (which are the values of best_classifiers_metrics)
        common_index = set.intersection(*(set(s.index) for s in best_classifiers_metrics.values()))
        common_index = sorted(common_index)
        # Filter each Series to keep only the common index
        filtered = {k: v[common_index] for k, v in best_classifiers_metrics.items()}
        best_classifiers_df = pd.DataFrame.from_dict(filtered, orient='index')

        best_classifiers_df.index.name = 'model_name'
        best_classifiers_df.to_csv(best_classifier_dir / 'best_classifier_from_each_class.csv')
        logger.info(f"Aggregated the best classifiers from each classifier (after hyper-parameter tuning), and saved "
                    f"them to {best_classifier_dir / 'best_classifier_from_each_class.csv'}")

        plot_models_comparison(best_classifiers_df, best_classifier_dir, self.config.metric)

        # Save metadata
        metadata = {
            'numpy_version': np.__version__,
            'joblib_version': joblib.__version__,
            'sklearn_version': sklearn.__version__,
            'pandas_version': pd.__version__,
            'xgboost_version': xgboost.__version__,
            'lightning_version': L.__version__,
            'torch_version': torch.__version__,
        }
        with open(self.model_train_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f)

    def convert_X_to_features(self, X_df):
        temperature_cols = [col for col in X_df.columns if col.startswith("Temperature ")]
        salinity_cols = [col for col in X_df.columns if col.startswith("Salinity ")]
        time_cols = [col for col in X_df.columns if col.startswith("Time ")]

        def extract_number(col):
            match = re.match(r'Time (\d+)', col)
            return int(match.group(1)) if match else -1

        df_features = pd.DataFrame()
        df_features["min_temp"] = X_df[temperature_cols].min(axis=1)
        df_features["max_temp"] = X_df[temperature_cols].max(axis=1)
        df_features["max_temp_diff"] = df_features["max_temp"] - df_features["min_temp"]

        df_features["min_sal"] = X_df[salinity_cols].min(axis=1)
        df_features["max_sal"] = X_df[salinity_cols].max(axis=1)
        df_features["max_sal_diff"] = df_features["max_sal"] - df_features["min_sal"]

        df_features["days_passed"] = X_df[max(time_cols, key=extract_number)]

        return df_features

    def train_tree_models(self, logger, Xs_train, Ys_train, Xs_test, Ys_test, cv_splitter, best_classifiers_metrics):
        Xs_train_features = self.convert_X_to_features(Xs_train)
        Xs_test_features = self.convert_X_to_features(Xs_test)
        Xs_train_features.to_csv(self.model_data_dir / 'Xs_train_features.csv', index=False)
        Xs_test_features.to_csv(self.model_data_dir / 'Xs_test_features.csv', index=False)
        logger.info(f"Converted Xs_train and Xs_test to features, and saved them to "
                    f"{self.model_data_dir / 'Xs_train_features.csv'} and {self.model_data_dir / 'Xs_test_features.csv'}")

        self.plot_univariate_features_with_respect_to_label(Xs_train_features, Xs_test_features, Ys_train, Ys_test)
        logger.info(f"Plotted univariate features with respect to label, and saved")

        classifiers = self.create_classifiers_and_param_grids(Ys_train, self.config.balance_classes)
        for classifier, param_grid in classifiers:
            class_name = classifier.__class__.__name__
            classifier_output_dir = self.model_train_dir / convert_pascal_to_snake_case(class_name)
            classifier_output_dir.mkdir(exist_ok=True, parents=True)

            logger.info(f"Training Classifier {class_name} with hyperparameters tuning using GridSearch and StratifiedKFold cross-validation.")

            if self.config.balance_classes:
                logger.info(f"Resampling train examples using RandomUnderSampler.")
                estimator = IMBPipeline([
                    ('undersample', RandomUnderSampler(random_state=self.config.random_state, sampling_strategy=self.config.max_classes_ratio)),
                    ('clf', classifier)
                ])
            else:
                estimator = classifier

            grid = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=cv_splitter,
                scoring=METRIC_NAME_TO_SKLEARN_SCORER,
                refit=self.config.metric,
                return_train_score=True,
                verbose=1,
                n_jobs=self.config.cpus
            )

            try:
                grid.fit(Xs_train_features, Ys_train)
                grid_results = pd.DataFrame.from_dict(grid.cv_results_)
                grid_results.to_csv(classifier_output_dir / f'{class_name}_grid_results.csv')

                best_estimator = grid.best_estimator_ if not self.config.balance_classes else grid.best_estimator_.named_steps['clf']
                best_estimator_path = classifier_output_dir / f"best_{class_name}.pkl"
                joblib.dump(best_estimator, best_estimator_path)

                best_estimator_results = grid_results.loc[grid.best_index_]
                best_estimator_results_path = classifier_output_dir / f"best_{class_name}_results.csv"
                best_estimator_results.to_csv(best_estimator_results_path)

                # Note: grid.best_score_ == grid_results['mean_test_{metric}'][grid.best_index_] (the mean cross-validated score of the best_estimator)
                logger.info(
                    f"Best params: {grid.best_params_}, Best index: {grid.best_index_}, Best score: {grid.best_score_}")

                logger.info(
                    f"Best estimator - MCC on train: {grid_results['mean_train_mcc'][grid.best_index_]}, "
                    f"AUPRC on train: {grid_results['mean_train_auprc'][grid.best_index_]}, "
                    f"F1 on train: {grid_results['mean_train_f1'][grid.best_index_]}, "
                    f"MCC on held-out: {grid_results['mean_test_mcc'][grid.best_index_]}, "
                    f"AUPRC on held-out: {grid_results['mean_test_auprc'][grid.best_index_]}, "
                    f"F1 on held-out: {grid_results['mean_test_f1'][grid.best_index_]}")

                best_classifiers_metrics[class_name] = best_estimator_results

                if class_name in ['DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier',
                                  'XGBClassifier']:
                    self.plot_feature_importance(class_name, Xs_train_features.columns,
                                                 best_estimator.feature_importances_, classifier_output_dir)

                if class_name == 'DecisionTreeClassifier':
                    self.plot_decision_tree(best_estimator, list(Xs_train_features.columns), classifier_output_dir)
                    if len(Xs_train_features.columns) >= 2 and not DEBUG_MODE:
                        best_params = grid.best_params_
                        if self.config.balance_classes:
                            best_params = {k.replace('clf__', ''): v for k, v in best_params.items()}
                        self.plot_decision_functions_of_features_pairs(Xs_train_features, Ys_train, best_params,
                                                                       best_estimator.feature_importances_,
                                                                       classifier_output_dir)

                Ys_test_predictions = best_estimator.predict_proba(Xs_test_features)
                y_pred_probs = Ys_test_predictions[:, 1]
                y_pred = Ys_test_predictions.argmax(axis=1)
                self.test_on_test_data(logger, Ys_test, y_pred_probs, y_pred, classifier_output_dir)

            except Exception as e:
                logger.exception(f"Failed to train classifier {class_name} with error: {e}")

    def create_classifiers_and_param_grids(self, Ys_train, adjust_to_pipeline=False):
        knn_grid = {
            'n_neighbors': [5, 10],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski']
        }

        logistic_regression_grid = {
            'solver': ['liblinear'],
            'penalty': ['l1', 'l2'],
            'C': [0.01, 0.1, 1, 10, 100],
            'max_iter': [5000],
            'random_state': [self.config.random_state],
            'class_weight': ['balanced', None],
        }

        mlp_grid = {
            'hidden_layer_sizes': [(10, 3), (30, 5), ],
            'activation': ['tanh', 'relu'],
            'solver': ['adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [400],
            'early_stopping': [True],
            'random_state': [self.config.random_state],
        }

        gbc_grid = {
            'n_estimators': [5, 20, 100],
            'max_depth': [None, 3, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'random_state': [self.config.random_state],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.6, 1],
        }

        if not DEBUG_MODE:
            rfc_grid = {
                'n_estimators': [5, 10, 20],
                'max_depth': [3, 5],
                'min_samples_split': [2, 10],
                'min_samples_leaf': [1, 2, 5],
                'random_state': [self.config.random_state],
                'class_weight': ['balanced', None],
                'bootstrap': [True, False]
            }
        else:
            rfc_grid = {
                'n_estimators': [5, 10],
                'max_depth': [None],
                'min_samples_split': [2],
                'min_samples_leaf': [1],
                'random_state': [self.config.random_state],
                'class_weight': [None],
                'bootstrap': [True]
            }

        if not DEBUG_MODE:
            decision_tree_grid = {
                'max_depth': [3, 5],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5],
                'random_state': [self.config.random_state],
                'class_weight': ['balanced', None],
            }
        else:
            decision_tree_grid = {
                'max_depth': [3, 5],
                'min_samples_split': [2],
                'min_samples_leaf': [1],
                'random_state': [self.config.random_state],
                'class_weight': [None],
            }

        if not DEBUG_MODE:
            Ys_trains_classes_counts = Counter(Ys_train)
            xgboost_grid = {
                'learning_rate': [0.01, 0.1],
                'n_estimators': [5, 10, 20],
                'max_depth': [3, 5],
                'booster': ['dart'],
                'n_jobs': [1],
                'random_state': [self.config.random_state],
                'subsample': [0.6, 1],
                'scale_pos_weight': [1, Ys_trains_classes_counts[0] / Ys_trains_classes_counts[1]],
            }
        else:
            xgboost_grid = {
                'learning_rate': [0.01],
                'n_estimators': [5, 10],
                'max_depth': [3],
                'booster': ['dart'],
                'n_jobs': [1],
                'random_state': [self.config.random_state],
                'subsample': [1],
                'scale_pos_weight': [1],
            }

        if adjust_to_pipeline:
            # Adjust the grids for pipeline usage
            rfc_grid = {f'clf__{k}': v for k, v in rfc_grid.items()}
            decision_tree_grid = {f'clf__{k}': v for k, v in decision_tree_grid.items()}
            xgboost_grid = {f'clf__{k}': v for k, v in xgboost_grid.items()}

        return [
            # (KNeighborsClassifier(), knn_grid),
            # (LogisticRegression(), logistic_regression_grid),
            # (MLPClassifier(), mlp_grid),
            # (GradientBoostingClassifier(), gbc_grid),
            (RandomForestClassifier(), rfc_grid),
            (DecisionTreeClassifier(), decision_tree_grid),
            (XGBClassifier(), xgboost_grid)
        ]

    def train_lstm(self, logger, Xs_train, Xs_test, Ys_train, Ys_test, cv_splitter, best_classifiers_metrics):
        logger.info(f"Training LSTM Classifier with hyperparameters tuning using 5-fold cross-validation")

        device = torch.device('cpu')
        L.seed_everything(self.config.random_state, workers=True)
        classifier_output_dir = self.model_train_dir / 'lstm_classifier'

        # Hyperparameter grid
        if not DEBUG_MODE:
            hyperparameter_grid = {
                'hidden_size': [8, 16, 32],
                'num_layers': [1, 2],
                'lr': [1e-4, 1e-3, 1e-2],
                'batch_size': [16, 32, 64]
            }
        else:
            hyperparameter_grid = {
                'hidden_size': [8],
                'num_layers': [1],
                'lr': [1e-4, 1e-3],
                'batch_size': [32]
            }

        # Grid search
        if self.config.run_lstm_configurations_in_parallel:
            train_lstm_cv_script = ROOT_DIR / 'train_lstm_cv.py'
            logs_dir = classifier_output_dir / 'logs'
            logs_dir.mkdir(exist_ok=True, parents=True)

            i = 0
            for hidden_size, num_layers, lr, batch_size in itertools.product(hyperparameter_grid['hidden_size'],
                                                                             hyperparameter_grid['num_layers'],
                                                                             hyperparameter_grid['lr'],
                                                                             hyperparameter_grid['batch_size']):
                param_list = [self.config.outputs_dir_path / 'config.csv', classifier_output_dir, hidden_size,
                              num_layers, lr, batch_size, self.model_data_dir / 'train.csv']
                submit_mini_batch(logger, train_lstm_cv_script, [param_list], logs_dir, f'lstm_cv_{i}',
                                  self.config.error_file_path, num_of_cpus=4)
                i += 1

            wait_for_results(logger, train_lstm_cv_script, logs_dir, i, self.config.error_file_path)

            all_results = []
            for grid_combination_dir in classifier_output_dir.iterdir():
                if grid_combination_dir.name.startswith('hs_'):
                    df = pd.read_csv(grid_combination_dir / 'results.csv', index_col=0).squeeze('columns')
                    all_results.append(df)
            results_df = pd.DataFrame(all_results).reset_index(drop=True)
        else:
            all_results = []
            for hidden_size, num_layers, lr, batch_size in itertools.product(hyperparameter_grid['hidden_size'],
                                                                             hyperparameter_grid['num_layers'],
                                                                             hyperparameter_grid['lr'],
                                                                             hyperparameter_grid['batch_size']):
                result = train_lstm_with_hyperparameters_cv(logger, self.config, classifier_output_dir,
                                                              hidden_size, num_layers, lr, batch_size,
                                                              Xs_train, Ys_train,
                                                              device, cv_splitter)
                all_results.append(result)
            results_df = pd.DataFrame(all_results)

        # Save all hyperparameter results to CSV
        results_df.to_csv(classifier_output_dir / 'lstm_grid_results.csv')
        logger.info(f"Hyperparameter search results saved to {classifier_output_dir / 'hyperparameter_results.csv'}")

        # Find the best hyperparameters
        best_model_index = results_df[f'mean_test_{self.config.metric}'].idxmax()
        best_model_results = results_df.loc[best_model_index]
        best_model_results_path = classifier_output_dir / 'best_lstm_results.csv'
        best_model_results.to_csv(best_model_results_path)

        logger.info(f"Best LSTM model results after 5-fold cross-validation:\n{best_model_results}")

        # Now, train again using the best hyperparameters using 90/10 split of the training set
        logger.info(f"Training final LSTM Classifier with the best hyperparameters...")
        final_train_dir = classifier_output_dir / 'final_train'
        final_train_dir.mkdir(exist_ok=True, parents=True)

        X_train, X_val, y_train, y_val = train_test_split(Xs_train, Ys_train, test_size=0.1, random_state=self.config.random_state)
        _, _, _, _, _, _, final_model_path = train_lstm_with_hyperparameters_train_val(
            logger, self.config, final_train_dir, int(best_model_results['hidden_size']), int(best_model_results['num_layers']),
            best_model_results['lr'], int(best_model_results['batch_size']), X_train, y_train, X_val, y_val, device)

        best_classifiers_metrics['LSTMClassifier'] = best_model_results

        # Test final model on test data
        scaler = StandardScaler()
        scaler.fit(X_train)
        joblib.dump(scaler, final_train_dir / 'scaler.pkl')
        Xs_test_scaled = pd.DataFrame(scaler.transform(Xs_test), columns=Xs_test.columns, index=Xs_test.index)
        X_test_tensor = convert_data_to_tensor_for_rnn(Xs_test_scaled, device)
        test_dataset = TensorDataset(X_test_tensor, torch.tensor(Ys_test.values, device=device))
        test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        final_model = LSTMModel.load_from_checkpoint(final_model_path)
        final_model.to(device)
        final_model.eval()
        with torch.no_grad():
            y_pred_probs = torch.cat([final_model(x) for x, _ in test_data_loader]).cpu().numpy().flatten()
            y_pred = (y_pred_probs > 0.5).astype(int)
        self.test_on_test_data(logger, Ys_test, y_pred_probs, y_pred, classifier_output_dir)
