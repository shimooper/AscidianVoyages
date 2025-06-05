import itertools
from collections import Counter
from concurrent.futures import as_completed, ProcessPoolExecutor

import joblib
import pandas as pd
import json
import re
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

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

from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from utils import setup_logger, convert_pascal_to_snake_case, get_column_groups_sorted, convert_columns_to_int, \
    get_lived_columns_to_consider, merge_dicts_average, convert_data_to_tensor_for_rnn, \
    plot_models_comparison, SURVIVAL_COLORS
from configuration import METRIC_NAME_TO_SKLEARN_SCORER, DEBUG_MODE, Config
from model_lstm import LSTMModel


class Model:
    def __init__(self, config: Config, model_id, number_of_days_to_consider):
        self.config = config
        self.model_id = model_id
        self.number_of_days_to_consider = number_of_days_to_consider

        output_dir_path = config.models_dir_path / f'days_to_consider_{number_of_days_to_consider}'
        self.model_data_dir = output_dir_path / 'data'
        self.model_train_dir = output_dir_path / 'train_outputs'
        self.model_data_dir.mkdir(exist_ok=True, parents=True)
        self.model_train_dir.mkdir(exist_ok=True, parents=True)

        np.random.seed(self.config.random_state)

    def convert_routes_to_model_data(self, df):
        lived_columns, temperature_columns, salinity_columns = get_column_groups_sorted(df)

        days_data = []
        for index, row in df.iterrows():
            for col in lived_columns[-1:2:-1]:
                col_day = int(col.split(' ')[1])
                if pd.isna(row[f'Lived {col_day}']):
                    continue

                temperature_values = [row[f'Temp {col_day - i}'] for i in range(self.number_of_days_to_consider)][::-1]
                salinity_values = [row[f'Salinity {col_day - i}'] for i in range(self.number_of_days_to_consider)][::-1]
                time_values = [col_day - i for i in range(self.number_of_days_to_consider)][::-1]
                lived_cols_to_consider = get_lived_columns_to_consider(row, col_day, self.config.number_of_future_days_to_consider_death)

                new_row = [*temperature_values, *salinity_values, *time_values, any(row[lived_cols_to_consider])]
                days_data.append(new_row)

        days_df = pd.DataFrame(days_data, columns=['Temperature 1', 'Temperature 2', 'Temperature 3', 'Temperature 4',
                                                   'Salinity 1', 'Salinity 2', 'Salinity 3', 'Salinity 4',
                                                   'Time 1', 'Time 2', 'Time 3', 'Time 4',
                                                   'death'])
        convert_columns_to_int(days_df)

        return days_df

    def create_model_data(self):
        train_df = pd.read_csv(self.config.data_dir_path / 'train.csv')
        model_train_df = self.convert_routes_to_model_data(train_df)
        model_train_df.to_csv(self.model_data_dir / 'train.csv', index=False)

        test_df = pd.read_csv(self.config.data_dir_path / 'test.csv')
        model_test_df = self.convert_routes_to_model_data(test_df)
        model_test_df.to_csv(self.model_data_dir / 'test.csv', index=False)

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

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        sns.stripplot(data=df_temp, x='feature', y='value', hue='death_label', palette=SURVIVAL_COLORS, jitter=True, alpha=0.7, ax=axes[0],
                      legend=False)
        axes[0].set_ylabel("Temperature (celsius)", fontsize=12)
        # axes[0].tick_params(axis='x', rotation=45)
        axes[0].set_xlabel(None)

        sns.stripplot(data=df_salinity, x='feature', y='value', hue='death_label', palette=SURVIVAL_COLORS, jitter=True, alpha=0.7, ax=axes[1])
        # axes[1].yaxis.set_label_position("right")
        # axes[1].yaxis.tick_right()
        axes[1].set_ylabel("Salinity (ppt)", fontsize=12)
        # axes[1].tick_params(axis='x', rotation=45)
        axes[1].set_xlabel(None)
        axes[1].legend(title=None)

        plt.savefig(self.model_data_dir / "scatter_plot.png", dpi=600, bbox_inches='tight')
        plt.close()

    def run_analysis(self):
        raise NotImplementedError

    @staticmethod
    def plot_feature_importance(classifier_name, feature_names, features_importance, output_dir):
        indices = np.argsort(features_importance)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.bar(list(range(len(feature_names))), features_importance[indices], align="center")
        plt.xticks(list(range(len(feature_names))), [feature_names[i] for i in indices], rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / f'{classifier_name}_feature_importance.png', dpi=600)
        plt.close()

    @staticmethod
    def plot_decision_tree(model, feature_names, output_dir):
        plt.figure(figsize=(24, 16))
        plot_tree(
            model,
            feature_names=feature_names,  # Custom feature names
            class_names=["Alive", "Death"],  # Class names
            filled=True,  # Color nodes by class
            rounded=True,  # Rounded corners
            fontsize=14  # Font size
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
            f"Best estimator - MCC on test: {mcc_on_test}, AUPRC on test: {auprc_on_test}, F1 on test: {f1_on_test}")

        test_results = pd.DataFrame({'mcc': [mcc_on_test], 'auprc': [auprc_on_test], 'f1': [f1_on_test]})
        test_results.to_csv(output_dir / 'best_classifier_test_results.csv', index=False)


class ScikitModel(Model):
    def run_analysis(self):
        Xs_train, Ys_train, Xs_test, Ys_test = self.create_model_data()

        # Train
        train_logger = setup_logger(self.model_train_dir / 'classifiers_train.log', f'MODEL_{self.model_id}_TRAIN')

        # if self.config.do_feature_selection:
        #     selected_features_mask = self.feature_selection_on_train_data(train_logger, Xs_train, Ys_train)
        # else:
        #     selected_features_mask = [True] * len(Xs_train.columns)

        # Xs_train_selected = Xs_train.loc[:, selected_features_mask]
        # Xs_test_selected = Xs_test.loc[:, selected_features_mask]

        self.fit_and_test(train_logger, Xs_train, Ys_train, Xs_test, Ys_test)

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
        # if self.config.balance_classes:
        #     logger.info(f"Resampling train examples using RandomUnderSampler. Before: {y_train.value_counts()}")
        #     rus = RandomUnderSampler(random_state=self.config.random_state, sampling_strategy=self.config.max_classes_ratio)
        #     X_train, y_train = rus.fit_resample(X_train, y_train)
        #     logger.info(f"After resampling: {y_train.value_counts()}")

        cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config.random_state)
        best_classifiers_metrics = {}
        self.train_tree_models(logger, Xs_train, Ys_train, Xs_test, Ys_test, cv_splitter, best_classifiers_metrics)
        self.train_lstm(logger, Xs_train, Xs_test, Ys_train, Ys_test, cv_splitter, best_classifiers_metrics)

        best_classifier_dir = self.model_train_dir / 'best_classifier'
        best_classifier_dir.mkdir(exist_ok=True, parents=True)
        best_classifiers_df = pd.DataFrame.from_dict(best_classifiers_metrics, orient='index',
                                                     columns=['best_index', 'train mcc', 'train auprc', 'train f1',
                                                              'validation mcc', 'validation auprc', 'validation f1',
                                                              'model_path', 'model_results'])
        best_classifiers_df.index.name = 'model_name'
        best_classifiers_df.to_csv(best_classifier_dir / 'best_classifier_from_each_class.csv')
        logger.info(f"Aggregated the best classifiers from each classifier (after hyper-parameter tuning), and saved "
                    f"them to {best_classifier_dir / 'best_classifier_from_each_class.csv'}")

        plot_models_comparison(best_classifiers_df[['validation f1', 'validation auprc', 'validation mcc']].reset_index()
                               .rename(columns={'validation f1': 'F1', 'validation auprc': 'AUPRC', 'validation mcc': 'MCC'}),
                               best_classifier_dir, f'Classifiers comparison - validation set - {self.number_of_days_to_consider} days')

        best_classifier_class = best_classifiers_df[f'validation {self.config.metric}'].idxmax()
        best_classifier_results = best_classifiers_df.loc[best_classifier_class]
        best_classifier_results.to_csv(best_classifier_dir / 'best_classifier.csv')
        logger.info(f'Best classifier (validation {self.config.metric}): {best_classifier_class}. '
                    f'Saved its metrics to {best_classifier_dir / "best_classifier.csv"}')

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

        self.plot_univariate_features_with_respect_to_label(Xs_train_features, Xs_test_features, Ys_train, Ys_test)

        classifiers = self.create_classifiers_and_param_grids(Ys_train)
        for classifier, param_grid in classifiers:
            class_name = classifier.__class__.__name__
            classifier_output_dir = self.model_train_dir / convert_pascal_to_snake_case(class_name)
            classifier_output_dir.mkdir(exist_ok=True, parents=True)

            logger.info(f"Training Classifier {class_name} with hyperparameters tuning using GridSearch and StratifiedKFold cross-validation.")
            grid = GridSearchCV(
                estimator=classifier,
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
                best_estimator_path = classifier_output_dir / f"best_{class_name}.pkl"
                joblib.dump(grid.best_estimator_, best_estimator_path)
                best_estimator_results_path = classifier_output_dir / f"best_{class_name}_results.csv"
                grid_results.loc[grid.best_index_].to_csv(best_estimator_results_path)

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

                best_classifiers_metrics[class_name] = (grid.best_index_,
                                                        grid_results['mean_train_mcc'][grid.best_index_],
                                                        grid_results['mean_train_auprc'][grid.best_index_],
                                                        grid_results['mean_train_f1'][grid.best_index_],
                                                        grid_results['mean_test_mcc'][grid.best_index_],
                                                        grid_results['mean_test_auprc'][grid.best_index_],
                                                        grid_results['mean_test_f1'][grid.best_index_],
                                                        best_estimator_path,
                                                        best_estimator_results_path,
                                                        )

                if class_name in ['DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier',
                                  'XGBClassifier']:
                    self.plot_feature_importance(class_name, Xs_train_features.columns,
                                                 grid.best_estimator_.feature_importances_, classifier_output_dir)

                if class_name == 'DecisionTreeClassifier' and not DEBUG_MODE:
                    self.plot_decision_tree(grid.best_estimator_, list(Xs_train_features.columns), classifier_output_dir)
                    if len(Xs_train_features.columns) >= 2:
                        self.plot_decision_functions_of_features_pairs(Xs_train_features, Ys_train, grid.best_params_,
                                                                       grid.best_estimator_.feature_importances_,
                                                                       classifier_output_dir)

                Ys_test_predictions = grid.best_estimator_.predict_proba(Xs_test_features)
                y_pred_probs = Ys_test_predictions[:, 1]
                y_pred = Ys_test_predictions.argmax(axis=1)
                self.test_on_test_data(logger, Ys_test, y_pred_probs, y_pred, classifier_output_dir)

            except Exception as e:
                logger.error(f"Failed to train classifier {class_name} with error: {e}")

    def create_classifiers_and_param_grids(self, Ys_train):
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
                'n_estimators': [5],
                'max_depth': [None],
                'min_samples_split': [2],
                'min_samples_leaf': [1],
                'random_state': [self.config.random_state],
                'class_weight': ['balanced'],
                'bootstrap': [True]
            }

        if not DEBUG_MODE:
            decision_tree_grid = {
                'max_depth': [2, 3],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 5],
                'random_state': [self.config.random_state],
                'class_weight': ['balanced', None],
            }
        else:
            decision_tree_grid = {
                'max_depth': [5],
                'min_samples_split': [2],
                'min_samples_leaf': [1],
                'random_state': [self.config.random_state],
                'class_weight': ['balanced'],
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
                'n_estimators': [10],
                'max_depth': [3],
                'booster': ['dart'],
                'n_jobs': [1],
                'random_state': [self.config.random_state],
                'subsample': [1],
                'scale_pos_weight': [1],
            }

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
                'lr': [1e-4],
                'batch_size': [32]
            }

        # Grid search
        all_results = []

        if self.config.run_lstm_configurations_in_parallel:
            with ProcessPoolExecutor(max_workers=self.config.cpus) as executor:
                futures = []
                for hidden_size, num_layers, lr, batch_size in itertools.product(hyperparameter_grid['hidden_size'],
                                                                                 hyperparameter_grid['num_layers'],
                                                                                 hyperparameter_grid['lr'],
                                                                                 hyperparameter_grid['batch_size']):
                    futures.append(executor.submit(self.train_lstm_with_hyperparameters_cv, logger, classifier_output_dir,
                                                   hidden_size, num_layers, lr, batch_size, Xs_train, Ys_train,
                                                   device, self.config.metric, self.config.nn_max_epochs, cv_splitter))

                for future in as_completed(futures):
                    # Append result to the results list
                    all_results.append(future.result())
        else:
            for hidden_size, num_layers, lr, batch_size in itertools.product(hyperparameter_grid['hidden_size'],
                                                                             hyperparameter_grid['num_layers'],
                                                                             hyperparameter_grid['lr'],
                                                                             hyperparameter_grid['batch_size']):
                result = self.train_lstm_with_hyperparameters_cv(logger, classifier_output_dir,
                                                              hidden_size, num_layers, lr, batch_size,
                                                              Xs_train, Ys_train,
                                                              device, self.config.metric, self.config.nn_max_epochs, cv_splitter)
                all_results.append(result)

        # Save all hyperparameter results to CSV
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(classifier_output_dir / 'lstm_grid_results.csv')
        logger.info(f"Hyperparameter search results saved to {classifier_output_dir / 'hyperparameter_results.csv'}")

        # Find the best hyperparameters
        best_model_index = results_df[f'mean_test_{self.config.metric}'].idxmax()
        best_model_results = results_df.loc[best_model_index]
        best_model_results_path = classifier_output_dir / 'best_lstm_results.csv'
        best_model_results.to_csv(best_model_results_path)

        logger.info(f"Best LSTM model results after 5-fold cross-validation:\n{best_model_results}")

        # Now, train again using the best hyperparameters using 90/10 split of the training set
        logger.info(f"Training LSTM Classifier with the best hyperparameters...")
        final_train_dir = classifier_output_dir / 'final_train'
        final_train_dir.mkdir(exist_ok=True, parents=True)

        X_train, X_val, y_train, y_val = train_test_split(Xs_train, Ys_train, test_size=0.1, random_state=self.config.random_state)
        _, _, _, _, _, _, final_model_path = self.train_lstm_with_hyperparameters_train_val(
            final_train_dir, best_model_results['hidden_size'], best_model_results['num_layers'],
            best_model_results['lr'], best_model_results['batch_size'], X_train, y_train, X_val, y_val, device,
            self.config.metric, self.config.nn_max_epochs)

        best_classifiers_metrics['LSTMClassifier'] = [best_model_index, best_model_results['mean_train_mcc'],
                                            best_model_results['mean_train_auprc'], best_model_results['mean_train_f1'],
                                            best_model_results['mean_test_mcc'], best_model_results['mean_test_auprc'],
                                            best_model_results['mean_test_f1'], final_model_path, best_model_results_path]

        # Test final model on test data
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_test = scaler.transform(Xs_test)
        test_dataset = TensorDataset(X_test, torch.tensor(Ys_test.values, device=device))
        test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        final_model = LSTMModel.load_from_checkpoint(final_model_path)
        final_model.to(device)
        final_model.eval()
        with torch.no_grad():
            y_pred_probs = torch.cat([final_model(x) for x, _ in test_data_loader]).cpu().numpy().flatten()
            y_pred = (y_pred_probs > 0.5).astype(int)
        self.test_on_test_data(logger, Ys_test, y_pred_probs, y_pred, classifier_output_dir)

    def train_lstm_with_hyperparameters_train_val(self, output_dir, hidden_size, num_layers, lr, batch_size,
                                                  X_train, y_train, X_val, y_val, device, metric, nn_max_epochs):
        # Standardize the data
        scaler = StandardScaler()
        X_train_fold = scaler.fit_transform(X_train)
        X_val_fold = scaler.transform(X_val)

        X_train_tensor = convert_data_to_tensor_for_rnn(X_train_fold, device)
        X_val_tensor = convert_data_to_tensor_for_rnn(X_val_fold, device)

        train_dataset = TensorDataset(X_train_tensor, torch.tensor(y_train.values, device=device))
        val_dataset = TensorDataset(X_val_tensor, torch.tensor(y_val.values, device=device))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

        model = LSTMModel(hidden_size=hidden_size, num_layers=num_layers, lr=lr)
        model.to(device)

        checkpoint_callback = ModelCheckpoint(monitor=f'val_{metric}', mode='max', save_top_k=1,
                                              dirpath=output_dir / 'checkpoints',
                                              filename='best_model-epoch-{epoch}')
        trainer = L.Trainer(
            max_epochs=nn_max_epochs,
            logger=True,
            default_root_dir=output_dir,  # logs directory
            callbacks=[EarlyStopping(monitor=f'val_loss', patience=3, mode='min'), checkpoint_callback],
            deterministic=True
        )
        trainer.fit(model, train_loader, val_loader)

        best_model_path = checkpoint_callback.best_model_path
        best_model = LSTMModel.load_from_checkpoint(best_model_path)
        best_model.to(device)
        best_model.eval()

        with torch.no_grad():
            train_loader_for_evaluation = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            y_train_pred_probs = torch.cat(
                [best_model(x) for x, _ in train_loader_for_evaluation]).cpu().numpy().flatten()
            y_train_pred = (y_train_pred_probs > 0.5).astype(int)
            y_val_pred_probs = torch.cat([best_model(x) for x, _ in val_loader]).cpu().numpy().flatten()
            y_val_pred = (y_val_pred_probs > 0.5).astype(int)

        train_mcc = matthews_corrcoef(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        train_auprc = average_precision_score(y_train, y_train_pred_probs)
        val_mcc = matthews_corrcoef(y_val, y_val_pred)
        val_f1 = f1_score(y_val, y_val_pred)
        val_auprc = average_precision_score(y_val, y_val_pred_probs)

        return train_mcc, train_f1, train_auprc, val_mcc, val_f1, val_auprc, best_model_path

    def train_lstm_with_hyperparameters_cv(self, logger, classifier_output_dir, hidden_size, num_layers, lr, batch_size,
                                           X_train, y_train, device, metric, nn_max_epochs, cv_splitter):
        grid_combination_dir = classifier_output_dir / f'hidden_size_{hidden_size}_num_layers_{num_layers}_lr_{lr}_batch_size_{batch_size}'
        grid_combination_dir.mkdir(exist_ok=True, parents=True)

        logger.info(f"Training LSTM with hidden_size={hidden_size}, num_layers={num_layers}, lr={lr}, batch_size={batch_size}, "
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
                self.train_lstm_with_hyperparameters_train_val(fold_dir, hidden_size, num_layers, lr, batch_size,
                                                               X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                                                               device, metric, nn_max_epochs)

            results[f'split_{fold_id}_train_mcc'] = train_mcc
            results[f'split_{fold_id}_train_f1'] = train_f1
            results[f'split_{fold_id}_train_auprc'] = train_auprc
            results[f'split_{fold_id}_test_mcc'] = val_mcc
            results[f'split_{fold_id}_test_f1'] = val_f1
            results[f'split_{fold_id}_test_auprc'] = val_auprc

        results[f'mean_train_mcc'] = np.mean([results[f'split_{fold_id}_train_mcc'] for fold_id in range(cv_splitter.n_splits)])
        results[f'mean_train_f1'] = np.mean([results[f'split_{fold_id}_train_f1'] for fold_id in range(cv_splitter.n_splits)])
        results[f'mean_train_auprc'] = np.mean([results[f'split_{fold_id}_train_auprc'] for fold_id in range(cv_splitter.n_splits)])
        results[f'mean_test_mcc'] = np.mean([results[f'split_{fold_id}_test_mcc'] for fold_id in range(cv_splitter.n_splits)])
        results[f'mean_test_f1'] = np.mean([results[f'split_{fold_id}_test_f1'] for fold_id in range(cv_splitter.n_splits)])
        results[f'mean_test_auprc'] = np.mean([results[f'split_{fold_id}_test_auprc'] for fold_id in range(cv_splitter.n_splits)])

        return results
