import itertools
import shutil
from collections import Counter

import joblib
import pandas as pd
import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
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
from sklearn.model_selection import train_test_split

from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

import torch
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import optuna

from utils import setup_logger, convert_pascal_to_snake_case, get_column_groups_sorted, convert_columns_to_int, \
    get_lived_columns_to_consider, merge_dicts_average, convert_features_df_to_tensor_for_rnn, downsample_negative_class, \
    DAYS_DESCRIPTIONS, plot_models_comparison
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
        self.model_test_dir = output_dir_path / 'test_outputs'
        self.model_data_dir.mkdir(exist_ok=True, parents=True)
        self.model_train_dir.mkdir(exist_ok=True, parents=True)
        self.model_test_dir.mkdir(exist_ok=True, parents=True)

        np.random.seed(self.config.random_state)

    def convert_routes_to_model_data(self, df):
        lived_columns, temperature_columns, salinity_columns = get_column_groups_sorted(df)

        days_data = []
        for index, row in df.iterrows():
            for col in lived_columns[-1:2:-1]:
                col_day = int(col.split(' ')[1])
                if pd.isna(row[f'Lived {col_day}']):
                    continue

                temperature_columns = {f'{DAYS_DESCRIPTIONS[i]} Temperature': row[f'Temp {col_day - i}'] for i in range(self.number_of_days_to_consider)}
                salinity_columns = {f'{DAYS_DESCRIPTIONS[i]} Salinity': row[f'Salinity {col_day - i}'] for i in range(self.number_of_days_to_consider)}
                lived_cols_to_consider = get_lived_columns_to_consider(row, col_day, self.config.number_of_future_days_to_consider_death)

                new_row = {
                    **temperature_columns, **salinity_columns,
                    'death': any(row[lived_cols_to_consider]),
                }
                days_data.append(new_row)

        days_df = pd.DataFrame(days_data)
        convert_columns_to_int(days_df)

        return days_df

    def create_model_data(self):
        train_df = pd.read_csv(self.config.data_dir_path / 'train.csv')
        model_train_df = self.convert_routes_to_model_data(train_df)
        model_train_df.to_csv(self.model_data_dir / 'train.csv', index=False)

        test_df = pd.read_csv(self.config.data_dir_path / 'test.csv')
        model_test_df = self.convert_routes_to_model_data(test_df)
        model_test_df.to_csv(self.model_data_dir / 'test.csv', index=False)

        if not DEBUG_MODE:
            self.plot_feature_pairs(model_train_df, model_test_df)

        Xs_train = model_train_df.drop(columns=['death'])
        Ys_train = model_train_df['death']
        Xs_test = model_test_df.drop(columns=['death'])
        Ys_test = model_test_df['death']

        return Xs_train, Ys_train, Xs_test, Ys_test

    def plot_feature_pairs(self, train_df, test_df):
        full_df = pd.concat([train_df, test_df], axis=0)

        full_df['death_label'] = full_df['death'].map({1: 'Dead', 0: 'Alive'}).astype('category')
        full_df.drop(columns=['death'], inplace=True)

        sns.pairplot(full_df, hue='death_label', palette={'Dead': 'red', 'Alive': 'green'})

        plt.grid(alpha=0.3)
        plt.savefig(self.model_data_dir / "scatter_plot.png", dpi=300, bbox_inches='tight')
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
            fontsize=10  # Font size
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

    def test_on_test_data(self, model_path, Xs_test, Ys_test):
        logger = setup_logger(self.model_test_dir / 'classifiers_test.log', f'MODEL_{self.model_id}_TEST')

        if model_path.endswith('.pkl'):
            model = joblib.load(model_path)
            Ys_test_predictions = model.predict_proba(Xs_test)
            y_pred_probs = Ys_test_predictions[:, 1]
            y_pred = Ys_test_predictions.argmax(axis=1)
        elif model_path.endswith('.ckpt'):
            Xs_test = convert_features_df_to_tensor_for_rnn(Xs_test)
            dataset = TensorDataset(Xs_test, torch.tensor(Ys_test.values))
            data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
            model = LSTMModel.load_from_checkpoint(model_path)
            model.eval()
            with torch.no_grad():
                y_pred_probs = torch.cat([model(x) for x, _ in data_loader]).cpu().numpy().flatten()
                y_pred = (y_pred_probs > 0.5).astype(int)
        else:
            raise ValueError(f"Unknown model file type: {model_path}")

        mcc_on_test = matthews_corrcoef(Ys_test, y_pred)
        auprc_on_test = average_precision_score(Ys_test, y_pred_probs)
        f1_on_test = f1_score(Ys_test, y_pred)

        logger.info(
            f"Best estimator ({model.__class__}) - MCC on test: {mcc_on_test}, AUPRC on test: {auprc_on_test}, F1 on test: {f1_on_test}")

        test_results = pd.DataFrame({'mcc': [mcc_on_test], 'auprc': [auprc_on_test], 'f1': [f1_on_test]})
        test_results.to_csv(self.model_test_dir / 'best_classifier_test_results.csv', index=False)


class ScikitModel(Model):
    def run_analysis(self):
        Xs_train, Ys_train, Xs_test, Ys_test = self.create_model_data()

        # Train
        train_logger = setup_logger(self.model_train_dir / 'classifiers_train.log', f'MODEL_{self.model_id}_TRAIN')

        # if self.config.downsample_majority_class:
        #     Xs_train, Ys_train = downsample_negative_class(train_logger, Xs_train, Ys_train, self.config.random_state, self.config.max_classes_ratio)

        if self.config.do_feature_selection:
            selected_features_mask = self.feature_selection_on_train_data(train_logger, Xs_train, Ys_train)
        else:
            selected_features_mask = [True] * len(Xs_train.columns)

        Xs_train_selected = Xs_train.loc[:, selected_features_mask]
        best_model_path = self.fit_on_train_data(train_logger, Xs_train_selected, Ys_train)

        # Test
        Xs_test_selected = Xs_test.loc[:, selected_features_mask]
        self.test_on_test_data(best_model_path, Xs_test_selected, Ys_test)

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

    def fit_on_train_data(self, logger, Xs_train, Ys_train):
        best_classifiers_metrics = {}
        classifiers = self.create_classifiers_and_param_grids(Ys_train)
        for classifier, param_grid in classifiers:
            class_name = classifier.__class__.__name__
            classifier_output_dir = self.model_train_dir / convert_pascal_to_snake_case(class_name)
            classifier_output_dir.mkdir(exist_ok=True, parents=True)

            logger.info(f"Training Classifier {class_name} with hyperparameters tuning using GridSearch and Stratified-KFold CV.")

            if self.config.balance_classes:
                logger.info(f"Using RandomUnderSampler to balance classes during training (integrated with GridSearchCV.")
                rus = RandomUnderSampler(random_state=self.config.random_state, sampling_strategy=self.config.max_classes_ratio)
                estimator = Pipeline([
                    ('undersample', rus),
                    ('clf', classifier)
                ])
            else:
                estimator = classifier

            grid = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                scoring=METRIC_NAME_TO_SKLEARN_SCORER,
                refit=self.config.metric,
                return_train_score=True,
                verbose=1,
                n_jobs=self.config.cpus
            )

            try:
                grid.fit(Xs_train, Ys_train)
                grid_results = pd.DataFrame.from_dict(grid.cv_results_)
                grid_results.to_csv(classifier_output_dir / f'{class_name}_grid_results.csv')
                best_estimator_path = classifier_output_dir / f"best_{class_name}.pkl"
                joblib.dump(grid.best_estimator_, best_estimator_path)

                # Note: grid.best_score_ == grid_results['mean_test_{metric}'][grid.best_index_] (the mean cross-validated score of the best_estimator)
                logger.info(
                    f"Best params: {grid.best_params_}, Best index: {grid.best_index_}, Best score: {grid.best_score_}")

                logger.info(
                    f"Best estimator - Mean MCC on train folds: {grid_results['mean_train_mcc'][grid.best_index_]}, "
                    f"Mean AUPRC on train folds: {grid_results['mean_train_auprc'][grid.best_index_]}, "
                    f"Mean F1 on train fold: {grid_results['mean_train_f1'][grid.best_index_]}, "
                    f"Mean MCC on held-out folds: {grid_results['mean_test_mcc'][grid.best_index_]}, "
                    f"Mean AUPRC on held-out folds: {grid_results['mean_test_auprc'][grid.best_index_]}, "
                    f"Mean F1 o held-out folds: {grid_results['mean_test_f1'][grid.best_index_]}")

                best_classifiers_metrics[class_name] = (grid.best_index_,
                                                        grid_results['mean_train_mcc'][grid.best_index_],
                                                        grid_results['mean_train_auprc'][grid.best_index_],
                                                        grid_results['mean_train_f1'][grid.best_index_],
                                                        grid_results['mean_test_mcc'][grid.best_index_],
                                                        grid_results['mean_test_auprc'][grid.best_index_],
                                                        grid_results['mean_test_f1'][grid.best_index_],
                                                        best_estimator_path)

                if class_name in ['DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier',
                                  'XGBClassifier']:
                    self.plot_feature_importance(class_name, Xs_train.columns,
                                                 grid.best_estimator_.feature_importances_, classifier_output_dir)

                if class_name == 'DecisionTreeClassifier' and not DEBUG_MODE:
                    self.plot_decision_tree(grid.best_estimator_, list(Xs_train.columns), classifier_output_dir)
                    if len(Xs_train.columns) >= 2:
                        self.plot_decision_functions_of_features_pairs(Xs_train, Ys_train, grid.best_params_,
                                                                       grid.best_estimator_.feature_importances_,
                                                                       classifier_output_dir)

            except Exception as e:
                logger.error(f"Failed to train classifier {class_name} with error: {e}")

        self.train_lstm(logger, Xs_train, Ys_train, best_classifiers_metrics)

        best_classifier_dir = self.model_train_dir / 'best_classifier'
        best_classifier_dir.mkdir(exist_ok=True, parents=True)
        best_classifiers_df = pd.DataFrame.from_dict(best_classifiers_metrics, orient='index',
                                                     columns=['best_index', 'train mcc', 'train auprc', 'train f1',
                                                              'validation mcc', 'validation auprc', 'validation f1',
                                                              'model_path'])
        best_classifiers_df.index.name = 'model_name'
        best_classifiers_df.to_csv(best_classifier_dir / 'best_classifier_from_each_class.csv')
        logger.info(f"Aggregated the best classifiers from each classifier (after hyper-parameter tuning), and saved "
                    f"them to {best_classifier_dir / 'best_classifier_from_each_class.csv'}")

        plot_models_comparison(best_classifiers_df[['validation mcc', 'validation auprc', 'validation f1']].reset_index(),
                               best_classifier_dir, f'Models Comparison - Validation set(s) - {self.number_of_days_to_consider} days')

        best_classifier_class = best_classifiers_df[f'validation {self.config.metric}'].idxmax()
        best_classifier_results = best_classifiers_df.loc[best_classifier_class]
        best_classifier_results.to_csv(best_classifier_dir / 'best_classifier.csv')
        logger.info(f'Best classifier (validation {self.config.metric}): {best_classifier_class}. '
                    f'Saved its metrics to {best_classifier_dir / "best_classifier.csv"}')

        # Copy the best classifier to the best_classifier_dir
        best_model_path = shutil.copy(best_classifier_results['model_path'], best_classifier_dir)

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
        with open(best_classifier_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f)

        return best_model_path

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
                'n_estimators': [5, 20, 100],
                'max_depth': [None, 3, 5, 10],
                'min_samples_split': [2, 5, 10],
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
                'max_depth': [None, 3, 5, 10],
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
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [10, 50, 100],
                'max_depth': [3, 5, 10],
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

        if self.config.balance_classes:
            for grid in decision_tree_grid, rfc_grid, xgboost_grid:
                for key in list(grid.keys()):
                    grid[f'clf__{key}'] = grid.pop(key)

        return [
            # (KNeighborsClassifier(), knn_grid),
            # (LogisticRegression(), logistic_regression_grid),
            # (MLPClassifier(), mlp_grid),
            # (GradientBoostingClassifier(), gbc_grid),
            (RandomForestClassifier(), rfc_grid),
            (DecisionTreeClassifier(), decision_tree_grid),
            (XGBClassifier(), xgboost_grid)
        ]

    def train_lstm(self, logger, Xs_train, Ys_train, best_classifiers_metrics):
        logger.info(f"Training LSTM Classifier with hyperparameters tuning using a fixed train-validation split")

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
                'batch_size': [16]
            }

        all_results = []

        X_train, X_val, y_train, y_val = train_test_split(Xs_train, Ys_train,
                                                          test_size=self.config.nn_validation_set_size,
                                                          random_state=self.config.random_state,
                                                          stratify=Ys_train)
        logger.info(f"Split the train data to train and validation sets (validation ratio is {self.config.nn_validation_set_size})")

        if self.config.balance_classes:
            logger.info(f"Resampling train examples using RandomUnderSampler. Before: {y_train.value_counts()}")
            rus = RandomUnderSampler(random_state=self.config.random_state, sampling_strategy=self.config.max_classes_ratio)
            X_train, y_train = rus.fit_resample(X_train, y_train)
            logger.info(f"After resampling: {y_train.value_counts()}")

        X_train = convert_features_df_to_tensor_for_rnn(X_train)
        X_val = convert_features_df_to_tensor_for_rnn(X_val)

        # Grid search
        for hidden_size, num_layers, lr, batch_size in itertools.product(hyperparameter_grid['hidden_size'],
                                                                         hyperparameter_grid['num_layers'],
                                                                         hyperparameter_grid['lr'],
                                                                         hyperparameter_grid['batch_size']):
            grid_combination_dir = classifier_output_dir / f'hidden_size_{hidden_size}_num_layers_{num_layers}_lr_{lr}_batch_size_{batch_size}'
            grid_combination_dir.mkdir(exist_ok=True, parents=True)

            train_dataset = TensorDataset(X_train, torch.tensor(y_train.values))
            val_dataset = TensorDataset(X_val, torch.tensor(y_val.values))

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)

            model = LSTMModel(hidden_size=hidden_size, num_layers=num_layers, lr=lr)

            checkpoint_callback = ModelCheckpoint(monitor=f'val_{self.config.metric}', mode='max', save_top_k=1,
                                                  dirpath=grid_combination_dir / 'checkpoints', filename='best_model-epoch-{epoch}')
            trainer = L.Trainer(
                max_epochs=self.config.nn_max_epochs,
                logger=False,
                callbacks=[EarlyStopping(monitor=f'val_loss', patience=10, mode='min'), checkpoint_callback],
                deterministic=True
            )
            trainer.fit(model, train_loader, val_loader)

            best_model_path = checkpoint_callback.best_model_path
            best_model = LSTMModel.load_from_checkpoint(best_model_path)
            best_model.eval()

            with torch.no_grad():
                y_train_pred_probs = torch.cat([best_model(x) for x, _ in train_loader]).cpu().numpy().flatten()
                y_train_pred = (y_train_pred_probs > 0.5).astype(int)
                y_val_pred_probs = torch.cat([best_model(x) for x, _ in val_loader]).cpu().numpy().flatten()
                y_val_pred = (y_val_pred_probs > 0.5).astype(int)

            train_mcc = matthews_corrcoef(y_train, y_train_pred)
            train_f1 = f1_score(y_train, y_train_pred)
            train_auprc = average_precision_score(y_train, y_train_pred_probs)
            val_mcc = matthews_corrcoef(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred)
            val_auprc = average_precision_score(y_val, y_val_pred_probs)

            logger.info(f"Trained LSTM with hidden_size={hidden_size}, num_layers={num_layers}, lr={lr}, "
                        f"batch_size={batch_size}. Train MCC: {train_mcc}, Train F1: {train_f1}, Train AUPRC: {train_auprc}, "
                        f"Val MCC: {val_mcc}, Val F1: {val_f1}, Val AUPRC: {val_auprc}")

            # The keys of the metrics are like this to match the ones outputted from GridSearchCV
            all_results.append({'hidden_size': hidden_size, 'num_layers': num_layers, 'lr': lr, 'batch_size': batch_size,
                                'validation mcc': val_mcc, 'validation f1': val_f1, 'validation auprc': val_auprc,
                                'train mcc': train_mcc, 'train f1': train_f1, 'train auprc': train_auprc,
                                'best_model_path': best_model_path})

        # Save all hyperparameter results to CSV
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(classifier_output_dir / 'lstm_grid_results.csv')
        logger.info(f"Hyperparameter search results saved to {classifier_output_dir / 'hyperparameter_results.csv'}")

        # Find the best hyperparameters
        best_model_index = results_df[f'validation {self.config.metric}'].idxmax()
        best_model = results_df.loc[best_model_index]

        logger.info(f"Best LSTM model:\n{best_model}")
        best_classifiers_metrics['LSTMClassifier'] = [best_model_index, best_model['train mcc'],
                                            best_model['train auprc'], best_model['train f1'],
                                            best_model['validation mcc'], best_model['validation auprc'],
                                            best_model['validation f1'], best_model['best_model_path']]


class OptunaModel(Model):
    def run_analysis(self):
        Xs_train, Ys_train, Xs_test, Ys_test = self.create_model_data()

        # Train
        best_model_path, selected_feature_names = self.fit_on_train_data(Xs_train, Ys_train)

        # Test
        Xs_test_selected = Xs_test[selected_feature_names]
        self.test_on_test_data(best_model_path, Xs_test_selected, Ys_test)

    def fit_on_train_data(self, Xs_train, Ys_train):
        feature_names = np.array(Xs_train.columns)

        def objective(trial):
            if self.config.do_feature_selection:
                # Feature Selection: Binary mask (1 = keep, 0 = remove)
                trial_selected_features = [trial.suggest_categorical(f"feature_{i}", [0, 1]) for i in range(Xs_train.shape[1])]

                # Ensure at least one feature is selected
                if sum(trial_selected_features) == 0:
                    return 0.0  # Bad score to discard this selection

                trial_selected_feature_names = feature_names[np.array(trial_selected_features) == 1]
                trial_Xs_train_selected = Xs_train[trial_selected_feature_names]
            else:
                trial_Xs_train_selected = Xs_train

            n_estimators = trial.suggest_int('n_estimators', 1, 100)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
            class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
            bootstrap = trial.suggest_categorical('bootstrap', [True, False])

            max_depth_use_none = trial.suggest_categorical("use_none_max_depth", [True, False])
            if max_depth_use_none:
                max_depth = None  # No depth limit
            else:
                max_depth = trial.suggest_int("max_depth_int", 3, 10)

            # Model training and evaluation
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                class_weight=class_weight,
                bootstrap=bootstrap,
                random_state=self.config.random_state,
                n_jobs=self.config.cpus
            )

            cv = StratifiedKFold(shuffle=True, random_state=self.config.random_state)
            score = cross_val_score(model, trial_Xs_train_selected, Ys_train, cv=cv,
                                    scoring=METRIC_NAME_TO_SKLEARN_SCORER[self.config.metric]).mean()

            return score

        logger = setup_logger(self.model_train_dir / 'classifiers_train.log', f'MODEL_{self.model_id}_TRAIN')
        logger.info(f"Performing feature selection and hyperparameter optimization using Optuna on the training data. "
                    f"Original features are: {feature_names}")
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.config.optuna_number_of_trials, n_jobs=self.config.cpus)

        # Get best hyperparameters and selected features
        best_params = study.best_params
        selected_feature_mask = np.array([best_params.pop(f"feature_{i}") for i in range(Xs_train.shape[1])])
        selected_feature_names = feature_names[selected_feature_mask == 1]
        self.fix_max_depth_param(best_params)

        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Selected features ({len(selected_feature_names)}): {selected_feature_names}")
        logger.info(f"Best trial score (=the mean {self.config.metric} on held-out sets): {study.best_value}")

        best_classifier_dir = self.model_train_dir / 'best_classifier'
        best_classifier_dir.mkdir(exist_ok=True, parents=True)

        # Train final model on selected features
        Xs_train_selected = Xs_train[selected_feature_names]
        final_model = RandomForestClassifier(**best_params, random_state=self.config.random_state, n_jobs=self.config.cpus)
        final_model.fit(Xs_train_selected, Ys_train)

        # Save the trained classifier to disk
        best_model_path = best_classifier_dir / 'best_model.pkl'
        joblib.dump(final_model, best_model_path)

        # Save metadata
        metadata = {
            'numpy_version': np.__version__,
            'joblib_version': joblib.__version__,
            'sklearn_version': sklearn.__version__,
            'pandas_version': pd.__version__,
        }
        with open(best_classifier_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f)

        self.plot_feature_importance(RandomForestClassifier.__name__, Xs_train_selected.columns,
                                     final_model.feature_importances_, best_classifier_dir)

        return best_model_path, selected_feature_names

    @staticmethod
    def fix_max_depth_param(params):
        if params.pop('use_none_max_depth'):
            params['max_depth'] = None
        else:
            params['max_depth'] = params.pop('max_depth_int')
