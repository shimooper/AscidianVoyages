import itertools
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
from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, TensorDataset
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


import optuna

from utils import setup_logger, convert_pascal_to_snake_case, get_column_groups_sorted, convert_columns_to_int, \
    get_lived_columns_to_consider
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

    def convert_routes_to_model_data(self, df, number_of_days_to_consider):
        lived_columns, temperature_columns, salinity_columns = get_column_groups_sorted(df)

        days_data = []
        for index, row in df.iterrows():
            for col in lived_columns[-1:2:-1]:
                col_day = int(col.split(' ')[1])
                if pd.isna(row[f'Lived {col_day}']):
                    continue

                temperature_columns = {f'Day {-i} Temperature': row[f'Temp {col_day - i}'] for i in range(number_of_days_to_consider)}
                salinity_columns = {f'Day {-i} Salinity': row[f'Salinity {col_day - i}'] for i in range(number_of_days_to_consider)}
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
        model_train_df = self.convert_routes_to_model_data(train_df, self.number_of_days_to_consider)
        model_train_df.to_csv(self.model_data_dir / 'train.csv', index=False)

        test_df = pd.read_csv(self.config.data_dir_path / 'test.csv')
        model_test_df = self.convert_routes_to_model_data(test_df, self.number_of_days_to_consider)
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

        model = joblib.load(model_path)
        Ys_test_predictions = model.predict_proba(Xs_test)
        mcc_on_test = matthews_corrcoef(Ys_test, Ys_test_predictions.argmax(axis=1))
        auprc_on_test = average_precision_score(Ys_test, Ys_test_predictions[:, 1])
        f1_on_test = f1_score(Ys_test, Ys_test_predictions.argmax(axis=1))

        logger.info(
            f"Best estimator - MCC on test: {mcc_on_test}, AUPRC on test: {auprc_on_test}, F1 on test: {f1_on_test}")

        test_results = pd.DataFrame({'mcc': [mcc_on_test], 'auprc': [auprc_on_test], 'f1': [f1_on_test]})
        test_results.to_csv(self.model_test_dir / 'best_classifier_test_results.csv', index=False)


class ScikitModel(Model):
    def run_analysis(self):
        Xs_train, Ys_train, Xs_test, Ys_test = self.create_model_data()

        # Train
        train_logger = setup_logger(self.model_train_dir / 'classifiers_train.log', f'MODEL_{self.model_id}_TRAIN')
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
        best_classifiers = {}
        best_classifiers_metrics = {}
        classifiers = self.create_classifiers_and_param_grids()
        for classifier, param_grid in classifiers:
            class_name = classifier.__class__.__name__
            classifier_output_dir = self.model_train_dir / convert_pascal_to_snake_case(class_name)
            classifier_output_dir.mkdir(exist_ok=True, parents=True)

            logger.info(f"Training Classifier {class_name} with hyperparameters tuning using GridSearch and Stratified-KFold CV.")
            grid = GridSearchCV(
                estimator=classifier,
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
                best_classifiers[class_name] = grid.best_estimator_
                joblib.dump(grid.best_estimator_, classifier_output_dir / f"best_{class_name}.pkl")

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
                                                        grid_results['mean_test_f1'][grid.best_index_])

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

        best_classifier_dir = self.model_train_dir / 'best_classifier'
        best_classifier_dir.mkdir(exist_ok=True, parents=True)
        best_classifiers_df = pd.DataFrame.from_dict(best_classifiers_metrics, orient='index',
                                                     columns=['best_index', 'mean_mcc_on_train_folds',
                                                              'mean_auprc_on_train_folds', 'mean_f1_on_train_folds',
                                                              'mean_mcc_on_held_out_folds',
                                                              'mean_auprc_on_held_out_folds',
                                                              'mean_f1_on_held_out_folds'])
        best_classifiers_df.index.name = 'classifier_class'
        best_classifiers_df.to_csv(best_classifier_dir / 'best_classifier_from_each_class.csv')

        best_classifier_class = best_classifiers_df[f'mean_{self.config.metric}_on_held_out_folds'].idxmax()
        logger.info(f'Best classifier (according to mean_{self.config.metric}_on_held_out_folds): {best_classifier_class}')

        # Save the best classifier to disk
        best_model_path = best_classifier_dir / 'best_model.pkl'
        joblib.dump(best_classifiers[best_classifier_class], best_model_path)

        # Save metadata
        metadata = {
            'numpy_version': np.__version__,
            'joblib_version': joblib.__version__,
            'sklearn_version': sklearn.__version__,
            'pandas_version': pd.__version__,
        }
        with open(best_classifier_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f)

        best_classifier_metrics = best_classifiers_df.loc[[best_classifier_class]].reset_index()
        best_classifier_metrics.to_csv(best_classifier_dir / 'best_classifier_train_results.csv', index=False)

        return best_model_path

    def create_classifiers_and_param_grids(self):
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

        rfc_grid = {
            'n_estimators': [5, 20, 100],
            'max_depth': [None, 3, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'random_state': [self.config.random_state],
            'class_weight': ['balanced', None],
            'bootstrap': [True, False]
        }

        rfc_grid_debug = {
            'n_estimators': [5, 20],
            'max_depth': [None],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'random_state': [self.config.random_state],
            'class_weight': ['balanced', None],
            'bootstrap': [True]
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

        decision_tree_grid = {
            'max_depth': [None, 3, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'random_state': [self.config.random_state],
            'class_weight': ['balanced', None],
        }

        decision_tree_grid_debug = {
            'max_depth': [None, 5],
            'min_samples_split': [2],
            'min_samples_leaf': [1],
            'random_state': [self.config.random_state],
            'class_weight': ['balanced', None],
        }

        xgboost_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [10, 50, 200],
            'max_depth': [3, 5, 10],
            'booster': ['dart'],
            'n_jobs': [1],
            'random_state': [self.config.random_state],
            'subsample': [0.6, 1],
        }

        xgboost_grid_debug = {
            'learning_rate': [0.01],
            'n_estimators': [10],
            'max_depth': [3],
            'booster': ['dart'],
            'n_jobs': [1],
            'random_state': [self.config.random_state],
            'subsample': [1],
        }

        return [
            # (KNeighborsClassifier(), knn_grid),
            # (LogisticRegression(), logistic_regression_grid),
            # (MLPClassifier(), mlp_grid),
            (RandomForestClassifier(), rfc_grid if not DEBUG_MODE else rfc_grid_debug),
            # (GradientBoostingClassifier(), gbc_grid),
            (DecisionTreeClassifier(), decision_tree_grid if not DEBUG_MODE else decision_tree_grid_debug),
            (XGBClassifier(), xgboost_grid if not DEBUG_MODE else xgboost_grid_debug)
        ]

    def train_lstm(self, logger, Xs_train, Ys_train):
        seed_everything(self.config.random_state, workers=True)

        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(Xs_train, Ys_train,
                                                          test_size=self.config.nn_validation_set_size,
                                                          random_state=self.config.random_state, stratify=Ys_train)
        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1))
        val_dataset = TensorDataset(torch.tensor(X_val), torch.tensor(y_val).unsqueeze(1))

        # Hyperparameter grid
        hyperparameter_grid = {
            'hidden_size': [8, 16, 32],
            'lr': [1e-4, 1e-3, 1e-2],
            'batch_size': [16, 32, 64]
        }

        all_results = []

        # Grid search
        for hidden_size, lr, batch_size in zip(hyperparameter_grid['hidden_size'], hyperparameter_grid['lr'],
                                               hyperparameter_grid['batch_size']):
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model = LSTMModel(hidden_size=hidden_size, lr=lr)
            classifier_output_dir = self.model_train_dir / 'lstm'
            checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1,
                                                  dirpath=classifier_output_dir / 'checkpoints')
            trainer = Trainer(
                max_epochs=self.config.nn_max_epochs,
                logger=False,
                callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min'), checkpoint_callback],
                deterministic=True
            )
            trainer.fit(model, train_loader, val_loader)

            best_model_path = checkpoint_callback.best_model_path
            model.load_state_dict(torch.load(best_model_path))
            model.eval()

            with torch.no_grad():
                y_train_pred_probs = torch.cat([model(x) for x, _ in train_loader]).cpu().numpy().flatten()
                y_train_pred = (y_train_pred_probs > 0.5).astype(int)
                y_val_pred_probs = torch.cat([model(x) for x, _ in val_loader]).cpu().numpy().flatten()
                y_val_pred = (y_val_pred_probs > 0.5).astype(int)

            train_mcc = matthews_corrcoef(y_train, y_train_pred)
            train_f1 = f1_score(y_train, y_train_pred)
            train_auprc = average_precision_score(y_train, y_train_pred_probs)
            val_mcc = matthews_corrcoef(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred)
            val_auprc = average_precision_score(y_val, y_val_pred_probs)

            all_results.append(
                {'hidden_size': hidden_size, 'lr': lr, 'batch_size': batch_size, 'val_mcc': val_mcc, 'val_f1': val_f1,
                 'val_auprc': val_auprc, 'train_mcc': train_mcc, 'train_f1': train_f1, 'train_auprc': train_auprc})

        # Save all hyperparameter results to CSV
        results_df = pd.DataFrame(all_results)
        results_df.to_csv('hyperparameter_results.csv', index=False)
        print("Hyperparameter search results saved to 'hyperparameter_results.csv'")

        print(f"Best hyperparameters: {best_params}")

        # Retrain on the full dataset with the best hyperparameters
        final_dataset = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(1))
        final_loader = DataLoader(final_dataset, batch_size=best_params['batch_size'], shuffle=True)

        final_model = LSTMModel(hidden_size=best_params['hidden_size'], lr=best_params['lr'])
        checkpoint_callback_final = ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, dirpath='checkpoints',
                                                    filename='best_final_model')
        final_trainer = pl.Trainer(
            max_epochs=20,
            enable_checkpointing=True,
            logger=False,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min'), checkpoint_callback_final]
        )
        final_trainer.fit(final_model, final_loader)

        # Load best final model before saving
        best_final_model_path = checkpoint_callback_final.best_model_path
        final_model.load_state_dict(torch.load(best_final_model_path))
        final_model.eval()

        # Save final model
        final_model_state = final_model.state_dict()
        torch.save(final_model_state, 'final_lstm_model.pth')
        print("Final model saved as 'final_lstm_model.pth'")


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
