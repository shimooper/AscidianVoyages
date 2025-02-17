import itertools
import joblib
import pandas as pd
import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import matthews_corrcoef, average_precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import RFECV

from utils import setup_logger, convert_pascal_to_snake_case, get_column_groups_sorted, convert_columns_to_int, \
    get_lived_columns_to_consider
from configuration import METRIC_NAME_TO_SKLEARN_SCORER, DEBUG_MODE, Config, DO_FEATURE_SELECTION


class Model:
    def __init__(self, config: Config, model_id, number_of_future_days_to_consider_death):
        self.config = config
        self.model_id = model_id
        self.number_of_future_days_to_consider_death = number_of_future_days_to_consider_death

        output_dir_path = config.models_dir_path / f'future_days_to_consider_death_{number_of_future_days_to_consider_death}'
        self.model_data_dir = output_dir_path / 'data'
        self.model_train_dir = output_dir_path / 'train_outputs'
        self.model_test_dir = output_dir_path / 'test_outputs'
        self.model_data_dir.mkdir(exist_ok=True, parents=True)
        self.model_train_dir.mkdir(exist_ok=True, parents=True)
        self.model_test_dir.mkdir(exist_ok=True, parents=True)

        self.classifiers = self.create_classifiers_and_param_grids()

    def convert_routes_to_model_data(self, df, number_of_future_days_to_consider_death):
        lived_columns, temperature_columns, salinity_columns = get_column_groups_sorted(df)

        four_days_data = []
        for index, row in df.iterrows():
            for col in lived_columns[-1:2:-1]:
                col_day = int(col.split(' ')[1])
                if pd.isna(row[f'Lived {col_day}']):
                    continue

                temperature_columns = [f'Temp {col_day - i}' for i in range(4)]
                salinity_columns = [f'Salinity {col_day - i}' for i in range(4)]
                lived_cols_to_consider = get_lived_columns_to_consider(row, col_day, number_of_future_days_to_consider_death)
                new_row = {
                    'current day temperature': row[f'Temp {col_day}'],
                    'previous day temperature': row[f'Temp {col_day - 1}'],
                    '2 days ago temperature': row[f'Temp {col_day - 2}'],
                    '3 days ago temperature': row[f'Temp {col_day - 3}'],
                    'max temperature': row[temperature_columns].max(),
                    'min temperature': row[temperature_columns].min(),
                    'current day salinity': row[f'Salinity {col_day}'],
                    'previous day salinity': row[f'Salinity {col_day - 1}'],
                    '2 days ago salinity': row[f'Salinity {col_day - 2}'],
                    '3 days ago salinity': row[f'Salinity {col_day - 3}'],
                    'max salinity': row[salinity_columns].max(),
                    'min salinity': row[salinity_columns].min(),
                    'death': any(row[lived_cols_to_consider]),
                }
                four_days_data.append(new_row)

        four_days_df = pd.DataFrame(four_days_data)
        convert_columns_to_int(four_days_df)

        return four_days_df

    def create_model_data(self):
        train_df = pd.read_csv(self.config.data_dir_path / 'train.csv')
        model_train_df = self.convert_routes_to_model_data(train_df, self.number_of_future_days_to_consider_death)
        model_train_df.to_csv(self.model_data_dir / 'train.csv', index=False)

        test_df = pd.read_csv(self.config.data_dir_path / 'test.csv')
        model_test_df = self.convert_routes_to_model_data(test_df, self.number_of_future_days_to_consider_death)
        model_test_df.to_csv(self.model_data_dir / 'test.csv', index=False)

        return model_train_df, model_test_df

    def plot_feature_pairs(self, train_df, test_df):
        full_df = pd.concat([train_df, test_df], axis=0)

        full_df['death_label'] = full_df['death'].map({1: 'Dead', 0: 'Alive'}).astype('category')
        full_df.drop(columns=['death'], inplace=True)

        sns.pairplot(full_df, hue='death_label', palette={'Dead': 'red', 'Alive': 'green'})

        plt.grid(alpha=0.3)
        plt.savefig(self.model_data_dir / "scatter_plot.png", dpi=300, bbox_inches='tight')
        plt.close()

    def run_analysis(self):
        model_train_df, model_test_df = self.create_model_data()
        if not DEBUG_MODE:
            self.plot_feature_pairs(model_train_df, model_test_df)

        # Train
        train_logger = setup_logger(self.model_train_dir / 'classifiers_train.log', f'MODEL_{self.model_id}_TRAIN')
        Xs_train = model_train_df.drop(columns=['death'])
        Ys_train = model_train_df['death']

        if DO_FEATURE_SELECTION:
            selected_features_mask = self.feature_selection_on_train_data(train_logger, Xs_train, Ys_train)
        else:
            selected_features_mask = [True] * len(Xs_train.columns)

        Xs_train_selected = Xs_train.loc[:, selected_features_mask]
        best_model_path = self.fit_on_train_data(train_logger, Xs_train_selected, Ys_train)

        # Test
        Xs_test = model_test_df.drop(columns=['death'])
        Xs_test_selected = Xs_test.loc[:, selected_features_mask]
        Ys_test = model_test_df['death']
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
        for classifier, param_grid in self.classifiers:
            class_name = classifier.__class__.__name__
            classifier_output_dir = self.model_train_dir / convert_pascal_to_snake_case(class_name)
            classifier_output_dir.mkdir(exist_ok=True, parents=True)

            logger.info(f"Training Classifier {class_name} with hyperparameters tuning using Stratified-KFold CV.")
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

                if class_name in ['DecisionTreeClassifier', 'RandomForestClassifier', 'GradientBoostingClassifier']:
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

    def plot_feature_importance(self, classifier_name, feature_names, features_importance, output_dir):
        indices = np.argsort(features_importance)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importance")
        plt.bar(list(range(len(feature_names))), features_importance[indices], align="center")
        plt.xticks(list(range(len(feature_names))), [feature_names[i] for i in indices], rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / f'{classifier_name}_feature_importance.png', dpi=600)
        plt.close()

    def plot_decision_tree(self, model, feature_names, output_dir):
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

    def plot_decision_functions_of_features_pairs(self, Xs_train, Ys_train, best_params, feature_importances, output_dir):
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

        return [
            # (KNeighborsClassifier(), knn_grid),
            # (LogisticRegression(), logistic_regression_grid),
            # (MLPClassifier(), mlp_grid),
            (RandomForestClassifier(), rfc_grid if not DEBUG_MODE else rfc_grid_debug),
            # (GradientBoostingClassifier(), gbc_grid),
            (DecisionTreeClassifier(), decision_tree_grid if not DEBUG_MODE else decision_tree_grid_debug)
        ]
