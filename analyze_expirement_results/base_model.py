import itertools
import os
import joblib
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt

import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, matthews_corrcoef, average_precision_score, f1_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from utils import setup_logger, convert_pascal_to_snake_case


class BaseModel:
    def __init__(self, model_name, all_outputs_dir_path, metric_to_choose_best_model, random_state, model_id):
        self.model_name = model_name
        self.metric_to_choose_best_model = metric_to_choose_best_model
        self.train_file_path = all_outputs_dir_path / 'train.csv'
        self.test_file_path = all_outputs_dir_path / 'test.csv'

        self.output_dir_path = all_outputs_dir_path / 'models' / model_name
        self.model_data_dir = self.output_dir_path / 'data'
        self.model_train_set_path = self.model_data_dir / 'train.csv'
        self.model_test_set_path = self.model_data_dir / 'test.csv'
        self.model_train_dir = self.output_dir_path / 'train_outputs'
        self.model_test_dir = self.output_dir_path / 'test_outputs'
        os.makedirs(self.output_dir_path, exist_ok=True)
        os.makedirs(self.model_data_dir, exist_ok=True)
        os.makedirs(self.model_train_dir, exist_ok=True)
        os.makedirs(self.model_test_dir, exist_ok=True)

        self.create_classifiers_and_param_grids(random_state)
        self.model_id = model_id

    def convert_routes_to_model_data(self, df):
        raise NotImplementedError

    def create_model_data(self):
        train_df = pd.read_csv(self.train_file_path)
        model_train_df = self.convert_routes_to_model_data(train_df)
        model_train_df.to_csv(self.model_train_set_path, index=False)

        test_df = pd.read_csv(self.test_file_path)
        model_test_df = self.convert_routes_to_model_data(test_df)
        model_test_df.to_csv(self.model_test_set_path, index=False)

        return model_train_df, model_test_df

    def run_analysis(self):
        model_train_df, model_test_df = self.create_model_data()

        self.fit_on_train_data(model_train_df.drop(columns=['death']), model_train_df['death'], -1)
        self.test_on_test_data(self.model_train_dir / 'best_model.pkl', model_test_df.drop(columns=['death']),
                               model_test_df['death'])

    def fit_on_train_data(self, Xs_train, Ys_train, n_jobs):
        logger = setup_logger(self.model_train_dir / 'classifiers_train.log', f'MODEL_{self.model_id}_TRAIN')

        best_classifiers = {}
        best_classifiers_metrics = {}
        for classifier, param_grid in self.classifiers:
            class_name = classifier.__class__.__name__
            classifier_output_dir = self.model_train_dir / convert_pascal_to_snake_case(class_name)
            os.makedirs(classifier_output_dir, exist_ok=True)
            logger.info(f"Training Classifier {class_name} with hyperparameters tuning using Stratified-KFold CV.")
            grid = GridSearchCV(
                estimator=classifier,
                param_grid=param_grid,
                scoring={'mcc': make_scorer(matthews_corrcoef), 'f1': 'f1', 'auprc': 'average_precision'},
                refit=self.metric_to_choose_best_model,
                return_train_score=True,
                verbose=1,
                n_jobs=n_jobs
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

                if class_name == 'DecisionTreeClassifier':
                    self.plot_decision_tree(grid.best_estimator_, Xs_train.columns, classifier_output_dir)
                    if len(Xs_train.columns) >= 2:
                        self.plot_decision_functions_of_features_pairs(Xs_train, Ys_train, grid.best_params_, classifier_output_dir)

            except Exception as e:
                logger.error(f"Failed to train classifier {class_name} with error: {e}")

        best_classifiers_df = pd.DataFrame.from_dict(best_classifiers_metrics, orient='index',
                                                     columns=['best_index', 'mean_mcc_on_train_folds',
                                                              'mean_auprc_on_train_folds', 'mean_f1_on_train_folds',
                                                              'mean_mcc_on_held_out_folds',
                                                              'mean_auprc_on_held_out_folds',
                                                              'mean_f1_on_held_out_folds'])
        best_classifiers_df.index.name = 'classifier_class'
        best_classifiers_df.to_csv(self.model_train_dir / 'best_classifier_from_each_class.csv')

        best_classifier_class = best_classifiers_df[f'mean_{self.metric_to_choose_best_model}_on_held_out_folds'].idxmax()
        logger.info(f"Best classifier (according to mean_{self.metric_to_choose_best_model}_on_held_out_folds): {best_classifier_class}")

        # Save the best classifier to disk
        joblib.dump(best_classifiers[best_classifier_class], self.model_train_dir / "best_model.pkl")

        # Save metadata
        metadata = {
            'numpy_version': np.__version__,
            'joblib_version': joblib.__version__,
            'sklearn_version': sklearn.__version__,
            'pandas_version': pd.__version__,
        }
        with open(self.model_train_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f)

        best_classifier_metrics = best_classifiers_df.loc[[best_classifier_class]].reset_index()
        best_classifier_metrics.to_csv(self.model_train_dir / 'best_classifier_train_results.csv', index=False)

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

    def plot_decision_functions_of_features_pairs(self, Xs_train, Ys_train, best_params, output_dir):
        n_classes = 2
        plot_colors = "gr"
        plot_step = 0.02

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

    def create_classifiers_and_param_grids(self, random_state):
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
            'random_state': [random_state],
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
            'random_state': [random_state],
        }

        rfc_grid = {
            'n_estimators': [5, 20, 100],
            'max_depth': [None, 3, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'random_state': [random_state],
            'class_weight': ['balanced', None],
            'bootstrap': [True, False]
        }

        gbc_grid = {
            'n_estimators': [5, 20, 100],
            'max_depth': [None, 3, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'random_state': [random_state],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.6, 1],
        }

        decision_tree_grid = {
            'max_depth': [None, 3, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 5],
            'random_state': [random_state],
            'class_weight': ['balanced', None],
        }

        self.classifiers = [
            # (KNeighborsClassifier(), knn_grid),
            # (LogisticRegression(), logistic_regression_grid),
            # (MLPClassifier(), mlp_grid),
            (RandomForestClassifier(), rfc_grid),
            # (GradientBoostingClassifier(), gbc_grid),
            (DecisionTreeClassifier(), decision_tree_grid)
        ]
