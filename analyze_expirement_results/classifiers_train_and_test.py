import joblib
import pandas as pd
import os
import json

import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, matthews_corrcoef

from classifiers_params_grids import classifiers
from utils import setup_logger


def fit_on_train_data(Xs_train, Ys_train, output_dir, n_jobs, logger_name):
    logger = setup_logger(os.path.join(output_dir, 'classifier_training.log'), logger_name)

    best_classifiers = {}
    best_classifiers_metrics = {}
    for classifier, param_grid in classifiers:
        class_name = classifier.__class__.__name__
        logger.info(f"Training Classifier {class_name} with hyperparameters tuning using Stratified-KFold CV.")
        grid = GridSearchCV(
            estimator=classifier,
            param_grid=param_grid,
            scoring={'mcc': make_scorer(matthews_corrcoef), 'f1': 'f1', 'auprc': 'average_precision'},
            refit='mcc',
            return_train_score=True,
            verbose=1,
            n_jobs=n_jobs
        )

        try:
            grid.fit(Xs_train, Ys_train)
            grid_results = pd.DataFrame.from_dict(grid.cv_results_)
            grid_results.to_csv(os.path.join(output_dir, f'{class_name}_grid_results.csv'))
            best_classifiers[class_name] = grid.best_estimator_
            joblib.dump(grid.best_estimator_, os.path.join(output_dir, f"best_{class_name}.pkl"))

            # Note: grid.best_score_ == grid_results['mean_test_mcc'][grid.best_index_] (the mean cross-validated score of the best_estimator)
            logger.info(f"Best params: {grid.best_params_}, Best index: {grid.best_index_}, Best score: {grid.best_score_}")

            logger.info(f"Best estimator - Mean MCC on train folds: {grid_results['mean_train_mcc'][grid.best_index_]}, "
                         f"Mean AUPRC on train folds: {grid_results['mean_train_auprc'][grid.best_index_]}, "
                         f"Mean F1 on train fold: {grid_results['mean_train_f1'][grid.best_index_]}"
                         f"Mean MCC on held-out folds: {grid_results['mean_test_mcc'][grid.best_index_]}, "
                         f"Mean AUPRC on held-out folds: {grid_results['mean_test_auprc'][grid.best_index_]}, "
                         f"Mean F1 o held-out folds: {grid_results['mean_test_f1'][grid.best_index_]}")

            best_classifiers_metrics[class_name] = (grid.best_index_,
                                                    grid_results['mean_train_mcc'][grid.best_index_],
                                                    grid_results['mean_train_auprc'][grid.best_index_],
                                                    grid_results['mean_train_f1'][grid.best_index_],
                                                    grid.best_score_,
                                                    grid_results['mean_test_auprc'][grid.best_index_],
                                                    grid_results['mean_test_f1'][grid.best_index_])
        except Exception as e:
            logger.error(f"Failed to train classifier {class_name} with error: {e}")

    best_classifiers_df = pd.DataFrame.from_dict(best_classifiers_metrics, orient='index',
                                                 columns=['best_index', 'mean_mcc_on_train_folds',
                                                          'mean_auprc_on_train_folds', 'mean_f1_on_train_folds',
                                                          'mean_mcc_on_held_out_folds', 'mean_auprc_on_held_out_folds',
                                                          'mean_f1_on_held_out_folds'])
    best_classifiers_df.index.name = 'classifier_class'
    best_classifiers_df.to_csv(os.path.join(output_dir, 'best_classifier_from_each_class.csv'))

    best_classifier_class = best_classifiers_df['mean_mcc_on_held_out_folds'].idxmax()
    logger.info(f"Best classifier (according to mean_mcc_on_held_out_folds): {best_classifier_class}")

    # Save the best classifier to disk
    joblib.dump(best_classifiers[best_classifier_class], os.path.join(output_dir, "best_model.pkl"))

    # Save metadata
    metadata = {
        'numpy_version': np.__version__,
        'joblib_version': joblib.__version__,
        'sklearn_version': sklearn.__version__,
        'pandas_version': pd.__version__,
    }
    with open(os.path.join(output_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f)

    best_classifier_metrics = best_classifiers_df.loc[[best_classifier_class]].reset_index()
    best_classifier_metrics.to_csv(os.path.join(output_dir, 'best_classifier_train_results.csv'), index=False)
