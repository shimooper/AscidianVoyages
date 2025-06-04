import optuna
import numpy as np


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
