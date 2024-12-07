from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from utils import RANDOM_STATE


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
    'random_state': [RANDOM_STATE],
    'class_weight': ['balanced', None],
}

mlp_grid = {
    'hidden_layer_sizes': [(10,3),(30,5),],
    'activation': ['tanh', 'relu'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [400],
    'early_stopping': [True],
    'random_state': [RANDOM_STATE],
}

rfc_grid = {
        'n_estimators': [5, 20],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 4],
        'random_state': [RANDOM_STATE],
        'class_weight': ['balanced', None],
        'bootstrap': [True, False]
}

gbc_grid = {
    'learning_rate': [0.05, 0.1],
    'n_estimators': [10, 50, 200],
    'max_depth': [3, 5, 10],
    'random_state': [RANDOM_STATE],
    'subsample': [0.6, 1],
}

decision_tree_grid = {
    'max_depth': [None, 3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'random_state': [RANDOM_STATE],
    'class_weight': ['balanced', None],
}


classifiers = [
    # (KNeighborsClassifier(), knn_grid),
    # (LogisticRegression(), logistic_regression_grid),
    # (MLPClassifier(), mlp_grid),
    (RandomForestClassifier(), rfc_grid),
    # (GradientBoostingClassifier(), gbc_grid),
    (DecisionTreeClassifier(), decision_tree_grid)
]
