import pandas as pd
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import matthews_corrcoef
import pickle

OUTPUTS_PATH = Path(r"C:\repos\GoogleShips\analyze_expirement_results\outputs_cv_test_25\configuration_0")
MODEL_DIR = OUTPUTS_PATH / "models" / "3_day_interval"
DATA_DIR = MODEL_DIR / "data"
DECISION_TREE_PKL = MODEL_DIR / "train_outputs" / "decision_tree_classifier" / "best_DecisionTreeClassifier.pkl"


def main():
    model = joblib.load(DECISION_TREE_PKL)
    raw_df = pd.read_csv(DATA_DIR / "train.csv")
    features_df = pd.read_csv(DATA_DIR / "Xs_train_features.csv")

    y_pred_full = model.predict(features_df)
    mcc_full = matthews_corrcoef(raw_df['death'], y_pred_full)

    print(f"Full MCC: {mcc_full}")

    # Extract root node details
    tree = model.tree_
    feature_index = tree.feature[0]
    threshold = tree.threshold[0]

    # Manually apply the root decision rule
    # Get predicted class from left or right child
    left_class = np.argmax(tree.value[tree.children_left[0]])
    right_class = np.argmax(tree.value[tree.children_right[0]])

    # Apply the rule to test set
    y_pred_root_only = np.where(features_df.iloc[:, feature_index] <= threshold, left_class, right_class)

    # Compute MCC
    mcc_root_only = matthews_corrcoef(raw_df['death'], y_pred_root_only)

    print(f"MCC (root node only): {mcc_root_only:.3f}")


if __name__ == "__main__":
    main()

