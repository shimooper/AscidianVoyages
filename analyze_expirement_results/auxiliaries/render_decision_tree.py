from graphviz import Source
from pathlib import Path

DOT_FILE = Path(r"C:\repos\GoogleShips\analyze_expirement_results\outputs_cv_test_25\configuration_0\models\3_day_interval\train_outputs\decision_tree_classifier\DecisionTreeClassifier_plot_change_colors.dot")

# Load the .dot content (as a string or from a file)
with open(DOT_FILE) as f:
    dot_source = f.read()

# Create a Graphviz Source object
graph = Source(dot_source)

# Save as PNG
graph.render(DOT_FILE.parent / f"{DOT_FILE.stem}", format="png", cleanup=True)
