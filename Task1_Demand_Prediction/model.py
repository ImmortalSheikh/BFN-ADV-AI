from sklearn.ensemble import RandomForestClassifier


def build_model():
    """
    Build a Random Forest Classifier for demand prediction.

    Hyperparameter choices:
        n_estimators=100    - 100 decision trees, balances performance and speed
        max_depth=8         - limits tree depth to prevent overfitting
        min_samples_split=5 - requires at least 5 samples to split a node
        class_weight=balanced - adjusts for imbalance between reorder/no-reorder classes
        random_state=42     - ensures reproducibility
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
    )

    print("Model: Random Forest Classifier")
    print(f"  n_estimators    : {model.n_estimators}")
    print(f"  max_depth       : {model.max_depth}")
    print(f"  min_samples_split: {model.min_samples_split}")
    print(f"  class_weight    : {model.class_weight}")

    return model
