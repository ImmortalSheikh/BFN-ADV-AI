import os
import json
import pickle
from datetime import date

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

from data import build_features, FEATURE_COLS
from model import build_model

SAVE_PATH  = os.path.join(os.path.dirname(__file__), "model", "reorder_model.pkl")
META_PATH  = os.path.join(os.path.dirname(__file__), "model", "model_metadata.json")


def train():
    os.makedirs("model", exist_ok=True)

    # Load data
    print("Loading dataset...\n")
    X, y, features_df, cat_encoder = build_features()

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain samples : {len(X_train)}")
    print(f"Test samples  : {len(X_test)}\n")

    # Build and train model
    model = build_model()

    print("\nStarting Training...\n")
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc     = accuracy_score(y_test, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    print("=" * 50)
    print(f"  Test Accuracy      : {acc:.2%}")
    print(f"  5-Fold CV Accuracy : {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
    print("=" * 50)
    print()
    print(classification_report(y_test, y_pred, target_names=["No Reorder", "Will Reorder"]))

    # Feature importances
    feature_names = [
        "Order Count", "Days Since Last", "Avg Quantity",
        "Order Frequency", "Week of Year", "Category"
    ]
    importances = dict(zip(feature_names, model.feature_importances_.tolist()))
    print("Feature Importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {feat:22s}: {imp*100:.1f}%")

    # Save model
    with open(SAVE_PATH, "wb") as f:
        pickle.dump({"model": model, "cat_encoder": cat_encoder}, f)

    # Save metadata
    metadata = {
        "model_type":        "RandomForestClassifier",
        "accuracy":          round(acc * 100, 2),
        "cv_accuracy_mean":  round(cv_scores.mean() * 100, 2),
        "cv_accuracy_std":   round(cv_scores.std() * 100, 2),
        "trained_on":        date.today().isoformat(),
        "train_samples":     len(X_train),
        "test_samples":      len(X_test),
        "feature_names":     FEATURE_COLS,
        "feature_importances": {
            name: round(float(imp) * 100, 2)
            for name, imp in zip(feature_names, model.feature_importances_)
        },
        "explainability": {
            "order_count":      "How many times the customer ordered this product",
            "days_since_last":  "Days since the customer last ordered this product",
            "avg_quantity":     "Average quantity the customer orders each time",
            "order_frequency":  "How often this product appears in the customer orders",
            "week_of_year":     "Seasonal factor based on time of year",
            "category_encoded": "Product category (vegetables, fruit, dairy, bakery)",
        },
        "fairness_note": (
            "Suggestions are based solely on individual customer purchase history. "
            "No producer-level features are used, preventing producer bias."
        ),
        "monitoring_strategy": (
            "Customer override events are logged. "
            "If override rate exceeds 30%, model retraining is recommended. "
            "Accuracy is recalculated monthly against actual orders placed."
        ),
    }

    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to      : {SAVE_PATH}")
    print(f"Metadata saved to   : {META_PATH}")
    print("\nTraining Complete!")


if __name__ == "__main__":
    train()
