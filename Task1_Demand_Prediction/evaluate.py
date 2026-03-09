import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, RocCurveDisplay,
    precision_score, recall_score, f1_score,
)
from sklearn.model_selection import cross_val_score

from data import build_features, FEATURE_COLS, HISTORY_PATH, PRODUCTS

SAVE_PATH   = os.path.join(os.path.dirname(__file__), "model", "reorder_model.pkl")
META_PATH   = os.path.join(os.path.dirname(__file__), "model", "model_metadata.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "evaluation_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 120


def load_model():
    if not os.path.exists(SAVE_PATH):
        raise FileNotFoundError(
            f"No model found at {SAVE_PATH}. Run train.py first."
        )
    with open(SAVE_PATH, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["cat_encoder"]


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Reorder", "Will Reorder"]
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved to {path}")


def plot_roc_curve(y_test, y_proba, auc):
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(
        y_test, y_proba, ax=ax,
        name=f"Random Forest (AUC = {auc:.2f})",
        color="#2d6a2d"
    )
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ROC curve saved to {path}")


def plot_cross_validation(cv_scores):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(range(1, 6), cv_scores, color="steelblue", alpha=0.8, edgecolor="white")
    ax.axhline(
        cv_scores.mean(), color="red", linestyle="--",
        label=f"Mean: {cv_scores.mean():.2%}"
    )
    ax.set_title("5-Fold Cross-Validation Accuracy", fontsize=14, fontweight="bold")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "cross_validation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Cross-validation chart saved to {path}")


def plot_feature_importance(model):
    feature_names = [
        "Order Count", "Days Since Last", "Avg Quantity",
        "Order Frequency", "Week of Year", "Category"
    ]
    importances = model.feature_importances_
    sorted_idx  = np.argsort(importances)[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Feature Importance — What drives reorder predictions?",
        fontsize=13, fontweight="bold"
    )

    axes[0].barh(
        [feature_names[i] for i in sorted_idx[::-1]],
        importances[sorted_idx[::-1]],
        color=["#2d6a2d" if i == sorted_idx[0] else "steelblue"
               for i in range(len(feature_names))][::-1]
    )
    axes[0].set_xlabel("Importance Score")
    axes[0].set_title("Feature Importances")

    axes[1].pie(
        importances[sorted_idx],
        labels=[feature_names[i] for i in sorted_idx],
        autopct="%1.1f%%",
        colors=sns.color_palette("muted", len(feature_names))
    )
    axes[1].set_title("Relative Contribution")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Feature importance chart saved to {path}")


def plot_eda(orders_df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Purchase History — Exploratory Data Analysis",
                 fontsize=15, fontweight="bold")

    # Top products
    top = orders_df["product"].value_counts().head(10)
    axes[0, 0].barh(top.index[::-1], top.values[::-1], color="steelblue")
    axes[0, 0].set_title("Top 10 Most Ordered Products")
    axes[0, 0].set_xlabel("Number of Orders")

    # Category breakdown
    cat_counts = orders_df["category"].value_counts()
    axes[0, 1].pie(
        cat_counts.values, labels=cat_counts.index, autopct="%1.1f%%",
        colors=["#4CAF50", "#FF9800", "#2196F3", "#9C27B0"]
    )
    axes[0, 1].set_title("Orders by Category")

    # Weekly volume
    weekly = orders_df.groupby("week_number").size()
    axes[1, 0].plot(weekly.index, weekly.values, marker="o",
                    color="#2d6a2d", linewidth=2)
    axes[1, 0].fill_between(weekly.index, weekly.values, alpha=0.2, color="#2d6a2d")
    axes[1, 0].set_title("Weekly Order Volume")
    axes[1, 0].set_xlabel("Week Number")
    axes[1, 0].set_ylabel("Order Lines")

    # Revenue by producer
    rev = orders_df.groupby("producer")["subtotal"].sum().sort_values()
    axes[1, 1].barh(rev.index, rev.values, color=["#FF9800", "#2196F3", "#4CAF50"])
    axes[1, 1].set_title("Total Revenue by Producer")
    axes[1, 1].set_xlabel("Revenue (£)")
    axes[1, 1].xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}")
    )

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "eda_overview.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  EDA overview saved to {path}")


def plot_heatmap(orders_df):
    # Keep only top 15 customers by total orders for readability
    top_customers = (
        orders_df.groupby("customer")["order_id"]
        .nunique()
        .nlargest(15)
        .index
    )
    filtered = orders_df[orders_df["customer"].isin(top_customers)]
    pivot = filtered.groupby(["customer", "product"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(16, 9))
    sns.heatmap(
        pivot,
        annot=True,
        fmt="d",
        cmap="YlGn",
        linewidths=0.8,
        linecolor="white",
        annot_kws={"size": 11},
        cbar_kws={"label": "Times Ordered", "shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Customer x Product Order Frequency Heatmap (Top 15 Customers)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Product", fontsize=12, labelpad=10)
    ax.set_ylabel("Customer", fontsize=12, labelpad=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Heatmap saved to {path}")


def plot_fairness(features_df):
    prod_map  = {p["name"]: p["producer"] for p in PRODUCTS}
    high_conf = features_df[features_df["will_reorder"] == 1].copy()
    high_conf["producer"] = high_conf["product"].map(prod_map)
    dist = high_conf["producer"].value_counts()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Fairness Analysis — Producer Distribution in Suggestions",
                 fontsize=12, fontweight="bold")

    axes[0].pie(dist.values, labels=dist.index, autopct="%1.1f%%",
                colors=["#4CAF50", "#2196F3", "#FF9800"])
    axes[0].set_title("Share of Suggestions by Producer")

    axes[1].bar(dist.index, dist.values, color=["#4CAF50", "#2196F3", "#FF9800"])
    axes[1].set_title("Suggested Products by Producer")
    axes[1].set_ylabel("Count")
    plt.xticks(rotation=15, ha="right")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fairness.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Fairness chart saved to {path}")



def plot_reorder_suggestions(model, features_df):
    """
    For 6 sample customers, show their top reorder suggestions
    with confidence scores as a horizontal bar chart.
    This is the visual evidence for the re-order options requirement.
    """
    sample_customers = [
        "Alice Jones", "Bob Smith", "Carol White",
        "David Brown", "Emma Taylor", "Grace Moore"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Personalised Re-Order Suggestions — Top Products per Customer",
        fontsize=15, fontweight="bold", y=1.01
    )
    axes = axes.flatten()

    for idx, customer_name in enumerate(sample_customers):
        ax = axes[idx]
        cust_df = features_df[features_df["customer"] == customer_name].copy()

        if cust_df.empty:
            ax.set_visible(False)
            continue

        X_cust = cust_df[FEATURE_COLS].values
        proba  = model.predict_proba(X_cust)[:, 1]
        cust_df = cust_df.copy()
        cust_df["confidence"] = proba
        cust_df = cust_df.sort_values("confidence", ascending=False).head(5)

        # Colour bars: green if predicted to reorder, grey if not
        colours = ["#2d6a2d" if c >= 0.5 else "#b0bec5"
                   for c in cust_df["confidence"]]

        bars = ax.barh(
            cust_df["product"][::-1],
            cust_df["confidence"][::-1] * 100,
            color=colours[::-1],
            edgecolor="white",
            height=0.6,
        )

        # Add percentage labels on bars
        for bar, conf in zip(bars, cust_df["confidence"][::-1]):
            ax.text(
                bar.get_width() + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{conf*100:.0f}%",
                va="center", ha="left", fontsize=9, fontweight="bold"
            )

        ax.set_xlim(0, 115)
        ax.set_xlabel("Reorder Confidence (%)", fontsize=9)
        ax.set_title(customer_name, fontsize=11, fontweight="bold", pad=8)
        ax.axvline(50, color="red", linestyle="--", linewidth=0.8, alpha=0.6,
                   label="50% threshold")
        ax.tick_params(axis="y", labelsize=9)
        ax.set_facecolor("#f9f9f9")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Add a shared legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2d6a2d", label="Suggested (≥50% confidence)"),
        Patch(facecolor="#b0bec5", label="Not suggested (<50%)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "reorder_suggestions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Reorder suggestions chart saved to {path}")


def evaluate():
    import pandas as pd
    from data import HISTORY_PATH, FEATURES_PATH
    from sklearn.model_selection import train_test_split

    model, cat_encoder = load_model()

    print("\nRunning evaluation...\n")

    X, y, features_df, _ = build_features()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    auc       = roc_auc_score(y_test, y_proba)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

    print("=" * 50)
    print(f"  Test Accuracy      : {acc:.2%}")
    print(f"  Precision          : {precision:.2%}")
    print(f"  Recall             : {recall:.2%}")
    print(f"  F1 Score           : {f1:.2%}")
    print(f"  ROC-AUC Score      : {auc:.4f}")
    print(f"  5-Fold CV Accuracy : {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
    print("=" * 50)
    print()
    print(classification_report(y_test, y_pred,
                                target_names=["No Reorder", "Will Reorder"]))

    # Save metrics to text file
    metrics_path = os.path.join(RESULTS_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Test Accuracy      : {acc:.2%}\n")
        f.write(f"Precision          : {precision:.2%}\n")
        f.write(f"Recall             : {recall:.2%}\n")
        f.write(f"F1 Score           : {f1:.2%}\n")
        f.write(f"ROC-AUC Score      : {auc:.4f}\n")
        f.write(f"5-Fold CV Accuracy : {cv_scores.mean():.2%} ± {cv_scores.std():.2%}\n\n")
        f.write(classification_report(y_test, y_pred,
                                      target_names=["No Reorder", "Will Reorder"]))
    print(f"Metrics saved to {metrics_path}\n")

    # Generate all charts
    print("Saving charts...")
    orders_df = pd.read_csv(HISTORY_PATH, parse_dates=["order_date"])
    plot_eda(orders_df)
    plot_heatmap(orders_df)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba, auc)
    plot_cross_validation(cv_scores)
    plot_feature_importance(model)
    plot_fairness(features_df)
    plot_reorder_suggestions(model, features_df)

    print(f"\nAll evaluation results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    evaluate()
