import os
import pickle
import pandas as pd
import numpy as np
from data import FEATURES_PATH, HISTORY_PATH, FEATURE_COLS, PRODUCTS

SAVE_PATH = os.path.join(os.path.dirname(__file__), "model", "reorder_model.pkl")


def load_model():
    if not os.path.exists(SAVE_PATH):
        raise FileNotFoundError(
            f"No model found at {SAVE_PATH}. Run train.py first."
        )
    with open(SAVE_PATH, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["cat_encoder"]


def get_suggestions(customer_name, top_n=5):
    """
    Show reorder suggestions for a given customer with plain-English explanations.
    """
    model, _ = load_model()

    if not os.path.exists(FEATURES_PATH):
        print("No feature data found. Run data.py first.")
        return

    features_df   = pd.read_csv(FEATURES_PATH)
    cust_features = features_df[features_df["customer"] == customer_name].copy()

    if cust_features.empty:
        print(f"No purchase history found for '{customer_name}'.")
        print("Available customers:")
        for name in sorted(features_df["customer"].unique()):
            print(f"  - {name}")
        return

    X_cust = cust_features[FEATURE_COLS].values
    proba  = model.predict_proba(X_cust)[:, 1]
    cust_features["reorder_probability"] = proba
    cust_features = cust_features.sort_values("reorder_probability", ascending=False)

    print(f"\nReorder Suggestions for {customer_name}")
    print("=" * 60)

    shown = 0
    for _, row in cust_features.iterrows():
        if row["reorder_probability"] < 0.5 or shown >= top_n:
            break

        freq = round(1 / row["order_frequency"], 1) if row["order_frequency"] > 0 else "?"

        print(f"\n  Product    : {row['product']} ({row['category']})")
        print(f"  Producer   : {row['producer']}")
        print(f"  Confidence : {row['reorder_probability']*100:.0f}%")
        print(f"  Why        : Ordered {int(row['order_count'])} times. "
              f"Last ordered {int(row['days_since_last'])} days ago. "
              f"Appears in approx 1 in every {freq} of their orders.")
        shown += 1

    if shown == 0:
        print("  No strong suggestions at this time.")
    print()


def demand_forecast(top_n=10):
    """
    Show next-week demand forecast for the top products.
    """
    if not os.path.exists(HISTORY_PATH):
        print("No purchase history found. Run data.py first.")
        return

    history = pd.read_csv(HISTORY_PATH, parse_dates=["order_date"])

    print("\nDemand Forecast — Top Products")
    print("=" * 60)

    top_products = history["product"].value_counts().head(top_n).index

    for product_name in top_products:
        prod        = history[history["product"] == product_name]
        weekly      = prod.groupby("week_number")["quantity"].sum()
        sorted_weeks = sorted(weekly.index)

        recent_weeks = sorted_weeks[-4:]
        older_weeks  = sorted_weeks[-8:-4]

        recent_avg = np.mean([weekly[w] for w in recent_weeks]) if recent_weeks else 0
        older_avg  = np.mean([weekly[w] for w in older_weeks])  if older_weeks  else 0

        trend_pct = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
        forecast  = round(recent_avg * 1.05, 1)
        producer  = next((p["producer"] for p in PRODUCTS if p["name"] == product_name), "Unknown")

        if trend_pct > 15:
            trend = "Rising"
        elif trend_pct < -15:
            trend = "Falling"
        else:
            trend = "Stable"

        print(f"\n  {product_name:20s} | Producer : {producer}")
        print(f"  Forecast next week : {forecast} units | Trend : {trend} ({trend_pct:+.1f}%)")
    print()


if __name__ == "__main__":
    print("Demo: Reorder Suggestions")
    print("-" * 60)
    for customer in ["Alice Jones", "Bob Smith", "Grace Moore"]:
        get_suggestions(customer, top_n=3)

    print("Demo: Demand Forecast for Producers")
    print("-" * 60)
    demand_forecast(top_n=8)
