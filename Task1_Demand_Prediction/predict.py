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
    Return reorder suggestions for a given customer as structured data.
    """
    model, _ = load_model()

    if not os.path.exists(FEATURES_PATH):
        return {"error": "No feature data found. Run data.py first."}

    features_df = pd.read_csv(FEATURES_PATH)
    cust_features = features_df[features_df["customer"] == customer_name].copy()

    if cust_features.empty:
        return {
            "error": f"No purchase history found for '{customer_name}'.",
            "available_customers": sorted(features_df["customer"].unique().tolist())
        }

    X_cust = cust_features[FEATURE_COLS].values
    proba = model.predict_proba(X_cust)[:, 1]
    cust_features["reorder_probability"] = proba
    cust_features = cust_features.sort_values("reorder_probability", ascending=False)

    results = []

    for _, row in cust_features.iterrows():
        if row["reorder_probability"] < 0.5 or len(results) >= top_n:
            break

        freq = round(1 / row["order_frequency"], 1) if row["order_frequency"] > 0 else "?"

        results.append({
            "product": row["product"],
            "category": row["category"],
            "producer": row["producer"],
            "confidence": round(row["reorder_probability"] * 100, 2),
            "order_count": int(row["order_count"]),
            "days_since_last": int(row["days_since_last"]),
            "order_frequency_inverse": freq,
            "explanation": (
                f"Ordered {int(row['order_count'])} times. "
                f"Last ordered {int(row['days_since_last'])} days ago. "
                f"Appears in approx 1 in every {freq} of their orders."
            )
        })

    if not results:
        return {"message": "No strong suggestions at this time."}

    return results


def demand_forecast(top_n=10):
    """
    Return next-week demand forecast for top products as structured data.
    """
    if not os.path.exists(HISTORY_PATH):
        return {"error": "No purchase history found. Run data.py first."}

    history = pd.read_csv(HISTORY_PATH, parse_dates=["order_date"])
    top_products = history["product"].value_counts().head(top_n).index

    results = []

    for product_name in top_products:
        prod = history[history["product"] == product_name]
        weekly = prod.groupby("week_number")["quantity"].sum()
        sorted_weeks = sorted(weekly.index)

        recent_weeks = sorted_weeks[-4:]
        older_weeks = sorted_weeks[-8:-4]

        recent_avg = np.mean([weekly[w] for w in recent_weeks]) if recent_weeks else 0
        older_avg = np.mean([weekly[w] for w in older_weeks]) if older_weeks else 0

        trend_pct = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
        forecast = round(recent_avg * 1.05, 1)
        producer = next((p["producer"] for p in PRODUCTS if p["name"] == product_name), "Unknown")

        if trend_pct > 15:
            trend = "Rising"
        elif trend_pct < -15:
            trend = "Falling"
        else:
            trend = "Stable"

        results.append({
            "product": product_name,
            "producer": producer,
            "forecast_next_week": forecast,
            "trend": trend,
            "trend_percent": round(trend_pct, 1)
        })

    return results


if __name__ == "__main__":
    print("Demo: Reorder Suggestions")
    print("-" * 60)
    for customer in ["Alice Jones", "Bob Smith", "Grace Moore"]:
        suggestions = get_suggestions(customer, top_n=3)
        print(f"\nCustomer: {customer}")
        print(suggestions)

    print("\nDemo: Demand Forecast for Producers")
    print("-" * 60)
    forecast_results = demand_forecast(top_n=8)
    print(forecast_results)