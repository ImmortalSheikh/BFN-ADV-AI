import os
import random
import numpy as np
import pandas as pd
from datetime import date, timedelta
from sklearn.preprocessing import LabelEncoder

# Paths
DATASET_DIR   = os.path.join(os.path.dirname(__file__), "dataset")
HISTORY_PATH  = os.path.join(DATASET_DIR, "purchase_history.csv")
FEATURES_PATH = os.path.join(DATASET_DIR, "features.csv")

# Settings
WEEKS          = 52   # 1 full year
TRAIN_WEEKS    = 40   # features built from first 40 weeks
FUTURE_WEEKS   = 12   # label = did they order in last 12 weeks?
RANDOM_SEED    = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Feature columns used by the model
FEATURE_COLS = [
    "order_count",
    "days_since_last",
    "avg_quantity",
    "order_frequency",
    "week_of_year",
    "category_encoded",
]

# Product catalogue
PRODUCTS = [
    {"id": 1,  "name": "Carrots",        "category": "Vegetables", "price": 1.50, "unit": "kg",     "producer": "Green Valley Farm"},
    {"id": 2,  "name": "Potatoes",        "category": "Vegetables", "price": 2.00, "unit": "kg",     "producer": "Green Valley Farm"},
    {"id": 3,  "name": "Spinach",         "category": "Vegetables", "price": 1.80, "unit": "bag",    "producer": "Green Valley Farm"},
    {"id": 4,  "name": "Courgettes",      "category": "Vegetables", "price": 2.20, "unit": "kg",     "producer": "Green Valley Farm"},
    {"id": 5,  "name": "Apples",          "category": "Fruit",      "price": 2.50, "unit": "kg",     "producer": "Green Valley Farm"},
    {"id": 6,  "name": "Strawberries",    "category": "Fruit",      "price": 3.50, "unit": "punnet", "producer": "Green Valley Farm"},
    {"id": 7,  "name": "Free Range Eggs", "category": "Dairy",      "price": 3.00, "unit": "dozen",  "producer": "Bristol Dairy Co"},
    {"id": 8,  "name": "Whole Milk",      "category": "Dairy",      "price": 1.20, "unit": "litre",  "producer": "Bristol Dairy Co"},
    {"id": 9,  "name": "Cheddar Cheese",  "category": "Dairy",      "price": 4.50, "unit": "kg",     "producer": "Bristol Dairy Co"},
    {"id": 10, "name": "Natural Yoghurt", "category": "Dairy",      "price": 2.00, "unit": "jar",    "producer": "Bristol Dairy Co"},
    {"id": 11, "name": "Sourdough Loaf",  "category": "Bakery",     "price": 4.50, "unit": "loaf",   "producer": "Artisan Bakery Bristol"},
    {"id": 12, "name": "Wholemeal Bread", "category": "Bakery",     "price": 3.00, "unit": "loaf",   "producer": "Artisan Bakery Bristol"},
    {"id": 13, "name": "Croissants",      "category": "Bakery",     "price": 3.50, "unit": "pack",   "producer": "Artisan Bakery Bristol"},
]

# 30 customers with distinct buying preferences
CUSTOMERS = [
    {"id": 1,  "name": "Alice Jones",    "preferred": ["Carrots", "Potatoes", "Free Range Eggs", "Sourdough Loaf"]},
    {"id": 2,  "name": "Bob Smith",      "preferred": ["Apples", "Whole Milk", "Cheddar Cheese", "Carrots"]},
    {"id": 3,  "name": "Carol White",    "preferred": ["Spinach", "Natural Yoghurt", "Wholemeal Bread", "Strawberries"]},
    {"id": 4,  "name": "David Brown",    "preferred": ["Potatoes", "Free Range Eggs", "Cheddar Cheese", "Croissants"]},
    {"id": 5,  "name": "Emma Taylor",    "preferred": ["Carrots", "Apples", "Whole Milk", "Sourdough Loaf"]},
    {"id": 6,  "name": "Frank Wilson",   "preferred": ["Courgettes", "Free Range Eggs", "Wholemeal Bread", "Potatoes"]},
    {"id": 7,  "name": "Grace Moore",    "preferred": ["Strawberries", "Natural Yoghurt", "Croissants", "Spinach"]},
    {"id": 8,  "name": "Henry Davis",    "preferred": ["Carrots", "Potatoes", "Whole Milk", "Cheddar Cheese"]},
    {"id": 9,  "name": "Isla Martin",    "preferred": ["Sourdough Loaf", "Spinach", "Free Range Eggs", "Apples"]},
    {"id": 10, "name": "Jack Thompson",  "preferred": ["Whole Milk", "Croissants", "Potatoes", "Carrots"]},
    {"id": 11, "name": "Karen Harris",   "preferred": ["Natural Yoghurt", "Strawberries", "Wholemeal Bread", "Apples"]},
    {"id": 12, "name": "Liam Roberts",   "preferred": ["Cheddar Cheese", "Free Range Eggs", "Sourdough Loaf", "Courgettes"]},
    {"id": 13, "name": "Mia Lewis",      "preferred": ["Spinach", "Carrots", "Whole Milk", "Croissants"]},
    {"id": 14, "name": "Noah Walker",    "preferred": ["Potatoes", "Apples", "Cheddar Cheese", "Wholemeal Bread"]},
    {"id": 15, "name": "Olivia Hall",    "preferred": ["Strawberries", "Natural Yoghurt", "Free Range Eggs", "Sourdough Loaf"]},
    {"id": 16, "name": "Peter Allen",    "preferred": ["Courgettes", "Carrots", "Whole Milk", "Wholemeal Bread"]},
    {"id": 17, "name": "Quinn Young",    "preferred": ["Apples", "Spinach", "Croissants", "Cheddar Cheese"]},
    {"id": 18, "name": "Rachel King",    "preferred": ["Potatoes", "Free Range Eggs", "Natural Yoghurt", "Carrots"]},
    {"id": 19, "name": "Sam Wright",     "preferred": ["Sourdough Loaf", "Whole Milk", "Strawberries", "Courgettes"]},
    {"id": 20, "name": "Tara Scott",     "preferred": ["Spinach", "Cheddar Cheese", "Apples", "Wholemeal Bread"]},
    {"id": 21, "name": "Uma Green",      "preferred": ["Carrots", "Natural Yoghurt", "Croissants", "Free Range Eggs"]},
    {"id": 22, "name": "Victor Adams",   "preferred": ["Potatoes", "Whole Milk", "Sourdough Loaf", "Spinach"]},
    {"id": 23, "name": "Wendy Baker",    "preferred": ["Strawberries", "Apples", "Cheddar Cheese", "Carrots"]},
    {"id": 24, "name": "Xavier Clark",   "preferred": ["Free Range Eggs", "Wholemeal Bread", "Courgettes", "Whole Milk"]},
    {"id": 25, "name": "Yasmin Evans",   "preferred": ["Natural Yoghurt", "Sourdough Loaf", "Potatoes", "Apples"]},
    {"id": 26, "name": "Zoe Foster",     "preferred": ["Spinach", "Croissants", "Carrots", "Cheddar Cheese"]},
    {"id": 27, "name": "Aaron Hughes",   "preferred": ["Whole Milk", "Free Range Eggs", "Wholemeal Bread", "Strawberries"]},
    {"id": 28, "name": "Bella Price",    "preferred": ["Apples", "Courgettes", "Natural Yoghurt", "Sourdough Loaf"]},
    {"id": 29, "name": "Callum Reed",    "preferred": ["Potatoes", "Cheddar Cheese", "Carrots", "Croissants"]},
    {"id": 30, "name": "Diana Shaw",     "preferred": ["Spinach", "Strawberries", "Whole Milk", "Free Range Eggs"]},
]


def generate_purchase_history():
    """
    Generate 52 weeks of synthetic purchase history for 30 customers.

    Realistic behaviour modelled:
    - Each customer has a 75% chance of placing an order each week
    - Customers select 2-4 items from their preferred products
    - 20% chance of exploring a random product outside preferences
    - Some customers randomly stop buying certain products mid-year
      (simulates seasonal or lifestyle changes) — this is what
      prevents perfect prediction and gives realistic accuracy
    """
    os.makedirs(DATASET_DIR, exist_ok=True)

    today = date.today()
    product_name_to_id = {p["name"]: p["id"] for p in PRODUCTS}
    order_rows = []
    order_id   = 1

    # Randomly drop 1-2 preferred products per customer after week 20
    # This simulates changing habits and prevents leakage
    customer_drops = {}
    for customer in CUSTOMERS:
        drop_after = random.randint(18, 35)
        dropped    = random.sample(customer["preferred"], k=random.randint(1, 2))
        customer_drops[customer["id"]] = (drop_after, dropped)

    for week in range(WEEKS):
        order_date = today - timedelta(weeks=(WEEKS - week))
        week_num   = week + 1

        for customer in CUSTOMERS:
            if random.random() < 0.75:
                drop_after, dropped = customer_drops[customer["id"]]

                # After drop_after, remove dropped products from preferred
                if week_num > drop_after:
                    available = [p for p in customer["preferred"] if p not in dropped]
                else:
                    available = customer["preferred"]

                if not available:
                    available = customer["preferred"]

                num_items = random.randint(2, 4)
                chosen    = random.sample(available, min(num_items, len(available)))

                # 20% chance of exploring a random product
                if random.random() < 0.20:
                    extra = random.choice([p["name"] for p in PRODUCTS])
                    if extra not in chosen:
                        chosen.append(extra)

                for product_name in chosen:
                    qty   = random.randint(1, 5)
                    pid   = product_name_to_id[product_name]
                    price = next(p["price"] for p in PRODUCTS if p["id"] == pid)
                    order_rows.append({
                        "order_id":    order_id,
                        "customer_id": customer["id"],
                        "customer":    customer["name"],
                        "product_id":  pid,
                        "product":     product_name,
                        "category":    next(p["category"] for p in PRODUCTS if p["id"] == pid),
                        "producer":    next(p["producer"] for p in PRODUCTS if p["id"] == pid),
                        "quantity":    qty,
                        "unit_price":  price,
                        "subtotal":    round(qty * price, 2),
                        "order_date":  order_date,
                        "week_number": order_date.isocalendar()[1],
                        "sim_week":    week_num,
                    })
                order_id += 1

    df = pd.DataFrame(order_rows)
    df["order_date"] = pd.to_datetime(df["order_date"])
    df.to_csv(HISTORY_PATH, index=False)

    print(f"Purchase history saved to {HISTORY_PATH}")
    print(f"  Total order lines : {len(df):,}")
    print(f"  Unique orders     : {df['order_id'].nunique():,}")
    print(f"  Customers         : {df['customer'].nunique()}")
    print(f"  Products          : {df['product'].nunique()}")
    print(f"  Date range        : {df['order_date'].min().date()} to {df['order_date'].max().date()}")
    return df


def build_features(orders_df=None):
    """
    Engineer features using a time-based train/future split to prevent data leakage.

    - Features are built from weeks 1-40 (training window)
    - Label = did the customer actually order this product in weeks 41-52?

    This is the correct ML approach: the model learns from past behaviour
    and predicts future behaviour. Because customers randomly change their
    habits mid-year, the label cannot be perfectly derived from the features,
    which produces realistic accuracy (80-90%) rather than a suspicious 100%.

    All 390 (customer × product) pairs are included:
    - Pairs where the customer ordered in weeks 1-40: features filled in
    - Pairs where the customer never ordered: order_count=0, days_since=max

    Features:
        order_count      - times ordered in weeks 1-40
        days_since_last  - days since last order in training window
        avg_quantity     - average quantity per order
        order_frequency  - proportion of training orders containing this product
        week_of_year     - current week (seasonal signal)
        category_encoded - product category (label encoded)

    Label: will_reorder = 1 if ordered this product in weeks 41-52, else 0
    """
    if orders_df is None:
        if not os.path.exists(HISTORY_PATH):
            print("No purchase history found. Generating now...")
            orders_df = generate_purchase_history()
        else:
            orders_df = pd.read_csv(HISTORY_PATH, parse_dates=["order_date"])

    # Time-based split
    train_df  = orders_df[orders_df["sim_week"] <= TRAIN_WEEKS]
    future_df = orders_df[orders_df["sim_week"] > TRAIN_WEEKS]

    cat_encoder = LabelEncoder()
    cat_encoder.fit([p["category"] for p in PRODUCTS])

    train_cutoff = train_df["order_date"].max()
    feature_rows = []

    for customer in CUSTOMERS:
        cust_train   = train_df[train_df["customer_id"] == customer["id"]]
        cust_future  = future_df[future_df["customer_id"] == customer["id"]]
        total_orders = cust_train["order_id"].nunique()

        for product in PRODUCTS:
            prod_train  = cust_train[cust_train["product_id"] == product["id"]]
            prod_future = cust_future[cust_future["product_id"] == product["id"]]

            count = len(prod_train)

            if count > 0:
                last_date  = prod_train["order_date"].max()
                days_since = (train_cutoff - last_date).days
                avg_qty    = prod_train["quantity"].mean()
            else:
                days_since = TRAIN_WEEKS * 7   # max possible
                avg_qty    = 0.0

            frequency = count / max(total_orders, 1)
            week_num  = date.today().isocalendar()[1]
            cat_enc   = cat_encoder.transform([product["category"]])[0]

            # Label: did they actually order this in the future window?
            label = 1 if len(prod_future) > 0 else 0

            feature_rows.append({
                "customer_id":      customer["id"],
                "customer":         customer["name"],
                "product_id":       product["id"],
                "product":          product["name"],
                "category":         product["category"],
                "producer":         product["producer"],
                "order_count":      count,
                "days_since_last":  days_since,
                "avg_quantity":     round(float(avg_qty), 2),
                "order_frequency":  round(float(frequency), 4),
                "week_of_year":     week_num,
                "category_encoded": int(cat_enc),
                "will_reorder":     label,
            })

    features_df = pd.DataFrame(feature_rows)
    features_df.to_csv(FEATURES_PATH, index=False)

    X = features_df[FEATURE_COLS].values
    y = features_df["will_reorder"].values

    print(f"Feature matrix saved to {FEATURES_PATH}")
    print(f"  Total samples    : {len(features_df):,}")
    print(f"  Will reorder (1) : {y.sum():,} ({y.mean()*100:.1f}%)")
    print(f"  No reorder   (0) : {(y == 0).sum():,} ({(1-y.mean())*100:.1f}%)")

    return X, y, features_df, cat_encoder


if __name__ == "__main__":
    generate_purchase_history()
    build_features()
    print("\nData preparation complete.")
