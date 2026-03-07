import json
import os
from datetime import datetime
from grading import predict_image, load_trained_model, print_result
import torch

INVENTORY_FILE = "inventory.json"


def load_inventory():
    if os.path.exists(INVENTORY_FILE):
        with open(INVENTORY_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_inventory(inventory):
    with open(INVENTORY_FILE, 'w') as f:
        json.dump(inventory, f, indent=4)


def update_inventory(result):
    """
    Automatically updates inventory based on grading result.
    - Grade A: added to stock at full price
    - Grade B: added with discount applied
    - Grade C: flagged for removal or heavy discount
    """
    inventory = load_inventory()
    produce = result['produce']

    if produce not in inventory:
        inventory[produce] = {
            "grade_A": 0,
            "grade_B": 0,
            "grade_C": 0,
            "total_inspected": 0,
            "last_updated": None
        }

    grade = result['grade']
    inventory[produce][f"grade_{grade}"] += 1
    inventory[produce]["total_inspected"] += 1
    inventory[produce]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    save_inventory(inventory)
    return inventory


def print_inventory_summary():
    inventory = load_inventory()
    if not inventory:
        print("Inventory is empty.")
        return

    print("\n" + "="*60)
    print("  INVENTORY SUMMARY")
    print("="*60)
    print(f"  {'Produce':<20} {'Grade A':>8} {'Grade B':>8} {'Grade C':>8} {'Total':>8}")
    print("-"*60)
    for produce, data in inventory.items():
        print(f"  {produce:<20} {data['grade_A']:>8} {data['grade_B']:>8} {data['grade_C']:>8} {data['total_inspected']:>8}")
        print(f"  {'Last updated:':<20} {data['last_updated']}")
        print()
    print("="*60)


def run_batch_inspection(image_dir, model, class_names, device, limit=10):
    """
    Runs grading on a folder of images and updates inventory automatically.
    Simulates real-world batch inspection of produce.
    """
    print(f"\n🔍 Starting batch inspection of: {image_dir}\n")
    images = [f for f in os.listdir(image_dir)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:limit]

    if not images:
        print("No images found in directory.")
        return

    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        print(f"Inspecting: {img_name}")
        result = predict_image(img_path, model, class_names, device)
        print_result(result)
        update_inventory(result)

    print("\n✅ Batch inspection complete. Inventory updated.")
    print_inventory_summary()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_trained_model(device)

    # Run batch inspection on a sample folder
    sample_dirs = [
        os.path.join("dataset", "Fruit And Vegetable Diseases Dataset", "Apple__Healthy"),
        os.path.join("dataset", "Fruit And Vegetable Diseases Dataset", "Apple__Rotten"),
        os.path.join("dataset", "Fruit And Vegetable Diseases Dataset", "Banana__Healthy"),
        os.path.join("dataset", "Fruit And Vegetable Diseases Dataset", "Tomato__Rotten"),
    ]

    for dir_path in sample_dirs:
        run_batch_inspection(dir_path, model, class_names, device, limit=3)