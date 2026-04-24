"""
inventory.py
Automated inventory management system for the Bristol Regional Food Network.
Runs batch inspections on produce images, grades each item using the
quality classification model, and updates the inventory JSON file.
"""

import json
import os
from datetime import datetime
from grading import predict_image, load_trained_model, print_result
import torch

INVENTORY_FILE = "inventory.json"


def load_inventory():
    """
    Load the current inventory from the JSON file.

    Returns:
        dict: Inventory data keyed by produce name,
              or empty dict if the file does not exist.
    """
    if os.path.exists(INVENTORY_FILE):
        with open(INVENTORY_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_inventory(inventory):
    """
    Save the inventory dictionary to the JSON file.

    Args:
        inventory (dict): Inventory data to persist.
    """
    with open(INVENTORY_FILE, 'w') as f:
        json.dump(inventory, f, indent=4)


def update_inventory(result):
    """
    Update inventory counts based on a single grading result.
    Grade A items are stocked at full price, Grade B with discount,
    and Grade C items are flagged for removal or heavy discount.

    Args:
        result (dict): Grading result from predict_image containing
                       produce, condition, grade and other fields.

    Returns:
        dict: Updated inventory dictionary.
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
    """
    Print a formatted summary table of the current inventory,
    showing Grade A, B, C counts and total inspected per produce type.
    """
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
    Run grading on all images in a folder and update inventory automatically.
    Simulates a real-world batch inspection of incoming produce.

    Args:
        image_dir (str): Path to folder containing produce images.
        model (torch.nn.Module): Trained quality classification model.
        class_names (list): List of class name strings.
        device (torch.device): Device to run inference on.
        limit (int): Maximum number of images to inspect. Defaults to 10.
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

    # Run batch inspection on sample folders
    sample_dirs = [
        os.path.join("dataset", "Fruit And Vegetable Diseases Dataset", "Apple__Healthy"),
        os.path.join("dataset", "Fruit And Vegetable Diseases Dataset", "Apple__Rotten"),
        os.path.join("dataset", "Fruit And Vegetable Diseases Dataset", "Banana__Healthy"),
        os.path.join("dataset", "Fruit And Vegetable Diseases Dataset", "Tomato__Rotten"),
    ]

    for dir_path in sample_dirs:
        run_batch_inspection(dir_path, model, class_names, device, limit=3)