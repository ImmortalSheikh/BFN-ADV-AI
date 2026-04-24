"""
evaluate.py
Evaluation pipeline for the ResNet50 quality classification model.
Computes accuracy, precision, recall, F1 score and generates
confusion matrix, per-class accuracy and grade distribution charts.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score,
                             recall_score, f1_score)
from tqdm import tqdm
from preprocess import load_data
from model import build_model

SAVE_PATH = os.path.join(os.path.dirname(__file__), "model_registry", "saved_model.pth")
RESULTS_DIR = "evaluation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_trained_model(device):
    """
    Load the trained ResNet50 model from the model registry.

    Args:
        device (torch.device): Device to load the model onto.

    Returns:
        tuple: (model, class_names) where model is the loaded ResNet50
               and class_names is the list of 28 class label strings.
    """
    checkpoint = torch.load(SAVE_PATH, map_location=device)
    class_names = checkpoint['class_names']
    num_classes = checkpoint['num_classes']
    model = build_model(num_classes=num_classes, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, class_names


def get_predictions(model, loader, device):
    """
    Run inference on the test set and collect all predictions.

    Args:
        model (torch.nn.Module): Trained model.
        loader (DataLoader): Test data loader.
        device (torch.device): Device to run inference on.

    Returns:
        tuple: (labels, predictions) as numpy arrays of integer class indices.
    """
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(labels, preds, class_names):
    """
    Generate and save full 28-class and binary (Healthy/Rotten) confusion matrices.

    Args:
        labels (np.array): True class indices.
        preds (np.array): Predicted class indices.
        class_names (list): List of class name strings.
    """
    cm = confusion_matrix(labels, preds)

    # Full confusion matrix
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix — All 28 Classes', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print("✅ Confusion matrix saved!")

    # Binary healthy vs rotten confusion matrix
    binary_labels = [1 if class_names[l].endswith("Healthy") else 0 for l in labels]
    binary_preds = [1 if class_names[p].endswith("Healthy") else 0 for p in preds]
    binary_cm = confusion_matrix(binary_labels, binary_preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(binary_cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Rotten', 'Healthy'],
                yticklabels=['Rotten', 'Healthy'])
    plt.title('Binary Confusion Matrix — Healthy vs Rotten', fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'binary_confusion_matrix.png'), dpi=150)
    plt.close()
    print("✅ Binary confusion matrix saved!")


def plot_per_class_accuracy(labels, preds, class_names):
    """
    Generate and save a per-class accuracy bar chart.

    Args:
        labels (np.array): True class indices.
        preds (np.array): Predicted class indices.
        class_names (list): List of class name strings.
    """
    cm = confusion_matrix(labels, preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100

    plt.figure(figsize=(16, 6))
    bars = plt.bar(class_names, per_class_acc, color='steelblue')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy')
    plt.ylim(0, 105)
    for bar, acc in zip(bars, per_class_acc):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'per_class_accuracy.png'), dpi=150)
    plt.close()
    print("✅ Per-class accuracy chart saved!")


def plot_grade_distribution(labels, preds, class_names):
    """
    Generate and save a grade distribution bar chart showing how many
    test set items fall into Grade A, B, and C categories.

    Args:
        labels (np.array): True class indices (unused, kept for consistency).
        preds (np.array): Predicted class indices.
        class_names (list): List of class name strings.
    """
    from grading import simulate_quality_scores, assign_grade

    grades = {"A": 0, "B": 0, "C": 0}
    for pred in preds:
        class_name = class_names[pred]
        is_healthy = class_name.endswith("Healthy")
        color, size, ripeness = simulate_quality_scores(0.95 if is_healthy else 0.4, is_healthy)
        grade = assign_grade(color, size, ripeness)
        grades[grade] += 1

    plt.figure(figsize=(7, 5))
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    bars = plt.bar(grades.keys(), grades.values(), color=colors)
    plt.title('Grade Distribution Across Test Set', fontsize=14)
    plt.xlabel('Grade')
    plt.ylabel('Number of Items')
    for bar, val in zip(bars, grades.values()):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 10,
                 str(val), ha='center', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'grade_distribution.png'), dpi=150)
    plt.close()
    print("✅ Grade distribution chart saved!")


def evaluate():
    """
    Run the full evaluation pipeline on the test set.
    Computes accuracy, precision, recall and F1 score, saves metrics
    to a text file, and generates all evaluation charts.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, test_loader, class_names, num_classes = load_data()
    model, class_names = load_trained_model(device)

    print("\n🔍 Running evaluation on test set...\n")
    labels, preds = get_predictions(model, test_loader, device)

    # Core metrics
    acc = accuracy_score(labels, preds) * 100
    precision = precision_score(labels, preds, average='weighted', zero_division=0) * 100
    recall = recall_score(labels, preds, average='weighted', zero_division=0) * 100
    f1 = f1_score(labels, preds, average='weighted', zero_division=0) * 100

    print("=" * 50)
    print(f"  Test Accuracy  : {acc:.2f}%")
    print(f"  Precision      : {precision:.2f}%")
    print(f"  Recall         : {recall:.2f}%")
    print(f"  F1 Score       : {f1:.2f}%")
    print("=" * 50)

    # Save metrics to text file
    with open(os.path.join(RESULTS_DIR, 'metrics.txt'), 'w') as f:
        f.write(f"Test Accuracy  : {acc:.2f}%\n")
        f.write(f"Precision      : {precision:.2f}%\n")
        f.write(f"Recall         : {recall:.2f}%\n")
        f.write(f"F1 Score       : {f1:.2f}%\n\n")
        f.write(classification_report(labels, preds,
                                      target_names=class_names,
                                      zero_division=0))
    print("✅ Metrics saved to evaluation_results/metrics.txt")

    # Generate all charts
    plot_confusion_matrix(labels, preds, class_names)
    plot_per_class_accuracy(labels, preds, class_names)
    plot_grade_distribution(labels, preds, class_names)

    print("\n✅ All evaluation results saved to evaluation_results/")


if __name__ == "__main__":
    evaluate()