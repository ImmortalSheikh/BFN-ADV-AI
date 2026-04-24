"""
train.py
Training pipeline for the ResNet50 fruit and vegetable quality classifier.
Trains the model on the preprocessed dataset, tracks validation accuracy,
and saves the best performing model checkpoint to saved_model.pth.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from preprocess import load_data
from model import build_model

# Settings
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
SAVE_PATH = os.path.join(os.path.dirname(__file__), "saved_model.pth")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Run a single training epoch over the dataset.

    Args:
        model (torch.nn.Module): The model being trained.
        loader (DataLoader): Training data loader.
        criterion: Loss function.
        optimizer: Optimiser instance.
        device (torch.device): Device to run training on.

    Returns:
        tuple: (epoch_loss, epoch_accuracy) as floats.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc="Training", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """
    Evaluate the model on the validation set.

    Args:
        model (torch.nn.Module): The model to evaluate.
        loader (DataLoader): Validation data loader.
        criterion: Loss function.
        device (torch.device): Device to run evaluation on.

    Returns:
        tuple: (epoch_loss, epoch_accuracy) as floats.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        loop = tqdm(loader, desc="Validating", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def train():
    """
    Main training loop. Loads data, builds model, trains for NUM_EPOCHS,
    and saves the best model checkpoint based on validation accuracy.
    Uses ReduceLROnPlateau scheduler to reduce learning rate on plateaus.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader, test_loader, class_names, num_classes = load_data()

    # Build model
    model = build_model(num_classes=num_classes, device=device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    # Reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    best_val_acc = 0.0
    print("\nStarting Training...\n")

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "class_names": class_names,
                    "num_classes": num_classes,
                },
                SAVE_PATH,
            )
            print(f"  Best model saved! Val Acc: {val_acc:.2f}%")
        print()

    print(f"\nTraining Complete! Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {SAVE_PATH}")


if __name__ == "__main__":
    train()