import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Path to dataset
DATASET_PATH = os.path.join(
    os.path.dirname(__file__),
    "dataset",
    "Fruit And Vegetable Diseases Dataset"
)

# Image settings
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 2


def get_transforms(train=True):
    """
    Returns the image transformations.
    Training uses augmentation to improve generalisation.
    Validation/Test use only resizing + normalization.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def load_data():
    """
    Loads dataset and splits it into:
    70% training
    15% validation
    15% testing
    """

    # Load dataset
    full_dataset = datasets.ImageFolder(
        root=DATASET_PATH,
        transform=get_transforms(train=True)
    )

    # Extract class names
    class_names = full_dataset.classes
    num_classes = len(class_names)

    print(f"Found {num_classes} classes")
    print(class_names)
    print(f"Total images: {len(full_dataset)}")

    # Split dataset
    total = len(full_dataset)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_size = total - train_size - val_size

    train_set, val_set, test_set = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Validation/Test transforms without augmentation
    val_set.dataset.transform = get_transforms(train=False)
    test_set.dataset.transform = get_transforms(train=False)

    # Create dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print(f"Train: {train_size} | Val: {val_size} | Test: {test_size}")

    return train_loader, val_loader, test_loader, class_names, num_classes


if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names, num_classes = load_data()
    print("Dataset preprocessing complete.")