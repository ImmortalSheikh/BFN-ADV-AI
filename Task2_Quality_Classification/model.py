import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes, device):
    """
    Uses ResNet50 pretrained on ImageNet as the backbone.
    We replace the final layer to match our 28 classes.
    This is called Transfer Learning - a powerful technique
    that gives us high accuracy without training from scratch.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze early layers - keep pretrained features
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last two layers for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace final fully connected layer for our 28 classes
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    model = model.to(device)
    print(f"Model built! Running on: {device}")
    print(f"Output classes: {num_classes}")
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=28, device=device)
    print(model.fc)