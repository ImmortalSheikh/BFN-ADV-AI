import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes, device):

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Freeze pretrained layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze final ResNet block
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Replace classifier
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