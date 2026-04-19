import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from model import build_model
from gradcam import GradCAM

# -----------------------------
# Settings
# -----------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "saved_model.pth")
IMAGE_PATH = os.path.join(BASE_DIR, "rotten-apple.png")

# Folder to save Grad-CAM outputs
OUTPUT_DIR = os.path.join(BASE_DIR, "xai_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Image transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Load checkpoint
# -----------------------------
checkpoint = torch.load(MODEL_PATH, map_location=device)

num_classes = checkpoint["num_classes"]
class_names = checkpoint["class_names"]

model = build_model(num_classes=num_classes, device=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Last convolutional layer in ResNet50
target_layer = model.layer4[-1].conv3
gradcam = GradCAM(model, target_layer)

# -----------------------------
# Load image
# -----------------------------
image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# -----------------------------
# Predict class
# -----------------------------
with torch.no_grad():
    output = model(input_tensor)
    pred_class = torch.argmax(output, dim=1).item()
    pred_label = class_names[pred_class]

# -----------------------------
# Generate Grad-CAM heatmap
# -----------------------------
heatmap = gradcam.generate(input_tensor, class_idx=pred_class)

# -----------------------------
# Read original image for display/export
# -----------------------------
img = cv2.imread(IMAGE_PATH)
img = cv2.resize(img, (224, 224))

heatmap_uint8 = np.uint8(255 * heatmap)
heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
overlay = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

# -----------------------------
# Build output filenames
# -----------------------------
image_name = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
safe_label = pred_label.replace("__", "_")

original_out = os.path.join(OUTPUT_DIR, f"{image_name}_original.jpg")
heatmap_out = os.path.join(OUTPUT_DIR, f"{image_name}_heatmap.jpg")
overlay_out = os.path.join(OUTPUT_DIR, f"{image_name}_overlay_{safe_label}.jpg")
figure_out = os.path.join(OUTPUT_DIR, f"{image_name}_figure_{safe_label}.png")

# -----------------------------
# Save output images
# -----------------------------
cv2.imwrite(original_out, img)
cv2.imwrite(heatmap_out, heatmap_colored)
cv2.imwrite(overlay_out, overlay)

# -----------------------------
# Show and save combined figure
# -----------------------------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(heatmap, cmap="jet")
plt.title("Grad-CAM Heatmap")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title(f"Prediction: {pred_label}")
plt.axis("off")

plt.tight_layout()
plt.savefig(figure_out, dpi=300, bbox_inches="tight")

# -----------------------------
# XAI — Decision Explanation
# -----------------------------
from grading import simulate_quality_scores, assign_grade, get_inventory_action
import torch.nn.functional as F

parts = pred_label.split("__")
produce = parts[0]
is_healthy = parts[1].lower() == "healthy" if len(parts) > 1 else False

with torch.no_grad():
    output = model(input_tensor)
    probs = F.softmax(output, dim=1)
    confidence = probs.max().item()

color, size, ripeness = simulate_quality_scores(confidence, is_healthy)
grade = assign_grade(color, size, ripeness)
action = get_inventory_action(grade)

print("\n" + "=" * 50)
print("  XAI — Decision Explanation")
print("=" * 50)
print(f"  Produce       : {produce}")
print(f"  Condition     : {'Healthy' if is_healthy else 'Rotten'}")
print(f"  Confidence    : {round(confidence * 100, 2)}%")
print(f"  Color Score   : {color}%")
print(f"  Size Score    : {size}%")
print(f"  Ripeness      : {ripeness}%")
print(f"  Grade         : {grade}")
print(f"  Recommendation: {action}")
print("=" * 50)
print("  Grad-CAM highlights the regions that most")
print("  influenced this prediction (red = high importance)")
print("=" * 50)

plt.show()

print(f"\nPredicted class index: {pred_class}")
print(f"Predicted label: {pred_label}")
print("Saved files:")
print(f" - {original_out}")
print(f" - {heatmap_out}")
print(f" - {overlay_out}")
print(f" - {figure_out}")