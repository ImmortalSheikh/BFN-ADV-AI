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
plt.show()

print(f"Predicted class index: {pred_class}")
print(f"Predicted label: {pred_label}")
print("Saved files:")
print(f" - {original_out}")
print(f" - {heatmap_out}")
print(f" - {overlay_out}")
print(f" - {figure_out}")