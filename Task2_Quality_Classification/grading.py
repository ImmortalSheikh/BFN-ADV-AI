import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from model import build_model

# Paths
MODEL_PATH = "saved_model.pth"
IMG_SIZE = 224

# Image transform for inference
infer_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def load_trained_model(device):
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint['class_names']
    num_classes = checkpoint['num_classes']
    model = build_model(num_classes=num_classes, device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, class_names


def simulate_quality_scores(confidence, is_healthy):
    """
    Simulates colour, size, ripeness scores based on
    model confidence and healthy/rotten prediction.
    These represent what sensors or additional models would provide.
    """
    if is_healthy:
        base = 0.75 + (confidence * 0.25)
        color = round(min(base + 0.05, 1.0) * 100, 1)
        size = round(min(base + 0.03, 1.0) * 100, 1)
        ripeness = round(min(base - 0.02, 1.0) * 100, 1)
    else:
        base = confidence * 0.65
        color = round(max(base - 0.05, 0.0) * 100, 1)
        size = round(max(base, 0.0) * 100, 1)
        ripeness = round(max(base - 0.08, 0.0) * 100, 1)
    return color, size, ripeness


def assign_grade(color, size, ripeness):
    """
    Grade A: Color >= 75%, Size >= 80%, Ripeness >= 70%
    Grade B: Color >= 65%, Size >= 70%, Ripeness >= 60%
    Grade C: Below Grade B thresholds
    As specified in the case study.
    """
    if color >= 75 and size >= 80 and ripeness >= 70:
        return "A"
    elif color >= 65 and size >= 70 and ripeness >= 60:
        return "B"
    else:
        return "C"


def get_inventory_action(grade):
    """Recommend inventory action based on grade."""
    actions = {
        "A": "✅ Stock at full price — premium quality",
        "B": "⚠️  Apply 15-20% discount — good but not premium",
        "C": "❌ Remove from sale or apply heavy discount (>40%)"
    }
    return actions[grade]


def predict_image(image_path, model, class_names, device):
    """Run full prediction and grading pipeline on a single image."""
    image = Image.open(image_path).convert("RGB")
    tensor = infer_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)

    confidence = confidence.item()
    predicted_class = class_names[predicted_idx.item()]

    # Parse class name e.g. 'Apple__Healthy' → produce, condition
    parts = predicted_class.split("__")
    produce = parts[0]
    condition = parts[1] if len(parts) > 1 else "Unknown"
    is_healthy = condition.lower() == "healthy"

    # Get quality scores and grade
    color, size, ripeness = simulate_quality_scores(confidence, is_healthy)
    grade = assign_grade(color, size, ripeness)
    action = get_inventory_action(grade)

    result = {
        "produce": produce,
        "condition": condition,
        "confidence": round(confidence * 100, 2),
        "color_score": color,
        "size_score": size,
        "ripeness_score": ripeness,
        "grade": grade,
        "inventory_action": action
    }

    return result


def print_result(result):
    print("\n" + "="*50)
    print(f"  Produce      : {result['produce']}")
    print(f"  Condition    : {result['condition']}")
    print(f"  Confidence   : {result['confidence']}%")
    print("-"*50)
    print(f"  Color Score  : {result['color_score']}%")
    print(f"  Size Score   : {result['size_score']}%")
    print(f"  Ripeness     : {result['ripeness_score']}%")
    print("-"*50)
    print(f"  Grade        : {result['grade']}")
    print(f"  Action       : {result['inventory_action']}")
    print("="*50 + "\n")


if __name__ == "__main__":
    import sys
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_trained_model(device)

    # Test with a sample image from the dataset
    import os
    sample_dir = os.path.join("dataset", "Fruit And Vegetable Diseases Dataset", "Apple__Healthy")
    sample_image = os.path.join(sample_dir, os.listdir(sample_dir)[0])

    print(f"Testing with: {sample_image}")
    result = predict_image(sample_image, model, class_names, device)
    print_result(result)