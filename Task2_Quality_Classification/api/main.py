from fastapi import FastAPI
from pathlib import Path
import sys
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
sys.path.append(str(PROJECT_DIR))

from grading import load_trained_model, predict_image
from interaction_logger import log_interaction
from model_loader import get_latest_model_path

app = FastAPI(title="BFN AI Model Deployment API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, class_names = load_trained_model(device)


@app.get("/")
def home():
    return {"message": "BFN AI Deployment API running"}


@app.post("/predict")
def predict(user_id: str, image_path: str):
    result = predict_image(image_path, model, class_names, device)
    model_name = get_latest_model_path().name

    prediction_summary = f"{result['produce']} - {result['condition']} - Grade {result['grade']}"
    log_interaction(user_id, image_path, prediction_summary)

    return {
        "model_used": model_name,
        "user_id": user_id,
        "image_path": image_path,
        **result
    }