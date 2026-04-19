from fastapi import FastAPI
from pathlib import Path
import sys
import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
ROOT_DIR = PROJECT_DIR.parent

# Add Task2 folder to path
sys.path.append(str(PROJECT_DIR))

# Add Task1 folder to path
TASK1_DIR = ROOT_DIR / "Task1_Demand_Prediction"
sys.path.append(str(TASK1_DIR))

from grading import load_trained_model, predict_image
from interaction_logger import log_interaction
from model_loader import get_latest_model_path
from predict import get_suggestions, demand_forecast

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

    prediction_summary = (
        f"{result['produce']} - {result['condition']} - Grade {result['grade']}"
    )
    log_interaction(user_id, image_path, prediction_summary)

    return {
        "model_used": model_name,
        "user_id": user_id,
        "image_path": image_path,
        **result
    }


@app.get("/reorder")
def reorder(customer_name: str):
    suggestions = get_suggestions(customer_name, top_n=5)
    return {
        "customer": customer_name,
        "reorder_suggestions": suggestions
    }


@app.get("/forecast")
def forecast():
    forecast_data = demand_forecast(top_n=10)
    return {
        "demand_forecast": forecast_data
    }