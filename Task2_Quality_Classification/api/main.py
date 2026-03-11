from fastapi import FastAPI
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
sys.path.append(str(PROJECT_DIR))

from model_loader import load_model
from interaction_logger import log_interaction

app = FastAPI(title="BFN AI Model Deployment API")


@app.get("/")
def home():
    return {"message": "BFN AI Deployment API running"}


@app.post("/predict")
def predict(user_id: str, item: str):
    model, model_name = load_model()
    prediction = "fresh"
    log_interaction(user_id, item, prediction)

    return {
        "model_used": model_name,
        "item": item,
        "prediction": prediction
    }