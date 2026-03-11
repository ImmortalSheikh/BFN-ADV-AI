from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent / "model_registry"


def get_latest_model_path() -> Path:
    model_files = list(MODEL_DIR.glob("*.pt")) + list(MODEL_DIR.glob("*.pth"))

    if not model_files:
        raise FileNotFoundError(
            f"No model file found in {MODEL_DIR}. "
            "Please place a trained model in model_registry/."
        )

    return max(model_files, key=lambda p: p.stat().st_mtime)


def load_model():
    """
    Task 3 deployment demo:
    confirm that a model file exists in the registry and return its filename.
    """
    model_path = get_latest_model_path()
    return "registered_model", model_path.name