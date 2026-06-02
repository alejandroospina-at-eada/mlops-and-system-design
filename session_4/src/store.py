import os
import joblib
from datetime import datetime
from metadata import MODELS_FOLDER


def store_model(model, model_name: str) -> None:
    """Save the model as {model_name}-{timestamp}.joblib in the models folder."""
    os.makedirs(MODELS_FOLDER, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_path = f"{MODELS_FOLDER}/{model_name}-{timestamp}.joblib"
    joblib.dump(model, model_path)
    print(f"Model stored as: {model_path}")
