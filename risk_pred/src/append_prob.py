import pandas as pd
import numpy as np
import torch
import joblib
from pathlib import Path
from .model_grade import FlexibleScoringNet
from src.utils import preprocess_data

PATH = Path.cwd()
DATA_PATH = PATH / "data"
MODELS_PATH = PATH / "models"
GRADE_MODEL_PATH = MODELS_PATH / "grade_models"

LAYERS = (32,)
MODEL_NAME = "grade_model_32_final.pth"
METADATA_NAME = "metadata.joblib"
INPUT_FILE = "destilled_dataset.csv"
OUTPUT_FILE = "destilled_dataset_with_probs.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def apply_model_to_dataset():
    meta = joblib.load(GRADE_MODEL_PATH / METADATA_NAME)
    scaler = meta['scaler']
    feature_order = meta['feature_order']
    revol_99 = meta['revol_99']
    avg_cur_99 = meta['avg_cur_99']

    df = pd.read_csv(DATA_PATH / INPUT_FILE)
    print(f"Num of observations: {len(df)}")
    
    df_proc = preprocess_data(
        df, 
        feature_order=feature_order, 
        revol_99=revol_99, 
        avg_cur_99=avg_cur_99
    )

    X_scaled = scaler.transform(df_proc[feature_order])
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

    input_dim = len(feature_order)
    model = FlexibleScoringNet(input_dim, LAYERS).to(device)
    model.load_state_dict(torch.load(GRADE_MODEL_PATH / MODEL_NAME, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

    df['probability'] = probs

    df.to_csv(DATA_PATH / OUTPUT_FILE, index=False)
    print(f"Num of observations: {len(df)}")

if __name__ == "__main__":
    apply_model_to_dataset()