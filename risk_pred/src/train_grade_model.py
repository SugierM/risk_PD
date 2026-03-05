import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib
import os
import json
from pathlib import Path
from .model_grade import FlexibleScoringNet
from src.utils import preprocess_data

"""
Train grade model
"""

PATH = Path.cwd()
DATA_PATH = PATH / "data"
MODELS_PATH = PATH / "models"
GRADE_MODEL_PATH = MODELS_PATH / "grade_models"
REPORT_PATH = PATH / "reports" / "training_report.txt"
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(GRADE_MODEL_PATH, exist_ok=True)

MAX_EPOCHS = 20
LR = 0.001
LAYERS = (32,)
PATIENCE = 3
BATCH_SIZE = 1024


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, X_tensor, y_tensor, return_cm=False):
    model.eval()
    X_tensor = X_tensor.to(device)
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        y_true = y_tensor.cpu().numpy()
        auc = roc_auc_score(y_true, probs)
        recall = recall_score(y_true, preds)
        if return_cm:
            return auc, recall, confusion_matrix(y_true, preds)
        return auc, recall


def train_grade_model():
    df_train = pd.read_csv(DATA_PATH / "train_grade.csv")
    df_test = pd.read_csv(DATA_PATH / "test_grade.csv")

    revol_99 = df_train["revol_bal"].quantile(0.99)
    avg_cur_99 = df_train["avg_cur_bal"].quantile(0.99)
    df_train = preprocess_data(df_train, revol_99=revol_99, avg_cur_99=avg_cur_99)
    X = df_train.drop(columns=['target'])
    y = df_train['target'].values
    feature_order = X.columns.tolist()

    df_test_proc = preprocess_data(df_test, feature_order=feature_order, revol_99=revol_99, avg_cur_99=avg_cur_99)
    X_test = df_test_proc
    y_test = df_test['target'].values

    scaler = StandardScaler()
    scaler.fit(X[feature_order])
    X_scaled = scaler.transform(X)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump({'scaler': scaler, 'feature_order': feature_order, 'revol_99': revol_99, 'avg_cur_99': avg_cur_99}, GRADE_MODEL_PATH / 'metadata.joblib')

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

    class_counts = np.bincount(y.astype(int))
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    samples_weights = weights[y.astype(int)]
    sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

    train_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=BATCH_SIZE, sampler=sampler)

    model = FlexibleScoringNet(len(feature_order), LAYERS).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)

    best_auc = 0
    epochs_no_improve = 0
    name = "grade_model_32_final"

    print(f"Using: {device}")
    for epoch in range(MAX_EPOCHS):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        current_auc, _ = evaluate(model, X_test_tensor, y_test_tensor)
        scheduler.step(current_auc)
        
        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(train_loader):.4f} | AUC: {current_auc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if current_auc > best_auc:
            best_auc = current_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), GRADE_MODEL_PATH / f'{name}.pth')
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= PATIENCE:
            print(f"Early Stopping. Best AUC: {best_auc:.4f}")
            break

    model.load_state_dict(torch.load(GRADE_MODEL_PATH / f'{name}.pth'))
    f_auc, f_recall, f_cm = evaluate(model, X_test_tensor, y_test_tensor, return_cm=True)


    ########################################################################
    with open(DATA_PATH / "columns_grade.json", "r", encoding="utf-8") as f:
        columns_grade = json.load(f)


    report_lines = [
        "="*50,
        f"Report created at {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Config:\n",
        f"Model name: {name}",
        f"Number of neurons in each layer: {LAYERS}",
        f"Learning Rate: {LR}",
        f"Max Epochs: {MAX_EPOCHS}",
        
        "\nFeatures used",
        f"Number of raw features: {len(columns_grade)}",
        f"Number of features after prepare_df: {len(feature_order)}",
        f"Training dataset: {len(df_train)}",
        f"Test dataset: {len(df_test)}",
        
        "\nResults",
        f"Final AUC: {f_auc:.4f}",
        f"Final Recall: {f_recall:.4f}",
        "\nConfusion Matrix:",
        str(f_cm),
        
        "\nFiles",
        f"Model saved to: {GRADE_MODEL_PATH / f'{name}.pth'}",
        f"Metadata saved to {GRADE_MODEL_PATH / 'metadata.joblib'}",
    ]

    report_text = "\n".join(report_lines)
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write(report_text)

    print(f"Raport został zapisany w: {REPORT_PATH}")



if __name__ == "__main__":
    train_grade_model()