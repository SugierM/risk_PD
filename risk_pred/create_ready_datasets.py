import pandas as pd
import torch
import joblib
import numpy as np
import json
from model_grade import FlexibleScoringNet
from utils import prepare_df


DATA_PATH = 'data/lean_dataset.csv'
NN_MODEL_PATH = 'models/best_model_Arch_128-64-32_LR_0.0005.pth'
METADATA_PATH = 'metadata.joblib'


artifacts = joblib.load(METADATA_PATH)
feature_order = artifacts['feature_order']
scaler = artifacts['scaler']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_dim = len(feature_order)
best_arch_layers = [128, 64, 32]
nn_model = FlexibleScoringNet(input_dim, best_arch_layers).to(device)
nn_model.load_state_dict(torch.load(NN_MODEL_PATH, map_location=device))
nn_model.eval()

def add_nn_probabilities(df, model, scaler, features):
    X_nn = df[features].copy()
    X_scaled = scaler.transform(X_nn)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = outputs.cpu().numpy().flatten()
    
    df['probabilities'] = probs
    return df

df_all = pd.read_csv(DATA_PATH)
df_all['issue_d'] = pd.to_datetime(df_all['issue_d'])

train_mask = (df_all['issue_d'] >= '2013-01-01') & (df_all['issue_d'] <= '2015-12-31')
test_mask  = (df_all['issue_d'] >= '2016-01-01') & (df_all['issue_d'] <= '2016-12-31')
oot_mask   = (df_all['issue_d'] >= '2017-01-01') & (df_all['issue_d'] <= '2017-12-31')

datasets = {
    'train': df_all[train_mask].copy(),
    'test':  df_all[test_mask].copy(),
    'oot':   df_all[oot_mask].copy()
}

true_datasets = {
    'train': df_all[train_mask].copy(),
    'test':  df_all[test_mask].copy(),
    'oot':   df_all[oot_mask].copy()
}

for name, df in datasets.items():
    print(f"Przetwarzanie: {name}\n")
    
    df = prepare_df(df, feature_order) 
    
    df = add_nn_probabilities(df, nn_model, scaler, feature_order)
    
    df_save = true_datasets[name]
    df_save["probability"] = df["probabilities"]

    output_name = f'data/ready_{name}_dataset.csv'
    df_save.to_csv(output_name, index=False)
    print(f"Zapisano: {output_name} (Kolumny: {df.columns})")

print("\n\n\nProces zakończony pomyślnie.")