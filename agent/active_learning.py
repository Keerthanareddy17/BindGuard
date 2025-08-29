import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Config
LABELED_FP = "agent/data/labeled_fp.csv"
UNLABELED_FP = "agent/data/unlabeled_fp.csv"
SURROGATE_MODEL = "agent/model/surrogate_model.pt"

BATCH_SIZE = 16
MC_PASSES = 20
TOP_K = 5  
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Surrogate model 
class SurrogateMLP(nn.Module):
    """
    Surrogate model trained to predict binding scores.
    Architecture matches model.py used during training.
    """
    def __init__(self, input_dim=2048):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        return torch.sigmoid(self.out(x))

# Load trained model (always state_dict only)
model = SurrogateMLP().to(DEVICE)
checkpoint = torch.load(SURROGATE_MODEL, map_location=DEVICE)
if "model_state" in checkpoint:
    model.load_state_dict(checkpoint["model_state"])
else:
    model.load_state_dict(checkpoint)

model.eval()

# Enable dropout for MC-Dropout
def enable_dropout(m):
    """Enable dropout layers during inference for MC-dropout"""
    for layer in m.modules():
        if isinstance(layer, nn.Dropout):
            layer.train()

# Load unlabeled data
def load_unlabeled_data(csv_path):
    df = pd.read_csv(csv_path)
    features = df.filter(regex="^FP_").values
    return df, torch.tensor(features, dtype=torch.float32)

# MC-Dropout predictions
def mc_dropout_predictions(model, X_tensor, n_passes=MC_PASSES):
    enable_dropout(model)  # force dropout ON
    all_preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            preds = model(X_tensor.to(DEVICE)).cpu().numpy()
            all_preds.append(preds)
    all_preds = np.stack(all_preds)  # shape: (n_passes, n_samples, 1)
    mean_preds = all_preds.mean(axis=0).flatten()
    var_preds = all_preds.var(axis=0).flatten()
    return mean_preds, var_preds

# Select top uncertain molecules
def select_top_uncertain(df, X_tensor, top_k=TOP_K):
    mean_preds, var_preds = mc_dropout_predictions(model, X_tensor)
    df["predicted_score"] = mean_preds
    df["uncertainty"] = var_preds
    selected = df.sort_values("uncertainty", ascending=False).head(top_k)
    return selected

if __name__ == "__main__":
    print("Loading unlabeled data...")
    unlabeled_df, X_unlabeled = load_unlabeled_data(UNLABELED_FP)
    print(f"Unlabeled molecules: {len(unlabeled_df)}")

    print("Selecting top uncertain molecules...")
    selected_df = select_top_uncertain(unlabeled_df, X_unlabeled, TOP_K)
    print("Top uncertain molecules:")
    print(selected_df[["SMILES", "predicted_score", "uncertainty"]])

    # Save selection
    selected_df.to_csv("agent/data/selected_for_labeling.csv", index=False)
    print("\nSaved selected molecules to agent/data/selected_for_labeling.csv")
