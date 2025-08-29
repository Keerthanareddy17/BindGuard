import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import LabeledDataset
from model import SurrogateMLP
from prepare_fingerprints import process_file
from update_pools import update_pools
import matplotlib.pyplot as plt

# CONFIG 
MOLECULES_CSV = "agent/data/molecules.csv"
LABELED_FP = "agent/data/labeled_fp.csv"
UNLABELED_FP = "agent/data/unlabeled_fp.csv"
SELECTED_CSV = "agent/data/selected_for_labeling.csv"
SURROGATE_MODEL = "agent/model/surrogate_model.pt"

BATCH_SIZE = 16
MC_PASSES = 20
TOP_K = 5
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
EPOCHS = 50
LR = 5e-4
ITERATIONS = 5  # number of active learning cycles

iteration_metrics = {
    "iteration": [],
    "val_loss": [],
    "avg_uncertainty": []
}

# UTILS 
def enable_dropout(m):
    for layer in m.modules():
        if isinstance(layer, nn.Dropout):
            layer.train()

def mc_dropout_predictions(model, X_tensor, n_passes=MC_PASSES):
    enable_dropout(model)
    all_preds = []
    with torch.no_grad():
        for _ in range(n_passes):
            preds = model(X_tensor.to(DEVICE)).cpu().numpy()
            all_preds.append(preds)
    all_preds = np.stack(all_preds)
    mean_preds = all_preds.mean(axis=0).flatten()
    var_preds = all_preds.var(axis=0).flatten()
    return mean_preds, var_preds

def select_top_uncertain(model, df, X_tensor, top_k=TOP_K):
    if len(df) == 0:
        return pd.DataFrame(columns=["SMILES", "predicted_score", "uncertainty"]), 0.0

    mean_preds, var_preds = mc_dropout_predictions(model, X_tensor)
    
    # Ensuring numeric vals
    mean_preds = np.nan_to_num(mean_preds, nan=0.0)
    var_preds = np.nan_to_num(var_preds, nan=0.0)
    
    df = df.copy()
    df["predicted_score"] = mean_preds
    df["uncertainty"] = var_preds
    
    # Fallback if all variance is zero
    if var_preds.max() == 0:
        print("All uncertainties zero, selecting random molecules as fallback")
        selected = df.sample(min(top_k, len(df)))
    else:
        selected = df.sort_values("uncertainty", ascending=False).head(top_k)
    
    avg_unc = selected["uncertainty"].mean() if not selected.empty else 0.0
    return selected, avg_unc

# SURROGATE RETRAIN 
def retrain_surrogate():
    labeled_ds = LabeledDataset(LABELED_FP)
    if len(labeled_ds) < 2:
        print("Not enough labeled data to train surrogate, skipping retrain.")
        return None, 0.0

    n_val = max(1, int(0.1 * len(labeled_ds)))
    train_ds, val_ds = torch.utils.data.random_split(labeled_ds, [len(labeled_ds)-n_val, n_val])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = SurrogateMLP().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    # Validation loss
    model.eval()
    val_losses = []
    with torch.no_grad():
        for X_val, y_val in val_loader:
            X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
            pred = model(X_val)
            loss = criterion(pred, y_val)
            val_losses.append(loss.item())
    val_loss = np.mean(val_losses)
    
    torch.save(model.state_dict(), SURROGATE_MODEL)
    print(f"Updated surrogate model saved! Val Loss: {val_loss:.4f}")
    return model, val_loss

if __name__ == "__main__":
    # 1. Prepare fingerprints if not exist
    for f in ["labeled.csv", "unlabeled.csv"]:
        fp_file = f"agent/data/{f.split('.')[0]}_fp.csv"
        if not os.path.exists(fp_file):
            process_file(f)

    # 2. Prepare initial pools if not exist
    if not os.path.exists(LABELED_FP) or not os.path.exists(UNLABELED_FP):
        from utils import prepare_initial_pool
        prepare_initial_pool(MOLECULES_CSV, labeled_fraction=0.5)

    # 3. Run active learning loop
    for iteration in range(ITERATIONS):
        print(f"\n--- Active Learning Iteration {iteration+1} ---")
        
        # Load unlabeled data
        df_unlabeled = pd.read_csv(UNLABELED_FP)
        if df_unlabeled.empty:
            print("No unlabeled molecules left, stopping loop.")
            break

        X_unlabeled = torch.tensor(df_unlabeled.filter(regex="^FP_").astype(float).values, dtype=torch.float32)
        
        # Load surrogate model
        model = SurrogateMLP().to(DEVICE)
        if os.path.exists(SURROGATE_MODEL):
            model.load_state_dict(torch.load(SURROGATE_MODEL, map_location=DEVICE))
        model.eval()
        
        # Select top uncertain molecules
        selected_df, avg_uncertainty = select_top_uncertain(model, df_unlabeled, X_unlabeled, TOP_K)
        selected_df.to_csv(SELECTED_CSV, index=False)
        print("Top uncertain molecules selected:")
        print(selected_df[["SMILES", "predicted_score", "uncertainty"]])
        
        # Update pools
        update_pools()
        
        # Retrain surrogate
        model, val_loss = retrain_surrogate()
        
        # Log metrics
        iteration_metrics["iteration"].append(iteration + 1)
        iteration_metrics["val_loss"].append(val_loss)
        iteration_metrics["avg_uncertainty"].append(avg_uncertainty)

    plt.figure(figsize=(8,5))
    plt.plot(iteration_metrics["iteration"], iteration_metrics["val_loss"], marker='o', label="Validation Loss")
    plt.xlabel("Active Learning Iteration")
    plt.ylabel("Validation MSE")
    plt.title("Surrogate Model Improvement Over Active Learning Iterations")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(8,5))
    plt.plot(iteration_metrics["iteration"], iteration_metrics["avg_uncertainty"], marker='o', color='orange', label="Avg Selected Uncertainty")
    plt.xlabel("Active Learning Iteration")
    plt.ylabel("Average Uncertainty")
    plt.title("Reduction in Model Uncertainty Over Iterations")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Savin metrics to CSV
    pd.DataFrame(iteration_metrics).to_csv("agent/data/active_learning_metrics.csv", index=False)
    print("\nMetrics saved to agent/data/active_learning_metrics.csv")
