import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import joblib

df = pd.read_csv("/Users/katasanikeerthanareddy/Documents/BindGuard/data/tox21/tox21_cleaned.csv")

# Targets and features
target_cols = [
    "NR-AR","NR-AhR","NR-ER","NR-AR-LBD","NR-ER-LBD",
    "NR-PPAR-gamma","NR-Aromatase","SR-ARE","SR-HSE",
    "SR-MMP","SR-p53","SR-ATAD5"
]
feature_cols = [col for col in df.columns if col.startswith("FP_")]

# Filter rows with missing labels (-1)
df_clean = df[df[target_cols].ge(0).all(axis=1)].reset_index(drop=True)

# Features + targets
X = df_clean[feature_cols].values
y = df_clean[target_cols].values.astype(np.float32)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")  # save for inference

# Train/val/test split (80/10/10)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_scaled, y, test_size=0.1, random_state=42, shuffle=True
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1111, random_state=42, shuffle=True
) 
# 0.1111 x 0.9 ≈ 0.1

# Class weights (for BCEWithLogitsLoss)
pos_weights = []
for i, col in enumerate(target_cols):
    pos_count = np.sum(y_train[:, i] == 1)
    neg_count = np.sum(y_train[:, i] == 0)
    if pos_count == 0:  # safeguard against divide-by-zero
        w = 1.0
    else:
        w = neg_count / pos_count
    pos_weights.append(w)
    print(f"{col}: Pos={pos_count}, Neg={neg_count}, Pos%={pos_count/(pos_count+neg_count)*100:.2f}%, pos_weight={w:.2f}")

# Convert to torch tensor
pos_weights = torch.tensor(pos_weights, dtype=torch.float32)

# Convertin data to PyTorch Tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
joblib.dump({
    "X_train": X_train_tensor,
    "y_train": y_train_tensor,
    "X_val": X_val_tensor,
    "y_val": y_val_tensor,
    "X_test": X_test_tensor,
    "y_test": y_test_tensor,
    "pos_weights": pos_weights,
    "target_cols": target_cols
}, "prepared_tox21_data.pkl")

print("\n Saved processed data to prepared_tox21_data.pkl")
print("\nData ready for PyTorch MLP!")
print("pos_weights tensor:", pos_weights)
print("Train size:", len(train_dataset), "Val size:", len(val_dataset), "Test size:", len(test_dataset))
