import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
import joblib
import numpy as np
import datetime
import os


data = joblib.load("prepared_tox21_data.pkl")
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val     = data["X_val"], data["y_val"]
X_test, y_test   = data["X_test"], data["y_test"]
pos_weights      = data["pos_weights"]  

# Ensure tensors
X_train_t = X_train if isinstance(X_train, torch.Tensor) else torch.tensor(X_train, dtype=torch.float32)
y_train_t = y_train if isinstance(y_train, torch.Tensor) else torch.tensor(y_train, dtype=torch.float32)
X_val_t   = X_val   if isinstance(X_val, torch.Tensor)   else torch.tensor(X_val, dtype=torch.float32)
y_val_t   = y_val   if isinstance(y_val, torch.Tensor)   else torch.tensor(y_val, dtype=torch.float32)
X_test_t  = X_test  if isinstance(X_test, torch.Tensor)  else torch.tensor(X_test, dtype=torch.float32)
y_test_t  = y_test  if isinstance(y_test, torch.Tensor)  else torch.tensor(y_test, dtype=torch.float32)
pos_weights_t = pos_weights if isinstance(pos_weights, torch.Tensor) else torch.tensor(pos_weights, dtype=torch.float32)

# DataLoaders
batch_size = 128
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False)

# Model Definition
class Tox21MLP(nn.Module):
    def __init__(self, input_dim=2048, output_dim=12):
        super().__init__()
        self.fc1, self.bn1 = nn.Linear(input_dim, 1024), nn.BatchNorm1d(1024)
        self.fc2, self.bn2 = nn.Linear(1024, 512), nn.BatchNorm1d(512)
        self.fc3, self.bn3 = nn.Linear(512, 256), nn.BatchNorm1d(256)
        self.fc4, self.bn4 = nn.Linear(256, 128), nn.BatchNorm1d(128)
        self.out = nn.Linear(128, output_dim)
        self.drop1, self.drop2 = nn.Dropout(0.35), nn.Dropout(0.35)
        self.drop3, self.drop4 = nn.Dropout(0.25), nn.Dropout(0.20)

    def forward(self, x):
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.drop3(F.relu(self.bn3(self.fc3(x))))
        x = self.drop4(F.relu(self.bn4(self.fc4(x))))
        return self.out(x)  

# Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = Tox21MLP().to(device)
pos_weights_t = pos_weights_t.to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights_t)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

# Evaluation Function
def evaluate(model, loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(yb.cpu().numpy())
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    aucs = []
    for i in range(all_targets.shape[1]):
        if len(np.unique(all_targets[:, i])) > 1:  
            aucs.append(roc_auc_score(all_targets[:, i], all_preds[:, i]))
    return np.mean(aucs), aucs

# Training Loop with Early Stopping
best_val_auc, patience_counter = 0, 0
patience, epochs = 10, 100
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

os.makedirs("models", exist_ok=True)
best_model_path = f"models/tox21_mlp_best_{timestamp}.pt"

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    val_auc, per_task_aucs = evaluate(model, val_loader)
    print(f"Epoch {epoch+1:03d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f}")

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        patience_counter = 0
        torch.save({
            "model_state": model.state_dict(),
            "val_auc": float(val_auc),
            "epoch": int(epoch+1)
        }, best_model_path)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

# Reloadin Best Model & Evaluate
checkpoint = torch.load(best_model_path, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

test_auc, test_per_task = evaluate(model, test_loader)
print(f"\nBest model from epoch {checkpoint['epoch']} | Val AUC: {checkpoint['val_auc']:.4f}")
print(f"Test Mean AUC: {test_auc:.4f}")
print("Per-task AUCs:", test_per_task)

# TorchScript Conversion
example_input = torch.rand(1, 2048).to(device)  
scripted_model = torch.jit.trace(model, example_input)

scripted_name = f"models/tox21_mlp_scripted_{timestamp}.pt"
scripted_model.save(scripted_name)

print(f"\n TorchScript model saved at: {scripted_name}")
