import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset import LabeledDataset
from model import SurrogateMLP

# Load updated labeled dataset
labeled_ds = LabeledDataset("agent/data/labeled_fp.csv")
train_loader = DataLoader(labeled_ds, batch_size=4, shuffle=True)

# Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = SurrogateMLP().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)  # smaller LR for stability(.......since we are using a smaller and simulated dataset :) )

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        if torch.isnan(loss):
            raise ValueError("NaN loss encountered — check inputs or labels.")
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# Save updated model (always state_dict only)
torch.save(model.state_dict(), "agent/model/surrogate_model.pt")
print("Updated surrogate model saved at agent/model/surrogate_model.pt")
