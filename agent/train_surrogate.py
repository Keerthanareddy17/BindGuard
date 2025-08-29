import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from dataset import LabeledDataset
from model import SurrogateMLP

# Loadin the dataset
labeled_ds = LabeledDataset("agent/data/labeled_fp.csv")
train_loader = DataLoader(labeled_ds, batch_size=4, shuffle=True)

# Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = SurrogateMLP().to(device)
criterion = nn.MSELoss()  # regression
optimizer = optim.Adam(model.parameters(), lr=1e-3)

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
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "agent/model/surrogate_model.pt")
print("Surrogate model saved at agent/model/surrogate_model.pt")
