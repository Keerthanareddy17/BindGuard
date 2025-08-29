import torch
import torch.nn as nn
import torch.nn.functional as F

class SurrogateMLP(nn.Module):
    """
    Simple MLP surrogate model for predicting binding scores.
    Input: 2048-d fingerprint
    Output: single float binding score (0-1)
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
        x = torch.sigmoid(self.out(x))  # binding_score in [0,1]
        return x
