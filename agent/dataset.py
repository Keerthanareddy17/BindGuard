import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

# Labeled Dataset
class LabeledDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.X = df[[col for col in df.columns if col.startswith("FP_")]].values.astype(np.float32)
        self.y = df["binding_score"].values.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

# Unlabeled Dataset
class UnlabeledDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.X = df[[col for col in df.columns if col.startswith("FP_")]].values.astype(np.float32)
        self.indices = np.arange(len(df))  # Keepingg track of original index

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Return features + original index
        return torch.tensor(self.X[idx]), int(self.indices[idx])
