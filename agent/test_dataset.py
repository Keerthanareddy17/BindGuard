from dataset import LabeledDataset, UnlabeledDataset
from torch.utils.data import DataLoader

labeled_ds = LabeledDataset("agent/data/labeled_fp.csv")
unlabeled_ds = UnlabeledDataset("agent/data/unlabeled_fp.csv")

labeled_loader = DataLoader(labeled_ds, batch_size=4, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_ds, batch_size=4, shuffle=False)

for X, y in labeled_loader:
    print("Features:", X.shape, "Labels:", y.shape)
    break

for X, idx in unlabeled_loader:
    print("Features:", X.shape, "Indices:", idx)
    break
