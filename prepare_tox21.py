import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import os

# Config
input_csv = "data/tox21/tox21_10k_data_all.csv"
output_csv = "data/tox21/tox21_cleaned.csv"
fingerprint_radius = 2
fingerprint_size = 2048

# Multi-task label columns
target_cols = [
    "NR-AR", "NR-AhR", "NR-ER", "NR-AR-LBD",
    "NR-ER-LBD", "NR-PPAR-gamma", "NR-Aromatase",
    "SR-ARE", "SR-HSE", "SR-MMP", "SR-p53", "SR-ATAD5"
]

df = pd.read_csv(input_csv)

# Cleaning FW column (keep only main number)
df["FW"] = df["FW"].apply(lambda x: str(x).split()[0] if pd.notnull(x) else x)

# Replace missing labels with -1
df[target_cols] = df[target_cols].fillna(-1)

# Drop ROMol column
if "ROMol" in df.columns:
    df = df.drop(columns=["ROMol"])

# Computin Morgan fingerprints from SMILES
def smiles_to_fp(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits, dtype=int)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

# Create fingerprint matrix
fps = np.array([smiles_to_fp(s) for s in df["SMILES"]])

# Convert fingerprints to a DataFrame
fp_df = pd.DataFrame(fps, columns=[f"FP_{i}" for i in range(fingerprint_size)])

# Combine with labels
final_df = pd.concat([df[target_cols].reset_index(drop=True), fp_df], axis=1)

# Saveing cleaned dataset
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
final_df.to_csv(output_csv, index=False)

print(f"Cleaned Tox21 dataset saved at {output_csv}")
print(f"Shape: {final_df.shape}")
