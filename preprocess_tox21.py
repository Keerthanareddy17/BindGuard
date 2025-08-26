from rdkit import Chem
from rdkit.Chem import PandasTools
import pandas as pd

sdf_path = "/Users/katasanikeerthanareddy/Documents/BindGuard/data/tox21_10k_data_all.sdf"

# Load SDF into a Pandas DataFrame
df = PandasTools.LoadSDF(sdf_path)

# Check columns
print("Columns in SDF:", df.columns)

# Ensure 'SMILES' column exists
if 'SMILES' not in df.columns:
    df['SMILES'] = df['ROMol'].apply(lambda mol: Chem.MolToSmiles(mol) if mol else None)

# Drop rows without SMILES
df = df.dropna(subset=['SMILES'])

# Save to CSV
csv_path = "data/tox21/tox21_10k_data_all.csv"
df.to_csv(csv_path, index=False)
print(f"SDF converted to CSV and saved at {csv_path}")
