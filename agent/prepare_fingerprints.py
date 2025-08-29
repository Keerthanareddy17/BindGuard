import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import os

# Config
data_dir = "agent/data"
fingerprint_size = 2048
radius = 2

def smiles_to_fp(smiles, radius=radius, nBits=fingerprint_size):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def process_file(file_name):
    df = pd.read_csv(os.path.join(data_dir, file_name))
    fps = np.array([smiles_to_fp(s) for s in df['SMILES']])
    fp_df = pd.DataFrame(fps, columns=[f'FP_{i}' for i in range(fingerprint_size)])
    
    # Combine fingerprint + binding_score
    df_final = pd.concat([df.reset_index(drop=True), fp_df], axis=1)
    
    out_file = os.path.join(data_dir, f"{file_name.split('.')[0]}_fp.csv")
    df_final.to_csv(out_file, index=False)
    print(f"Processed {file_name} -> fingerprints saved at {out_file}")
    return df_final

if __name__ == "__main__":
    process_file("labeled.csv")
    process_file("unlabeled.csv")
