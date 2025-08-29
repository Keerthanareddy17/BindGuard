import torch
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Loadingg TorchScript Model + Scaler
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = torch.jit.load("models/tox21_mlp_scripted_20250829_093505.pt", map_location=device)
model.eval()

scaler = joblib.load("scaler.pkl")

# SMILES -> Morgan fingerprint
def smiles_to_fp(smiles, n_bits=2048, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((1, n_bits), dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr[0])
    return arr

# Inference Function
def predict_toxicity(smiles):
    # 1. Convert SMILES -> fingerprint
    fp = smiles_to_fp(smiles)

    # 2. Standardize using saved scaler
    fp_scaled = scaler.transform(fp)

    # 3. Convert to torch tensor
    x = torch.tensor(fp_scaled, dtype=torch.float32).to(device)

    # 4. Run model forward
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    return probs

# Task Mapping (Tox21 12 assays)
TASKS = {
    "Nuclear Receptor Pathways": [
        ("NR-AR", "Androgen Receptor"),
        ("NR-AR-LBD", "Androgen Receptor LBD"),
        ("NR-AhR", "Aryl Hydrocarbon Receptor"),
        ("NR-Aromatase", "Aromatase"),
        ("NR-ER", "Estrogen Receptor"),
        ("NR-ER-LBD", "Estrogen Receptor LBD"),
        ("NR-PPAR-gamma", "PPAR-gamma (Metabolism)")
    ],
    "Stress Response Pathways": [
        ("SR-ARE", "Oxidative Stress (Antioxidant Response)"),
        ("SR-ATAD5", "DNA Repair (ATAD5)"),
        ("SR-HSE", "Heat Shock Response"),
        ("SR-MMP", "Mitochondrial Membrane Potential"),
        ("SR-p53", "Tumor Suppressor (p53)")
    ]
}

# Run Demo
if __name__ == "__main__":
    test_smiles = "CCO"  # ethanol :)
    preds = predict_toxicity(test_smiles)

    print(f"\nPredicted toxicity probabilities for molecule: {test_smiles}\n")

    idx = 0
    for category, assays in TASKS.items():
        print(f"{category}:")
        for short_name, desc in assays:
            prob = preds[idx]
            print(f"  {short_name} ({desc}): {prob:.4f}")
            idx += 1
        print("")