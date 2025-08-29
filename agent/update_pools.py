import pandas as pd

# Paths
MOLECULES_CSV = "agent/data/molecules.csv"
LABELED_FP = "agent/data/labeled_fp.csv"
UNLABELED_FP = "agent/data/unlabeled_fp.csv"
SELECTED_CSV = "agent/data/selected_for_labeling.csv"

def update_pools():
    # Load files
    molecules = pd.read_csv(MOLECULES_CSV)
    labeled = pd.read_csv(LABELED_FP)
    unlabeled = pd.read_csv(UNLABELED_FP)
    selected = pd.read_csv(SELECTED_CSV)

    # Simulate labeling: get true binding scores
    selected = selected.merge(molecules[['SMILES', 'binding_score']], on='SMILES', how='left')

    # Add to labeled pool
    labeled_updated = pd.concat([labeled, selected], ignore_index=True)

    # Remove rows with missing or invalid binding_score
    labeled_updated = labeled_updated.dropna(subset=["binding_score"])
    labeled_updated = labeled_updated[labeled_updated["binding_score"].between(0, 1)]

    # Remove duplicate SMILES, keep latest
    labeled_updated = labeled_updated.drop_duplicates(subset=["SMILES"], keep="last")

    # Remove from unlabeled pool
    unlabeled_updated = unlabeled[~unlabeled['SMILES'].isin(selected['SMILES'])]

    # Save updated pools
    labeled_updated.to_csv(LABELED_FP, index=False)
    unlabeled_updated.to_csv(UNLABELED_FP, index=False)

    print("Pools updated!")
    print(f"Labeled molecules: {len(labeled_updated)}")
    print(f"Unlabeled molecules: {len(unlabeled_updated)}")

if __name__ == "__main__":
    update_pools()
