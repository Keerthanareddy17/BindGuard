import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_initial_pool(csv_path, labeled_fraction=0.5):
    df = pd.read_csv(csv_path)
    
    # Rows with binding_score are labeled
    labeled_df = df[df['binding_score'].notna()]
    
    # Split fraction of labeled_df as initial labeled set
    labeled_initial, _ = train_test_split(labeled_df, test_size=1-labeled_fraction, random_state=42)
    
    # Unlabeled pool: rows without binding_score + remaining labeled molecules
    unlabeled_df = pd.concat([df[df['binding_score'].isna()], labeled_df.drop(labeled_initial.index)])
    
    labeled_initial.to_csv("agent/data/labeled.csv", index=False)
    unlabeled_df.to_csv("agent/data/unlabeled.csv", index=False)
    
    print("Labeled and unlabeled pools created:")
    print("Labeled:", labeled_initial.shape[0], "molecules")
    print("Unlabeled:", unlabeled_df.shape[0], "molecules")
    
    return labeled_initial, unlabeled_df

if __name__ == "__main__":
    prepare_initial_pool("agent/data/molecules.csv", labeled_fraction=0.5)
