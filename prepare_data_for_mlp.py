import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_csv("/Users/katasanikeerthanareddy/Documents/BindGuard/data/tox21/tox21_cleaned.csv")

# targets and features
target_cols = [
    "NR-AR","NR-AhR","NR-ER","NR-AR-LBD","NR-ER-LBD",
    "NR-PPAR-gamma","NR-Aromatase","SR-ARE","SR-HSE",
    "SR-MMP","SR-p53","SR-ATAD5"
]
feature_cols = [col for col in df.columns if col.startswith("FP_")]

# Filtering out rows with missing targets (-1)
df_clean = df[df[target_cols].ge(0).all(axis=1)].reset_index(drop=True)

# Split features and targets
X = df_clean[feature_cols].values
y = df_clean[target_cols].values.astype(np.float32)  # float32 for PyTorch compatibility

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Saving scaler for future use
import joblib
joblib.dump(scaler, "scaler.pkl")

# 80% train, 10% validation, 10% test
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_scaled, y, test_size=0.1, random_state=42, shuffle=True
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1111, random_state=42, shuffle=True
) 
# 0.1111 x 0.9 ≈ 0.1, so validation is ~10% of total

# class balance check
for i, col in enumerate(target_cols):
    pos_count = np.sum(y_train[:, i] == 1)
    neg_count = np.sum(y_train[:, i] == 0)
    print(f"{col}: Pos={pos_count}, Neg={neg_count}, Pos%={pos_count/(pos_count+neg_count)*100:.2f}%")

# Shapes summary
print("X_train:", X_train.shape, "y_train:", y_train.shape)
print("X_val:", X_val.shape, "y_val:", y_val.shape)
print("X_test:", X_test.shape, "y_test:", y_test.shape)
