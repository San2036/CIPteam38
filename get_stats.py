
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

CSV_PATH = "g:/CIP/Decentralized_AI_Platform/synthetic_fraud_dataset.csv"
df = pd.read_csv(CSV_PATH)

# Preprocessing logic matching train.py
DROP_COLUMNS = ["Transaction_ID", "User_ID", "Timestamp"]
df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors='ignore')

CATEGORICAL_COLUMNS = ["Transaction_Type", "Device_Type", "Location", "Merchant_Category", "Card_Type", "Authentication_Method"]
for col in CATEGORICAL_COLUMNS:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

X = df.drop(columns=["Fraud_Label"]).values
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
cols = df.drop(columns=["Fraud_Label"]).columns.tolist()

print(f"COLUMNS: {cols}")
print(f"MEANS: {mean.tolist()}")
print(f"STDS: {std.tolist()}")
