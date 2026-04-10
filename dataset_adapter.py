"""
Universal Dataset Adapter
=========================
Converts any bank fraud CSV (regardless of column names) into the
standard 17-feature schema used by the federated learning model.

Supported Input Datasets:
  - creditcard.csv        (Bank 2 - EU Credit Card Fraud)
  - PaySim CSV            (Bank 3 - Mobile Money Fraud)
  - synthetic_fraud_dataset.csv  (Bank 1 - already compatible)

Output Schema (17 features + label):
  Transaction_Amount, Transaction_Type, Account_Balance, Device_Type,
  Location, Merchant_Category, IP_Address_Flag, Previous_Fraudulent_Activity,
  Daily_Transaction_Count, Avg_Transaction_Amount_7d, Failed_Transaction_Count_7d,
  Card_Type, Card_Age, Transaction_Distance, Authentication_Method,
  Risk_Score, Is_Weekend, Fraud_Label
"""

import pandas as pd
import numpy as np
import os

# ─── Standard schema the model expects ───────────────────────────────────────
STANDARD_COLUMNS = [
    "Transaction_Amount",
    "Transaction_Type",
    "Account_Balance",
    "Device_Type",
    "Location",
    "Merchant_Category",
    "IP_Address_Flag",
    "Previous_Fraudulent_Activity",
    "Daily_Transaction_Count",
    "Avg_Transaction_Amount_7d",
    "Failed_Transaction_Count_7d",
    "Card_Type",
    "Card_Age",
    "Transaction_Distance",
    "Authentication_Method",
    "Risk_Score",
    "Is_Weekend",
    "Fraud_Label"
]

# ─── Auto-detect dataset type ─────────────────────────────────────────────────
def detect_dataset_type(df: pd.DataFrame) -> str:
    cols = set(df.columns.str.lower())
    if "class" in cols and "v1" in cols:
        return "creditcard"
    if "isfraud" in cols and "nameOrig" in df.columns or "nameorig" in cols:
        return "paysim"
    if "fraud_label" in cols and "transaction_amount" in cols:
        return "synthetic"
    # Fallback: try to guess from columns
    if "amount" in cols and "class" in cols:
        return "creditcard"
    if "isfraud" in cols:
        return "paysim"
    return "unknown"


# ─── Adapter: Credit Card Dataset ────────────────────────────────────────────
def adapt_creditcard(df: pd.DataFrame) -> pd.DataFrame:
    """
    creditcard.csv schema:
      Time, V1..V28 (PCA), Amount, Class
    
    Strategy:
      - Amount            → Transaction_Amount
      - Class             → Fraud_Label
      - V1 median range   → proxy for various features
      - Time % 86400      → Is_Weekend proxy
    """
    print("  Adapting Credit Card dataset (EU Bank)...")
    out = pd.DataFrame()

    out["Transaction_Amount"]          = df["Amount"]
    # V1 range gives signed magnitude – map to transaction type bucket 0-3
    out["Transaction_Type"]            = (df["V1"].rank(pct=True) * 3).astype(int).clip(0, 3)
    # V3 correlates with account activity – use as Account_Balance proxy (scale to realistic range)
    out["Account_Balance"]             = ((df["V3"] - df["V3"].min()) /
                                          (df["V3"].max() - df["V3"].min()) * 100000).fillna(50000)
    # V4 → Device_Type bucket
    out["Device_Type"]                 = (df["V4"].rank(pct=True) * 3).astype(int).clip(0, 3)
    # V5 → Location bucket (0–50)
    out["Location"]                    = (df["V5"].rank(pct=True) * 50).astype(int).clip(0, 50)
    # V6 → Merchant_Category bucket (0–5)
    out["Merchant_Category"]           = (df["V6"].rank(pct=True) * 5).astype(int).clip(0, 5)
    # V7 extremity → IP_Address_Flag
    out["IP_Address_Flag"]             = ((df["V7"].abs() > df["V7"].abs().quantile(0.9)).astype(int))
    # V8 → Previous_Fraudulent_Activity proxy (0–10)
    out["Previous_Fraudulent_Activity"]= (df["V8"].rank(pct=True) * 10).astype(int).clip(0, 10)
    # V9 → Daily_Transaction_Count proxy (1–50)
    out["Daily_Transaction_Count"]     = (df["V9"].rank(pct=True) * 49 + 1).astype(int).clip(1, 50)
    # V10 → Avg_Transaction_Amount_7d (scaled like Amount)
    out["Avg_Transaction_Amount_7d"]   = (df["V10"].rank(pct=True) * df["Amount"].max()).clip(0)
    # V11 → Failed_Transaction_Count_7d (0–10)
    out["Failed_Transaction_Count_7d"] = (df["V11"].rank(pct=True) * 10).astype(int).clip(0, 10)
    # V12 → Card_Type (0=Credit, 1=Debit, 2=Prepaid, 3=Gift)
    out["Card_Type"]                   = (df["V12"].rank(pct=True) * 3).astype(int).clip(0, 3)
    # V13 → Card_Age months (1–120)
    out["Card_Age"]                    = (df["V13"].rank(pct=True) * 119 + 1).astype(int).clip(1, 120)
    # V14 → Transaction_Distance km (0–5000)
    out["Transaction_Distance"]        = (df["V14"].rank(pct=True) * 5000).clip(0, 5000)
    # V15 → Authentication_Method (0–3)
    out["Authentication_Method"]       = (df["V15"].rank(pct=True) * 3).astype(int).clip(0, 3)
    # Amount percentile → Risk_Score (0–1)
    out["Risk_Score"]                  = df["Amount"].rank(pct=True).round(4)
    # Time mod 86400 in midnight-to-6am window → weekend proxy
    out["Is_Weekend"]                  = ((df["Time"] % 86400) < 21600).astype(int)
    # Target
    out["Fraud_Label"]                 = df["Class"].astype(int)

    print(f"    Rows: {len(out):,}  |  Fraud rate: {out['Fraud_Label'].mean()*100:.2f}%")
    return out


# ─── Adapter: PaySim Dataset ──────────────────────────────────────────────────
def adapt_paysim(df: pd.DataFrame) -> pd.DataFrame:
    """
    PaySim schema:
      step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
      nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud
    
    Strategy:
      - amount              → Transaction_Amount
      - type                → Transaction_Type (encoded)
      - oldbalanceOrg       → Account_Balance
      - isFraud             → Fraud_Label
      - derived features from balances
    """
    print("  Adapting PaySim dataset (Mobile Money Bank)...")
    out = pd.DataFrame()

    # Encode transaction type
    type_map = {"CASH_IN": 0, "CASH_OUT": 1, "DEBIT": 2, "PAYMENT": 3, "TRANSFER": 4}
    df["type_encoded"] = df["type"].map(type_map).fillna(0).astype(int)

    out["Transaction_Amount"]           = df["amount"]
    out["Transaction_Type"]             = df["type_encoded"].clip(0, 3)
    out["Account_Balance"]              = df["oldbalanceOrg"].clip(0)
    # PaySim is mobile-only → Device_Type = 1 (Mobile)
    out["Device_Type"]                  = 1
    # step % 60 → Location bucket
    out["Location"]                     = (df["step"] % 50).astype(int)
    # type bucket as merchant category
    out["Merchant_Category"]            = df["type_encoded"].clip(0, 5)
    # Flag if dest account old balance was 0 (suspicious pattern)
    out["IP_Address_Flag"]              = (df["oldbalanceDest"] == 0).astype(int)
    # isFlaggedFraud → Previous_Fraudulent_Activity
    out["Previous_Fraudulent_Activity"] = df["isFlaggedFraud"].astype(int) * 10
    # step hour proxy → Daily_Transaction_Count
    out["Daily_Transaction_Count"]      = ((df["step"] % 24) + 1).astype(int).clip(1, 50)
    # 7d avg: use dest old balance as proxy for avg environment
    out["Avg_Transaction_Amount_7d"]    = df["oldbalanceDest"].clip(0, 500000)
    # Balance difference as Failed transaction proxy
    balance_diff = (df["oldbalanceOrg"] - df["newbalanceOrig"]).abs()
    out["Failed_Transaction_Count_7d"]  = (balance_diff.rank(pct=True) * 10).astype(int).clip(0, 10)
    # No card info in PaySim → default Debit (1)
    out["Card_Type"]                    = 1
    out["Card_Age"]                     = 36  # default 3 years
    # Distance: TRANSFER/CASH_OUT have higher distance
    out["Transaction_Distance"]         = df["type_encoded"].apply(
        lambda t: np.random.randint(100, 500) if t in [1, 4] else np.random.randint(0, 50)
    )
    # Authentication: 0=PIN, 1=Password, 2=Biometric, 3=OTP
    out["Authentication_Method"]        = df["type_encoded"].apply(
        lambda t: 3 if t == 4 else 0  # Wire/Transfer → OTP, else PIN
    )
    # Risk score based on amount percentile
    out["Risk_Score"]                   = df["amount"].rank(pct=True).round(4)
    # step // 168 gives week number, alternate weeks are "weekend"
    out["Is_Weekend"]                   = ((df["step"] // 24) % 7 >= 5).astype(int)
    # Target
    out["Fraud_Label"]                  = df["isFraud"].astype(int)

    print(f"    Rows: {len(out):,}  |  Fraud rate: {out['Fraud_Label'].mean()*100:.2f}%")
    return out


# ─── Adapter: Synthetic (already compatible) ─────────────────────────────────
def adapt_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    print("  Synthetic dataset already in standard format.")
    # Just ensure all required columns exist
    for col in STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    return df[STANDARD_COLUMNS]


# ─── Main adapter function ────────────────────────────────────────────────────
def adapt_dataset(csv_path: str, output_path: str, sample_size: int = 200000) -> bool:
    """
    Load a CSV from csv_path, auto-detect its type, adapt it to the
    standard 17-feature schema, and save to output_path.
    
    sample_size: max rows to use (for large datasets like PaySim)
    """
    if not os.path.exists(csv_path):
        print(f"  ERROR: File not found: {csv_path}")
        return False

    print(f"\nLoading: {csv_path}")
    df = pd.read_csv(csv_path, nrows=sample_size)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")

    dtype = detect_dataset_type(df)
    print(f"  Auto-detected type: {dtype}")

    if dtype == "creditcard":
        adapted = adapt_creditcard(df)
    elif dtype == "paysim":
        adapted = adapt_paysim(df)
    elif dtype == "synthetic":
        adapted = adapt_synthetic(df)
    else:
        print(f"  WARNING: Unknown dataset type. Attempting generic adaptation...")
        adapted = adapt_generic(df)

    # Ensure correct column order and fill any gaps
    for col in STANDARD_COLUMNS:
        if col not in adapted.columns:
            adapted[col] = 0
    adapted = adapted[STANDARD_COLUMNS]

    # Final cleanup
    adapted = adapted.fillna(0)
    adapted = adapted.replace([np.inf, -np.inf], 0)

    adapted.to_csv(output_path, index=False)
    print(f"  Saved adapted dataset -> {output_path}")
    print(f"  Final shape: {adapted.shape}")
    return True


def adapt_generic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback adapter for completely unknown schemas.
    Tries to find label column and uses numeric columns as features.
    """
    print("  Running generic adaptation...")
    out = pd.DataFrame()

    # Find target column
    label_candidates = ["fraud", "is_fraud", "isfraud", "label", "class", "target", "fraud_label"]
    label_col = None
    for c in df.columns:
        if c.lower() in label_candidates:
            label_col = c
            break

    if label_col is None:
        print("  ERROR: Cannot find fraud label column. Please check dataset.")
        return pd.DataFrame(columns=STANDARD_COLUMNS)

    # Find amount column
    amount_candidates = ["amount", "transaction_amount", "transactionamount", "amt"]
    amount_col = None
    for c in df.columns:
        if c.lower() in amount_candidates:
            amount_col = c
            break

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)

    out["Transaction_Amount"] = df[amount_col] if amount_col else (df[numeric_cols[0]] if numeric_cols else 0)
    out["Fraud_Label"] = df[label_col].astype(int)

    # Fill remaining features from numeric columns (cycling through them)
    feature_cols = STANDARD_COLUMNS[1:-1]  # skip Transaction_Amount and Fraud_Label
    for i, feat in enumerate(feature_cols):
        if i < len(numeric_cols):
            out[feat] = df[numeric_cols[i]].fillna(0)
        else:
            out[feat] = 0

    return out


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("  Universal Dataset Adapter - Multi-Bank FL Platform")
    print("=" * 60)

    # Bank 1: already in standard format (no adaptation needed)
    print("\n[Bank 1] synthetic_fraud_dataset.csv -> Already standard (no conversion needed)")

    # Bank 2: Credit Card Fraud (EU Bank)
    adapt_dataset(
        csv_path=os.path.join(BASE_DIR, "creditcard.csv"),
        output_path=os.path.join(BASE_DIR, "bank2_adapted.csv"),
        sample_size=200000
    )

    # Bank 3: PaySim Mobile Money (African Bank)
    adapt_dataset(
        csv_path=os.path.join(BASE_DIR, "PS_20174392719_1491204439457_log.csv"),
        output_path=os.path.join(BASE_DIR, "bank3_adapted.csv"),
        sample_size=200000
    )

    print("\n[DONE] All datasets adapted! Ready for federated training.")
    print("   Bank 1 -> synthetic_fraud_dataset.csv")
    print("   Bank 2 -> bank2_adapted.csv")
    print("   Bank 3 -> bank3_adapted.csv")
