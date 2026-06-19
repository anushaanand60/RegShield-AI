import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import (
    ARTIFACT_DIR, CATEGORICAL_COLS, HIGH_NULL_THRESHOLD,
    MONTH_MAP, RANDOM_STATE, RAW_DATA_PATH, TARGET_COL, TENURE_MAP,
)

warnings.filterwarnings("ignore")

def load_raw(path = RAW_DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Missing source dataset at target path: {path}")
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    return df


def structural_audit(df):
    n_rows, n_cols = df.shape
    target_counts = df[TARGET_COL].value_counts().to_dict()
    n_fraud = target_counts.get(1, 0)
    n_legit = target_counts.get(0, 0)
    imbalance_ratio = n_legit / max(n_fraud, 1)

    null_frac = df.isnull().mean()
    audit = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_fraud": int(n_fraud),
        "n_legit": int(n_legit),
        "imbalance_ratio": round(imbalance_ratio, 2),
        "n_cols_fully_null": int((null_frac == 1.0).sum()),
        "n_cols_high_null": int((null_frac > HIGH_NULL_THRESHOLD).sum()),
        "n_cols_any_null": int((null_frac > 0).sum()),
        "categorical_cols_confirmed": [c for c in CATEGORICAL_COLS if c in df.columns],
    }
    
    print("\n[INFO] Structural Data Audit Summary:")
    for k, v in audit.items():
        print(f"  {k}: {v}")

    if imbalance_ratio < 50:
        raise ValueError(f"Target distribution anomaly detected. Calculated ratio: {imbalance_ratio}:1")
        
    return audit


def engineer_features(df, amount_feats):
    df = df.copy()

    log_cols = []
    for f in amount_feats:
        if f in df.columns:
            new_col = f"{f}_log"
            df[new_col] = np.log1p(df[f].clip(lower=0).fillna(0))
            log_cols.append(new_col)

    null_frac = df.isnull().mean()
    high_null_cols = null_frac[null_frac > HIGH_NULL_THRESHOLD].index.tolist()
    high_null_cols = [c for c in high_null_cols if c != TARGET_COL]
    
    wasnull_cols = []
    for f in high_null_cols:
        new_col = f"{f}_wasnull"
        df[new_col] = df[f].isnull().astype(int)
        wasnull_cols.append(new_col)

    if "F3889" in df.columns:
        df["F3889_ord"] = df["F3889"].map(TENURE_MAP).fillna(0).astype(int)

    le = LabelEncoder()
    nominal_cats = ["F3886", "F3890", "F3891", "F3892", "F3893"]
    for c in nominal_cats:
        if c in df.columns:
            df[f"{c}_enc"] = le.fit_transform(df[c].astype(str))

    drop_cols = ["F2230", "F3886", "F3888", "F3889", "F3890", "F3891", "F3892", "F3893"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    meta = {"log_cols": log_cols, "wasnull_cols": wasnull_cols, "high_null_cols": high_null_cols}
    with open(ARTIFACT_DIR / "feature_engineering_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[SUCCESS] Processed {len(log_cols)} log transforms and {len(wasnull_cols)} missingness indicators.")
    return df


def main():
    skewed_amount_features = [
        "F3806", "F3807", "F1813", "F1165", "F1273",
        "F3805", "F3813", "F1382", "F1705", "F1825",
    ]

    print("[START] Executing RegShield AI Pipeline — Phase 0 (Base Ingestion)")
    
    raw_df = load_raw()
    structural_audit(raw_df)
    engineered_df = engineer_features(raw_df, skewed_amount_features)

    output_path = ARTIFACT_DIR / "engineered_dataset.parquet"
    engineered_df.to_parquet(output_path, engine="pyarrow", index=False)
    
    print(f"\n[COMPLETE] Clean dataset baseline serialized to: {output_path}")


if __name__ == "__main__":
    main()