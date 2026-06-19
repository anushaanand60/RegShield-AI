import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from config import (
    ARTIFACT_DIR, CATEGORICAL_COLS, DOMINANT_FLAG_COL, HIGH_NULL_THRESHOLD,
    HINT_FEATURES, MONTH_MAP, RANDOM_STATE, RAW_DATA_PATH, TARGET_COL, TENURE_MAP,
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

def leakage_safe_screen(df, target_col = TARGET_COL, n_splits = 5, top_k = 60):
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].fillna(df[feature_cols].median(numeric_only=True))
    y = df[target_col]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_importances = pd.DataFrame(0.0, index=feature_cols, columns=[f"fold_{i}" for i in range(n_splits)])

    print("\n[INFO] Starting loop-isolated feature screening...")
    for fold_idx, (train_idx, _) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        
        rf = RandomForestClassifier(
            n_estimators=150, max_depth=8, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        fold_importances[f"fold_{fold_idx}"] = rf.feature_importances_
        print(f"   -> Fold {fold_idx + 1}/{n_splits} complete.")

    mean_importance = fold_importances.mean(axis=1).sort_values(ascending=False)
    cv_std = fold_importances.std(axis=1)

    result = pd.DataFrame({
        "feature": mean_importance.index,
        "mean_importance": mean_importance.values,
        "cv_std": cv_std.reindex(mean_importance.index).values,
    })
    result["is_hint_feature"] = result["feature"].isin(HINT_FEATURES)
    result["rank"] = range(1, len(result) + 1)

    result.to_csv(ARTIFACT_DIR / "feature_screening_full.csv", index=False)
    top_features = result.head(top_k)
    top_features.to_csv(ARTIFACT_DIR / "feature_screening_top_k.csv", index=False)

    print(f"[SUCCESS] Top {top_k} features isolated and saved to Outputs/ directory.")
    return result


def f3912_ablation_screen(screening_result, flag_col = DOMINANT_FLAG_COL):
    if flag_col not in screening_result["feature"].values:
        return {"flag_col": flag_col, "present": False}

    row = screening_result.loc[screening_result["feature"] == flag_col].iloc[0]
    total_mass = screening_result["mean_importance"].sum()
    share_pct = 100 * float(row["mean_importance"]) / total_mass

    verdict = {
        "flag_col": flag_col,
        "present": True,
        "rank": int(row["rank"]),
        "importance": round(float(row["mean_importance"]), 5),
        "share_of_total_importance_pct": round(share_pct, 2),
        "pipeline_requirement": "Bifurcate Phase 1 execution into Model-A (with) and Model-B (without)."
    }
    
    with open(ARTIFACT_DIR / "f3912_ablation_screen.json", "w") as fh:
        json.dump(verdict, fh, indent=2)
    return verdict

def build_validation_strategy(df, target_col = TARGET_COL):
    n_fraud = int((df[target_col] == 1).sum())
    n_total = len(df)
    k_folds = 5

    strategy = {
        "n_fraud_total": n_fraud,
        "recommended_k": k_folds,
        "expected_fraud_per_fold": round(n_fraud / k_folds, 1),
        "scoring_metrics": ["f1", "average_precision", "recall", "precision", "roc_auc"],
        "metrics_to_block": ["accuracy"],
        "all_legit_baseline_accuracy": round(100 * (n_total - n_fraud) / n_total, 2),
        "execution_protocol": "Execute StratifiedKFold over 3 localized random seeds. Report mean +/- std."
    }

    with open(ARTIFACT_DIR / "validation_strategy.json", "w") as fh:
        json.dump(strategy, fh, indent=2)
    return strategy

def main():
    skewed_amount_features = [
        "F3806", "F3807", "F1813", "F1165", "F1273",
        "F3805", "F3813", "F1382", "F1705", "F1825",
    ]

    print("[START] Executing RegShield AI Pipeline — Phase 0 (Base Ingestion)")
    
    raw_df = load_raw()
    structural_audit(raw_df)
    engineered_df = engineer_features(raw_df, skewed_amount_features)
    screening_res = leakage_safe_screen(engineered_df)

    f3912_ablation_screen(screening_res)
    build_validation_strategy(engineered_df)
    output_path = ARTIFACT_DIR / "engineered_dataset.parquet"
    engineered_df.to_parquet(output_path, engine="pyarrow", index=False)
    
    print(f"\n[COMPLETE] Clean dataset baseline serialized to: {output_path}")
    print(f"Data Matrix Dimensions: {engineered_df.shape[0]} rows x {engineered_df.shape[1]} features.")

if __name__ == "__main__":
    main()