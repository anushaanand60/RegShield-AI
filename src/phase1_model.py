import json
import pickle
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import ARTIFACT_DIR, RANDOM_STATE, TARGET_COL

warnings.filterwarnings("ignore")

def load_engineered_data():
    data_path = ARTIFACT_DIR / "engineered_dataset.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing engineered dataset at: {data_path}. Run Phase 0 first.")
    
    df = pd.read_parquet(data_path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    return X, y

def train_evaluate_baseline(X, y):
    print("\n[INFO] Starting Baseline CV Loop...")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    metrics_accumulator = {"f1": [], "precision": [], "recall": [], "roc_auc": []}
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_weight = n_neg / max(n_pos, 1)

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist"
        )

        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        metrics_accumulator["f1"].append(f1_score(y_test, preds))
        metrics_accumulator["precision"].append(precision_score(y_test, preds))
        metrics_accumulator["recall"].append(recall_score(y_test, preds))
        metrics_accumulator["roc_auc"].append(roc_auc_score(y_test, probs))
        print(f"   -> Fold {fold_idx + 1}/5 complete.")

    summary_metrics = {
        "mean_f1": float(np.mean(metrics_accumulator["f1"])),
        "mean_precision": float(np.mean(metrics_precision := metrics_accumulator["precision"])),
        "mean_recall": float(np.mean(metrics_accumulator["recall"])),
        "mean_roc_auc": float(np.mean(metrics_accumulator["roc_auc"]))
    }
    return summary_metrics


def main():
    print("[START] Executing RegShield AI Pipeline — Phase 1 (Baseline Building)")
    
    X, y = load_engineered_data()
    baseline_metrics = train_evaluate_baseline(X, y)
    
    with open(ARTIFACT_DIR / "baseline_model_metrics.json", "w") as fh:
        json.dump(baseline_metrics, fh, indent=2)

    print("\n[COMPLETE] Baseline model tracking complete.")
    print(f"  Baseline Mean F1 (0.5 Cutoff): {baseline_metrics['mean_f1']:.4f}")

if __name__ == "__main__":
    main()