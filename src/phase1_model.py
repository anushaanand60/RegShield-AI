import json
import pickle
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from config import ARTIFACT_DIR, DOMINANT_FLAG_COL, RANDOM_STATE, TARGET_COL

warnings.filterwarnings("ignore")

def load_engineered_data():
    data_path = ARTIFACT_DIR / "engineered_dataset.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing engineered dataset at: {data_path}. Run Phase 0 first.")
    
    df = pd.read_parquet(data_path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    return X, y

def optimize_threshold(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return (
        float(best_threshold),
        float(f1_scores[best_idx]),
        float(precisions[best_idx]),
        float(recalls[best_idx])
    )

def train_evaluate_dual_track(X, y, drop_dominant = False):
    track_label = "Model-B (Without F3912)" if drop_dominant else "Model-A (With F3912)"
    print(f"\n[INFO] Starting Multi-Seed CV Loop for: {track_label}")
    
    if drop_dominant and DOMINANT_FLAG_COL in X.columns:
        X = X.drop(columns=[DOMINANT_FLAG_COL])

    seeds = [RANDOM_STATE, RANDOM_STATE + 10]
    k_folds = 5
    
    metrics_accumulator = {
        "f1": [], "precision": [], "recall": [], "pr_auc": [], "roc_auc": [], "opt_threshold": []
    }
    
    oof_predictions = np.zeros(len(X))
    feature_importances = np.zeros(X.shape[1])

    for seed in seeds:
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
        
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
                random_state=seed,
                n_jobs=-1,
                tree_method="hist"
            )

            model.fit(X_train, y_train)
            
            probs = model.predict_proba(X_test)[:, 1]
            if seed == RANDOM_STATE:
                oof_predictions[test_idx] = probs
            feature_importances += model.feature_importances_ / (len(seeds) * k_folds)

            thresh, f1, prec, rec = optimize_threshold(y_test.values, probs)
            pr_auc = average_precision_score(y_test, probs)
            roc_auc = roc_auc_score(y_test, probs)

            metrics_accumulator["opt_threshold"].append(thresh)
            metrics_accumulator["f1"].append(f1)
            metrics_accumulator["precision"].append(prec)
            metrics_accumulator["recall"].append(rec)
            metrics_accumulator["pr_auc"].append(pr_auc)
            metrics_accumulator["roc_auc"].append(roc_auc)
            print(f"   -> Fold {fold_idx + 1}/5 complete.")

    summary_metrics = {
        "mean_f1": float(np.mean(metrics_accumulator["f1"])),
        "std_f1": float(np.std(metrics_accumulator["f1"])),
        "mean_precision": float(np.mean(metrics_accumulator["precision"])),
        "mean_recall": float(np.mean(metrics_accumulator["recall"])),
        "mean_pr_auc": float(np.mean(metrics_accumulator["pr_auc"])),
        "mean_roc_auc": float(np.mean(metrics_accumulator["roc_auc"])),
        "recommended_operational_threshold": float(np.mean(metrics_accumulator["opt_threshold"]))
    }
    print(f"   -> Processed {len(seeds) * k_folds} total folds. Mean PR-AUC: {summary_metrics['mean_pr_auc']:.4f}")
    
    n_neg_full = (y == 0).sum()
    n_pos_full = (y == 1).sum()
    
    final_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        scale_pos_weight=n_neg_full / n_pos_full,
        objective="binary:logistic", eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1
    )
    final_model.fit(X, y)

    return final_model, summary_metrics, oof_predictions, feature_importances, list(X.columns)


def main():
    print("[START] Executing RegShield AI Pipeline — Phase 1 (Baseline Building)")
    
    X, y = load_engineered_data()
    # Track A: Complete Feature Universe Execution
    model_a, metrics_a, oof_a, importances_a, features_a = train_evaluate_dual_track(X, y, drop_dominant=False)
    
    # Track B: Ablation Isolation Execution (Dropping F3912)
    model_b, metrics_b, oof_b, importances_b, features_b = train_evaluate_dual_track(X, y, drop_dominant=True)
    
    with open(ARTIFACT_DIR / "model_a_metrics.json", "w") as fh:
        json.dump(metrics_a, fh, indent=2)
    with open(ARTIFACT_DIR / "model_b_metrics.json", "w") as fh:
        json.dump(metrics_b, fh, indent=2)
    np.save(ARTIFACT_DIR / "model_a_oof_probs.npy", oof_a)
    np.save(ARTIFACT_DIR / "model_b_oof_probs.npy", oof_b)

    with open(ARTIFACT_DIR / "model_a_xgb.pkl", "b+w") as fh:
        pickle.dump(model_a, fh)
    with open(ARTIFACT_DIR / "model_b_xgb.pkl", "b+w") as fh:
        pickle.dump(model_b, fh)

    print("\n[COMPLETE] Phase 1 Execution Complete. Artifacts successfully written to Outputs/")
    print(f"  Model-A Mean F1: {metrics_a['mean_f1']:.4f} | Recommended Threshold: {metrics_a['recommended_operational_threshold']:.4f}")
    print(f"  Model-B Mean F1: {metrics_b['mean_f1']:.4f} | Recommended Threshold: {metrics_b['recommended_operational_threshold']:.4f}")

if __name__ == "__main__":
    main()