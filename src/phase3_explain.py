import json
import pickle
import warnings
import numpy as np
import pandas as pd
import shap
from config import ARTIFACT_DIR, DOMINANT_FLAG_COL, TARGET_COL

warnings.filterwarnings("ignore")

def load_pipeline_states():
    with open(ARTIFACT_DIR / "model_b_xgb.pkl", "rb") as fh:
        model = pickle.load(fh)
    df_parquet = pd.read_parquet(ARTIFACT_DIR / "engineered_dataset.parquet")
    X = df_parquet.drop(columns=[TARGET_COL])
    if DOMINANT_FLAG_COL in X.columns:
        X = X.drop(columns=[DOMINANT_FLAG_COL])
    y = df_parquet[TARGET_COL].astype(int)
    oof_probs = np.load(ARTIFACT_DIR / "model_b_oof_probs.npy")
    graph_df = pd.read_csv(ARTIFACT_DIR / "network_structure_metrics.csv")
    return model, X, y, oof_probs, graph_df

def main():
    print("[START] Executing RegShield AI Pipeline — Phase 3 (Explain Engine)")
    model, X, y, oof_probs, graph_df = load_pipeline_states()
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    feature_names = list(X.columns)
    top_fraud_indices = np.argsort(oof_probs)[::-1]
    
    case_profiles = {}
    for rank_idx, idx in enumerate(top_fraud_indices[:15]):
        account_idx = int(idx)
        raw_probs = float(oof_probs[account_idx])
        true_label = int(y.iloc[account_idx])
        
        account_shap = shap_values[account_idx]
        top_shap_features = np.argsort(np.abs(account_shap))[::-1][:3]
        
        reasons = []
        for f_idx in top_shap_features:
            f_name = feature_names[f_idx]
            f_val = float(X.iloc[account_idx, f_idx])
            shap_impact = float(account_shap[f_idx])
            direction = "elevated" if shap_impact > 0 else "suppressed"
            reasons.append({
                "feature": f_name,
                "value": round(f_val, 4),
                "shap_impact": round(shap_impact, 4),
                "influence": direction
            })
            
        graph_row = graph_df.loc[graph_df["account_idx"] == account_idx].iloc[0]
        
        case_profiles[account_idx] = {
            "investigator_rank": rank_idx + 1,
            "anomaly_probability_pct": round(raw_probs * 100, 2),
            "ground_truth_mule_status": true_label,
            "network_context": {
                "cluster_id": int(graph_row["network_cluster_id"]),
                "peer_connection_degree": int(graph_row["network_degree"]),
                "neighborhood_density_coefficient": round(float(graph_row["network_clustering_coefficient"]), 4)
            },
            "top_behavioral_triggers": reasons
        }
        
    output_path = ARTIFACT_DIR / "investigator_case_profiles.json"
    with open(output_path, "w") as fh:
        json.dump(case_profiles, fh, indent=2)
    print(f"[COMPLETE] Explainable logic profiles compiled and serialized to: {output_path}")

if __name__ == "__main__":
    main()