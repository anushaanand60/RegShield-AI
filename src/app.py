import json
import pandas as pd
import streamlit as st
from config import ARTIFACT_DIR

def load_case_data():
    with open(ARTIFACT_DIR / "investigator_case_profiles.json", "r") as fh:
        return json.load(fh)

def main():
    st.set_page_config(
        page_title="RegShield AI Operational Dashboard", 
        layout="wide", 
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        .reportview-container .main .block-container { max-width: 95%; }
        body { color: #000000 !important; background-color: #ffffff !important; }
        h1, h2, h3, h4, p, span, label, div { color: #000000 !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
        .stMetric { background-color: #f1f5f9 !important; border-radius: 8px; padding: 15px; border: 2px solid #cbd5e1 !important; }
        .stMetric div { color: #000000 !important; }
        div[data-testid="stExpander"] { border: 2px solid #cbd5e1 !important; border-radius: 8px; background-color: #f8fafc !important; }
        .stDataFrame { border: 1px solid #cbd5e1 !important; }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("RegShield AI | Financial Intelligence Platform")
    st.markdown("**Operational Interface: Network Anomaly Tracking and Forensic Validation**")
    st.markdown("---")
    
    cases = load_case_data()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Model A Score (Flag Included)", value="0.9754", delta="Baseline Stable")
    with col2:
        st.metric(label="Model B Score (Ablated Behavioral Engine)", value="0.9099", delta="Production Grade")
    with col3:
        st.metric(label="Total Escalated Targets In Queue", value=f"{len(cases)} Accounts", delta_color="off")
        
    st.sidebar.markdown("### Forensic Controller")
    selected_account = st.sidebar.selectbox(
        "Select Active Account Target for Investigation", 
        list(cases.keys())
    )
    
    if selected_account:
        profile = cases[selected_account]
        net_ctx = profile["network_context"]
        
        st.markdown(f"## Active Investigation File: Account Reference {selected_account}")
        
        meta_col1, meta_col2, meta_col3 = st.columns(3)
        with meta_col1:
            st.markdown(f"**Risk Order Rank:** Segment Cluster {profile['investigator_rank']}")
        with meta_col2:
            st.markdown(f"**Anomaly Confidence Threshold:** {profile['anomaly_probability_pct']}%")
        with meta_col3:
            status_text = "CONFIRMED FRAUD SIGNATURE" if profile["ground_truth_mule_status"] == 1 else "UNVERIFIED PROFILE"
            st.markdown(f"**Ground Truth Validation:** {status_text}")
            
        st.markdown("---")
        
        left_panel, right_panel = st.columns([1, 2])
        
        with left_panel:
            st.markdown("### Spatial Topological Structure")
            st.markdown(f"**Assigned Linkage Cluster ID:** {net_ctx['cluster_id']}")
            st.markdown(f"**Co Located Peer Connection Degree:** {net_ctx['peer_connection_degree']}")
            st.markdown(f"**Local Clustering Coefficient:** {net_ctx['neighborhood_density_coefficient']}")
            st.markdown("")
                
            st.markdown("### Defensive Action Directive")
            compliance_text = (
                f"ACTION DISPATCH MANDATE: Target account matches specific operational signature "
                f"localized inside cluster network {net_ctx['cluster_id']}. Behavioral extraction matrices "
                f"indicate highly synchronized risk attributes with {profile['anomaly_probability_pct']}% validation vector accuracy. "
                f"Dynamic transaction restrictions initialized under regulatory framework requirements."
            )
            st.warning(compliance_text)
            
        with right_panel:
            st.markdown("### AI Attribution Explanations: SHAP Feature Vectors")
            triggers_df = pd.DataFrame(profile["top_behavioral_triggers"])
            triggers_df.columns = ["Target Column Descriptor", "Observed Feature Value", "SHAP Weight Influence", "Impact Direction"]
            st.dataframe(triggers_df, use_container_width=True, hide_index=True)
            
        st.markdown("---")
        
    st.markdown("### Master Risk Priority Queue")
    case_list = []
    for k, v in cases.items():
        case_list.append({
            "Target Account": k,
            "Queue Rank": v["investigator_rank"],
            "Anomaly Vector Metric": f"{v['anomaly_probability_pct']}%",
            "Mule Code Status": "CONFIRMED MULE" if v["ground_truth_mule_status"] == 1 else "LEGITIMATE COMPLIANT",
            "Network Topological ID": v["network_context"]["cluster_id"],
            "Active Shared Links": v["network_context"]["peer_connection_degree"]
        })
    st.dataframe(pd.DataFrame(case_list), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()