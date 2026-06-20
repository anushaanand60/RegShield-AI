import pandas as pd
import networkx as nx
import community as community_louvain
from config import ARTIFACT_DIR, RAW_DATA_PATH

def load_graph_context(path = RAW_DATA_PATH):
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    keep_cols = ["F3888", "F3890"]
    return df[keep_cols]

def main():
    print("[START] Executing RegShield AI Pipeline — Phase 2 (Graph Core)")
    context_df = load_graph_context()
    context_df["account_idx"] = context_df.index
    
    G = nx.Graph()
    G.add_nodes_from(context_df["account_idx"])
    
    grouped = context_df.groupby(["F3888", "F3890"])["account_idx"].apply(list)
    
    edges_to_add = []
    for account_list in grouped:
        n_accounts = len(account_list)
        if n_accounts > 1:
            for i in range(n_accounts):
                for j in range(i + 1, n_accounts):
                    edges_to_add.append((account_list[i], account_list[j]))
                    
    G.add_edges_from(edges_to_add)
    print(f"   -> Graph construction complete: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    
    partition = community_louvain.best_partition(G, random_state=42)
    degrees = dict(G.degree())
    clustering_coeff = nx.clustering(G)
    
    metrics_df = pd.DataFrame({
        "account_idx": list(G.nodes()),
        "network_cluster_id": [partition[node] for node in G.nodes()],
        "network_degree": [degrees[node] for node in G.nodes()],
        "network_clustering_coefficient": [clustering_coeff[node] for node in G.nodes()]
    })
    
    output_path = ARTIFACT_DIR / "network_structure_metrics.csv"
    metrics_df.to_csv(output_path, index=False)
    print(f"[COMPLETE] Network linkage signatures serialized to: {output_path}")

if __name__ == "__main__":
    main()