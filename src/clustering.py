"""
clustering.py
-------------
Groups countries into market archetypes using K-Means on dimension scores.

Clusters are labelled by average total score (descending), producing
business-meaningful archetypes such as Ready Markets and Watch & Wait.

Steps:
    1. Load dimension scores from scoring.py output
    2. Standardise features (K-Means is distance-based)
    3. Fit K-Means with k from config
    4. Validate with silhouette score
    5. Label clusters by average total score
    6. Profile each cluster
    7. Export results and optional visualisations
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


#  Configuration 

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load project configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


#  Data loading 

def load_scores_data(
    scores_path: str = "outputs/market_scores.csv",
) -> pd.DataFrame:
    """
    Load primary scores produced by scoring.py.

    Raises FileNotFoundError if the file is absent,
    prompting the user to run scoring.py first.
    """
    path = Path(scores_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Scores not found at {path}. Run scoring.py first."
        )

    df = pd.read_csv(path, index_col=0)
    dim_cols = _get_dim_cols(df)
    print(f"✓ Loaded {len(df)} countries  ·  dimensions: {dim_cols}")
    return df


def _get_dim_cols(df: pd.DataFrame) -> List[str]:
    """Return dimension score column names, excluding total and rank."""
    return [
        c for c in df.columns
        if c.startswith("score_") and "total" not in c and "rank" not in c
    ]


#  Feature preparation 

def prepare_features(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract dimension score columns as the clustering feature matrix.

    Dimension scores (rather than raw indicators) are used because they
    are already on a common 0–100 scale and capture the business-relevant
    groupings. Missing values are filled with column means before clustering.
    """
    dim_cols = _get_dim_cols(scores_df)

    if not dim_cols:
        raise ValueError(
            "No dimension score columns found. "
            "Expected columns like 'score_market_opportunity'. "
            "Run scoring.py first."
        )

    features = scores_df[dim_cols].copy()

    if features.isna().any().any():
        print("  ⚠ Missing values in features — filling with column means")
        features = features.fillna(features.mean())

    print(f"  Feature matrix: {features.shape[0]} × {features.shape[1]}")
    return features


#  Optional k-validation 

def validate_k(
    features_scaled: np.ndarray,
    max_k: int = 6,
) -> pd.DataFrame:
    """
    Compute silhouette score and inertia for k = 2 … max_k.

    Use this as a diagnostic check when the configured k is uncertain.
    The configured k is used for the final fit regardless of this output.
    """
    print("\n[k-validation]")
    records = []

    for k in range(2, max_k + 1):
        km        = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels    = km.fit_predict(features_scaled)
        sil       = silhouette_score(features_scaled, labels)
        records.append({"k": k, "silhouette": round(sil, 3), "inertia": round(km.inertia_)})
        print(f"  k={k}: silhouette={sil:.3f}  inertia={km.inertia_:.0f}")

    return pd.DataFrame(records)


#  Cluster labelling 

# Ordered labels assigned to clusters ranked by descending average total score.

CLUSTER_LABELS = [
    "Ready Markets",
    "Transition Markets",
    "Watch & Wait",
]


def label_clusters(df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """
    Label clusters by average total score rank.
    
    Because K-Means clusters on dimension profiles, a single outlier
    indicator (e.g. Senegal's high Energy Security score) can pull a
    country into a lower-scoring cluster despite a competitive total score.
    Re-ranking clusters by their mean total score after fitting corrects
    this without changing the clustering algorithm.
    """
    df = df.copy()

    # Rank clusters by mean total score — highest mean = best label
    cluster_means = (
        df.groupby('cluster_id')['total_score']
        .mean()
        .sort_values(ascending=False)
    )

    label_map = {
        cid: CLUSTER_LABELS[rank]
        for rank, cid in enumerate(cluster_means.index)
        if rank < len(CLUSTER_LABELS)
    }

    df['cluster_label'] = df['cluster_id'].map(label_map)

    # ── Sanity check: flag countries whose label contradicts their rank ──
    # If a country's total score is higher than the max of the cluster
    # below it, it may be mislabelled due to dimension profile outliers.
    # In that case, override based on score boundaries.
    boundaries = (
        df.groupby('cluster_label')['total_score']
        .agg(['min', 'max'])
        .reindex(CLUSTER_LABELS)
    )

    def resolve_label(row):
        score = row['total_score']
        for label in CLUSTER_LABELS:
            if label not in boundaries.index:
                continue
            lo = boundaries.loc[label, 'min']
            hi = boundaries.loc[label, 'max']
            if lo <= score <= hi:
                return label
        return row['cluster_label']   

    df['cluster_label'] = df.apply(resolve_label, axis=1)

    print('\n[Cluster Labels]')
    for label in CLUSTER_LABELS:
        subset = df[df['cluster_label'] == label]
        print(f'  {label}: {len(subset)} countries  '
              f'(score {subset["total_score"].min():.1f}–'
              f'{subset["total_score"].max():.1f})')

    return df

def apply_threshold_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Override cluster labels with score-based thresholds.
    
    This preserves the 3-tier structure (Ready / Transition / Watch & Wait)
    while using the cleaner indicator set. Thresholds are:
        - Ready Markets:   total_score >= 70
        - Watch & Wait:    total_score < 40
        - Transition:      40 <= total_score < 70
    """
    df = df.copy()
    
    df['cluster_label'] = 'Transition Markets'  # default
    
    df.loc[df['total_score'] >= 70, 'cluster_label'] = 'Ready Markets'
    df.loc[df['total_score'] < 40, 'cluster_label'] = 'Watch & Wait'
    
    print('\n[Threshold-based Labels]')
    for label in ['Ready Markets', 'Transition Markets', 'Watch & Wait']:
        subset = df[df['cluster_label'] == label]
        if len(subset) > 0:
            score_range = f"{subset['total_score'].min():.1f}–{subset['total_score'].max():.1f}"
            print(f'  {label}: {len(subset)} countries  (score {score_range})')
        else:
            print(f'  {label}: 0 countries')
    
    return df

#  Cluster profiling 

def profile_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean and std of dimension scores and total score per cluster.

    Returns a DataFrame indexed by cluster_label, sorted by
    total_score_mean descending.
    """
    dim_cols  = _get_dim_cols(df)
    agg_cols  = dim_cols + ["total_score"]

    profile   = df.groupby("cluster_label")[agg_cols].agg(["mean", "std"]).round(1)
    profile.columns = ["_".join(c) for c in profile.columns]
    profile["n_countries"] = df.groupby("cluster_label").size()

    return profile.sort_values("total_score_mean", ascending=False)


#  Visualisations (optional) 

def plot_cluster_radar(
    df: pd.DataFrame,
    output_path: str = "outputs/cluster_radar.png",
) -> None:
    """
    Radar chart of average dimension scores per cluster.
    Silently skipped if matplotlib is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping radar chart")
        return

    dim_cols    = _get_dim_cols(df)
    dim_labels  = [c.replace("score_", "").replace("_", " ").title() for c in dim_cols]
    means       = df.groupby("cluster_label")[dim_cols].mean()

    n    = len(dim_cols)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]   # close the polygon

    fig, ax = plt.subplots(figsize=(9, 7), subplot_kw={"projection": "polar"})

    for cluster in means.index:
        vals = means.loc[cluster].tolist() + [means.loc[cluster].iloc[0]]
        ax.plot(angles, vals, "o-", linewidth=2, label=cluster)
        ax.fill(angles, vals, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, size=10)
    ax.set_ylim(0, 100)
    ax.set_title("Cluster Profiles by Dimension", size=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Radar chart → {output_path}")


def plot_cluster_scatter(
    df: pd.DataFrame,
    output_path: str = "outputs/cluster_scatter.png",
) -> None:
    """
    Scatter plot of total_score vs rank, coloured by cluster.
    Annotates the top-3 and bottom-3 markets.
    Silently skipped if matplotlib is unavailable.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed — skipping scatter plot")
        return

    fig, ax = plt.subplots(figsize=(11, 7))

    for cluster in df["cluster_label"].unique():
        sub = df[df["cluster_label"] == cluster]
        ax.scatter(sub["rank"], sub["total_score"], label=cluster, s=90, alpha=0.75)

    # Annotate top-3 and bottom-3 by rank
    for code in df.head(3).index.tolist() + df.tail(3).index.tolist():
        ax.annotate(
            code,
            (df.loc[code, "rank"], df.loc[code, "total_score"]),
            xytext=(5, 5), textcoords="offset points", fontsize=8,
        )

    ax.invert_xaxis()   # rank 1 on the right
    ax.set_xlabel("Rank")
    ax.set_ylabel("Total Score (0–100)")
    ax.set_title("Market Clusters — Score vs Rank")
    ax.grid(True, alpha=0.25)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Scatter plot → {output_path}")


#  Export 

def export_clusters(
    df: pd.DataFrame,
    output_path: str = "outputs/market_clusters.csv",
) -> None:
    """Write the full clustered DataFrame (all dimension columns intact) to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    print(f"✓ Cluster assignments → {path}  ({len(df)} countries)")


def export_cluster_profiles(
    profile: pd.DataFrame,
    output_path: str = "outputs/cluster_profiles.csv",
) -> None:
    """Write cluster mean/std profiles to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile.to_csv(path)
    print(f"✓ Cluster profiles    → {path}")


#  Orchestrator 

def run_clustering(
    scores_df: Optional[pd.DataFrame] = None,
    scores_path: str = "outputs/market_scores.csv",
    config_path: str = "config/config.yaml",
    run_k_validation: bool = False,
) -> pd.DataFrame:
    """
    Run the full clustering pipeline and return the annotated DataFrame.
    """
    print(f"\n{'=' * 60}")
    print("CLUSTERING  ·  dimension scores → market archetypes")
    print(f"{'=' * 60}")

    config       = load_config(config_path)
    n_clusters   = config["clustering"]["n_clusters"]
    random_state = config["clustering"]["random_state"]

    if scores_df is None:
        scores_df = load_scores_data(scores_path)

    # Feature matrix
    print("\n[Step 1] Preparing features")
    features = prepare_features(scores_df)

    # Standardise — K-Means is sensitive to scale
    print("\n[Step 2] Standardising features")
    features_scaled = StandardScaler().fit_transform(features)

    # Optional k-sweep
    if run_k_validation:
        validate_k(features_scaled)

    # Fit K-Means
    print(f"\n[Step 3] K-Means  k={n_clusters}")
    km           = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clustered_df = scores_df.copy()
    clustered_df["cluster_id"] = km.fit_predict(features_scaled)

    # Silhouette validation
    sil = silhouette_score(features_scaled, clustered_df["cluster_id"])
    print(f"\n[Step 4] Silhouette score: {sil:.3f}", end="  ")
    if sil >= 0.50:
        print("(excellent separation)")
    elif sil >= 0.30:
        print("(good separation)")
    elif sil >= 0.20:
        print("(reasonable separation)")
    else:
        print("(weak separation — review dimensions or k)")

    # Label clusters using score thresholds
    print("\n[Step 5] Labelling clusters (threshold-based)")
    clustered_df = apply_threshold_labels(clustered_df)

    # Profile clusters
    print("\n[Step 6] Cluster profiles")
    profile = profile_clusters(clustered_df)
    dim_cols = _get_dim_cols(clustered_df)

    for label in profile.index:
        n     = int(profile.loc[label, "n_countries"])
        mean  = profile.loc[label, "total_score_mean"]
        std   = profile.loc[label, "total_score_std"]
        print(f"\n  {label}  (n={n}, score {mean:.1f} ± {std:.1f})")
        for dim in dim_cols:
            dm = profile.loc[label, f"{dim}_mean"]
            ds = profile.loc[label, f"{dim}_std"]
            print(f"    {dim.replace('score_','').replace('_',' '):30s} "
                  f"{dm:.1f} ± {ds:.1f}")

    # Country listing per cluster
    print("\n[Step 7] Countries by cluster")
    for label, grp in clustered_df.groupby("cluster_label"):
        grp = grp.sort_values("total_score", ascending=False)
        region_col = "region" if "region" in grp.columns else None
        print(f"\n  {label}  ({len(grp)} countries):")
        for code, row in grp.iterrows():
            region = f"  {row['region']}" if region_col else ""
            print(f"    {code}{region}: {row['total_score']:.1f}")

    # Export
    print("\n[Step 8] Exporting")
    export_clusters(clustered_df)
    export_cluster_profiles(profile)

    # Visualisations
    print("\n[Step 9] Visualisations")
    plot_cluster_radar(clustered_df)
    plot_cluster_scatter(clustered_df)

    print(f"\n{'=' * 60}")
    print("CLUSTERING COMPLETE")
    print(f"{'=' * 60}\n")

    return clustered_df


#  Run

if __name__ == "__main__":
    df = run_clustering(run_k_validation=True)

    display_cols = [c for c in ["region", "total_score", "rank", "cluster_label"]
                    if c in df.columns]
    print("\nTop 10 markets:")
    print(df[display_cols].head(15).to_string())

    print("\nCluster distribution:")
    for label, count in df["cluster_label"].value_counts().items():
        print(f"  {label}: {count} ({count / len(df) * 100:.0f}%)")