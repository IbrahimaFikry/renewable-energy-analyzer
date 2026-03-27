"""
scoring.py
----------
Computes weighted market attractiveness scores from normalised 0–100 data.

Two outputs:
    1. score_single_scenario()    — scores under one weight scenario
    2. run_sensitivity_analysis() — scores under all four scenarios,
       testing whether recommendations hold across different assumptions
"""

import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


#  Configuration 

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load project configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_dimension_indicators(config: dict) -> Dict[str, List[str]]:
    """Return {dimension: [indicator_names]} for all dimensions in config."""
    return {
        dim: list(indicators.keys())
        for dim, indicators in config["indicators"].items()
    }


#  Data loading 

def load_normalized_data(
    data_path: str = "data/processed/normalized_indicators.csv",
) -> pd.DataFrame:
    """
    Load normalised 0–100 indicator data produced by preprocessing.py.

    Raises FileNotFoundError if the file is absent, prompting the user
    to run preprocessing.py first.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Normalised data not found at {path}. Run preprocessing.py first."
        )

    df = pd.read_csv(path, index_col=0)
    print(f"✓ Loaded {len(df)} countries × {len(df.columns)} indicators")

    if "region" not in df.columns:
        print("  ⚠ No 'region' column — regional breakdowns will be unavailable")

    return df


#  Dimension scoring 

def compute_dimension_score(
    df: pd.DataFrame,
    dimension: str,
    indicator_config: dict,
    indicator_names: List[str],
) -> pd.Series:
    """
    Compute one dimension score as a weighted average of its indicators.

    If some indicators are missing from the data, their weights are
    redistributed proportionally across the remaining indicators so the
    dimension score remains on a 0–100 scale.

    Args:
        df:               Normalised indicator data
        dimension:        Dimension key (e.g. 'market_opportunity')
        indicator_config: Full config['indicators'] dict
        indicator_names:  Ordered list of indicator names for this dimension

    Returns:
        pd.Series: Dimension scores (0–100), indexed by country_code
    """
    weighted = pd.Series(0.0, index=df.index)
    total_weight = 0.0

    for name in indicator_names:
        props = indicator_config[dimension].get(name)
        if props is None:
            print(f"  ⚠ '{name}' not in config — skipping")
            continue

        if name not in df.columns:
            print(f"  ⚠ '{name}' not in data — skipping")
            continue

        if df[name].isna().all():
            print(f"  ⚠ '{name}' is entirely NaN — skipping")
            continue

        weight        = props["weight"]
        weighted     += df[name] * weight
        total_weight += weight

    if total_weight == 0:
        print(f"  ✗ All indicators missing for {dimension}")
        return pd.Series(float("nan"), index=df.index)

    # Redistribute weights if any indicators were skipped
    if total_weight < sum(
        indicator_config[dimension][n]["weight"]
        for n in indicator_names
        if n in indicator_config[dimension]
    ):
        used = sum(
            1 for n in indicator_names
            if n in df.columns and not df[n].isna().all()
        )
        print(f"  {dimension}: {used}/{len(indicator_names)} indicators "
              f"(weights redistributed)")

    return (weighted / total_weight).round(2)


#  Single-scenario scoring 

def score_single_scenario(
    df: pd.DataFrame,
    config: dict,
    scenario: str = "balanced",
) -> pd.DataFrame:
    """
    Score all markets under one weight scenario.

    Returns a DataFrame with:
        - score_<dimension>   one column per dimension (0–100)
        - total_score         weighted sum of dimension scores (0–100)
        - rank                1 = most attractive market
        - region              copied from df if present

    Args:
        df:       Normalised indicator data
        config:   Project configuration
        scenario: Key in config['weight_scenarios']

    Returns:
        pd.DataFrame sorted by rank ascending
    """
    print(f"\n[Scoring] Scenario: {scenario}")

    dim_weights   = config["weight_scenarios"][scenario]
    ind_config    = config["indicators"]
    dim_map       = get_dimension_indicators(config)

    scores = pd.DataFrame(index=df.index)

    if "region" in df.columns:
        scores["region"] = df["region"]

    # Dimension scores
    for dim, indicators in dim_map.items():
        scores[f"score_{dim}"] = compute_dimension_score(
            df, dim, ind_config, indicators
        )

    # Weighted total
    scores["total_score"] = 0.0
    total_weight_used     = 0.0

    for dim, weight in dim_weights.items():
        col = f"score_{dim}"
        if col in scores.columns:
            scores["total_score"] += scores[col] * weight
            total_weight_used     += weight

    # Renormalise if any dimension was unavailable
    if 0 < total_weight_used < 1.0:
        scores["total_score"] /= total_weight_used
        print(f"  Note: Renormalised — used {total_weight_used:.2f} of total weight")

    scores["total_score"] = scores["total_score"].round(2)
    scores["rank"]        = (
        scores["total_score"]
        .rank(ascending=False, method="min", na_option="bottom")
        .astype("Int64")
    )

    scores = scores.sort_values("rank")
    print(f"  Top 3: {scores.index[:3].tolist()}")
    return scores


#  Sensitivity analysis 

def classify_stability(std: float) -> str:
    """Map rank standard deviation to a human-readable stability label."""
    if std <= 1.5:
        return "Very High"
    if std <= 3.0:
        return "High"
    if std <= 5.0:
        return "Medium"
    return "Low"


def run_sensitivity_analysis(
    df: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """
    Score all markets under every defined weight scenario and measure rank stability.

    A low rank standard deviation means the country's position is robust to
    changes in investor preferences — a strong signal for capital allocation.

    Returns a DataFrame with:
        - score_<scenario>     total score per scenario
        - rank_<scenario>      rank per scenario
        - avg_rank             mean rank across scenarios
        - rank_std             standard deviation of ranks (stability proxy)
        - rank_min / rank_max  best and worst rank observed
        - rank_range           max − min
        - stability            Very High / High / Medium / Low
        - region               copied from df if present

    Returns:
        pd.DataFrame sorted by avg_rank ascending
    """
    scenarios = list(config["weight_scenarios"].keys())

    print(f"\n{'=' * 60}")
    print(f"SENSITIVITY ANALYSIS  ·  {len(scenarios)} scenarios: {scenarios}")
    print(f"{'=' * 60}")

    score_cols = {}
    rank_cols  = {}

    for scenario in scenarios:
        result                  = score_single_scenario(df, config, scenario)
        score_cols[scenario]    = result["total_score"]
        rank_cols[scenario]     = result["rank"]

    # Assemble comparison table
    comparison = pd.concat(
        {f"score_{s}": score_cols[s] for s in scenarios}, axis=1
    )
    for s in scenarios:
        comparison[f"rank_{s}"] = rank_cols[s]

    rc = [f"rank_{s}" for s in scenarios]
    comparison["avg_rank"]   = comparison[rc].mean(axis=1).round(1)
    comparison["rank_std"]   = comparison[rc].std(axis=1).round(1)
    comparison["rank_min"]   = comparison[rc].min(axis=1)
    comparison["rank_max"]   = comparison[rc].max(axis=1)
    comparison["rank_range"] = comparison["rank_max"] - comparison["rank_min"]
    comparison["stability"]  = comparison["rank_std"].apply(classify_stability)

    if "region" in df.columns:
        comparison["region"] = df["region"]

    comparison = comparison.sort_values("avg_rank")

    # Summary
    print(f"\n[Stability Summary]")
    for label, count in comparison["stability"].value_counts().items():
        print(f"  {label}: {count} ({count / len(comparison) * 100:.0f}%)")

    print(f"\n[Top Stable Markets]")
    stable = comparison[comparison["stability"].isin(["Very High", "High"])].head(5)
    for code, row in stable.iterrows():
        print(f"  {code}: avg rank {row['avg_rank']:.1f}  "
              f"range {row['rank_range']:.0f}  {row['stability']}")

    return comparison


#  Export 

def export_scores(
    df: pd.DataFrame,
    output_path: str = "outputs/market_scores.csv",
) -> None:
    """Write single-scenario scores to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    print(f"✓ Scores saved → {path}")


def export_sensitivity(
    df: pd.DataFrame,
    output_path: str = "outputs/sensitivity_analysis.csv",
) -> None:
    """Write sensitivity analysis results to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    print(f"✓ Sensitivity saved → {path}")


#  Orchestrator 

def run_scoring(
    processed_df: Optional[pd.DataFrame] = None,
    data_path: str = "data/processed/normalized_indicators.csv",
    config_path: str = "config/config.yaml",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full scoring pipeline and return both outputs.

    Args:
        processed_df: Pre-loaded normalised DataFrame (skips disk read if provided)
        data_path:    Path to normalised indicators CSV
        config_path:  Path to project configuration YAML

    Returns:
        (primary_scores, sensitivity_df)
            primary_scores:  Balanced-scenario scores, sorted by rank
            sensitivity_df:  Cross-scenario comparison with stability metrics
    """
    print(f"\n{'=' * 60}")
    print("SCORING  ·  normalised data → market attractiveness scores")
    print(f"{'=' * 60}")

    config = load_config(config_path)

    if processed_df is None:
        processed_df = load_normalized_data(data_path)

    primary_scores = score_single_scenario(processed_df, config, scenario="balanced")

    print("\nTop 5 markets (balanced scenario):")
    for code, row in primary_scores.head(5).iterrows():
        region = row.get("region", "—")
        print(f"  {code} ({region}): {row['total_score']:.1f}")

    sensitivity_df = run_sensitivity_analysis(processed_df, config)

    export_scores(primary_scores)
    export_sensitivity(sensitivity_df)

    print(f"\n{'=' * 60}")
    print("SCORING COMPLETE  ·  Next: python clustering.py")
    print(f"{'=' * 60}\n")

    return primary_scores, sensitivity_df


#  Run  

if __name__ == "__main__":
    primary_scores, sensitivity_df = run_scoring()

    # Dimension score columns for the display table
    dim_cols = [c for c in primary_scores.columns
                if c.startswith("score_") and c != "score_total"]
    display_cols = ["region", "total_score", "rank"] + dim_cols
    display_cols = [c for c in display_cols if c in primary_scores.columns]

    print("\nTop 10 markets — balanced scenario:")
    print(primary_scores[display_cols].head(10).to_string())

    print("\nStability breakdown:")
    for label, count in sensitivity_df["stability"].value_counts().items():
        print(f"  {label}: {count} ({count / len(sensitivity_df) * 100:.0f}%)")