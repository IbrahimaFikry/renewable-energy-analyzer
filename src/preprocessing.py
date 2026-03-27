"""
preprocessing.py
----------------
Cleans and normalises raw indicator data before scoring.

Pipeline:
    1. Load 6-year means from data/processed/indicators.csv
    2. Add region labels
    3. Impute missing values (regional median, global fallback)
    4. Winsorise outliers (5th / 95th percentile)
    5. Flip direction of lower_is_better indicators
    6. Min-max normalise all indicators to 0–100
    7. Validate and export
"""

import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List


#  Configuration 

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load project configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_indicator_directions(config: dict) -> Dict[str, str]:
    """Return {indicator_name: direction} for all indicators in config."""
    return {
        name: props["direction"]
        for dim in config["indicators"].values()
        for name, props in dim.items()
    }


def get_all_indicator_names(config: dict) -> List[str]:
    """Return a flat list of all indicator column names."""
    return [
        name
        for dim in config["indicators"].values()
        for name in dim
    ]


#  Data loading 

def load_processed_data(
    data_path: str = "data/processed/indicators.csv",
) -> pd.DataFrame:
    """
    Load 6-year mean data produced by data_ingestion.py.

    Raises FileNotFoundError if the file does not exist,
    prompting the user to run data_ingestion.py first.
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}. Run data_ingestion.py first."
        )

    df = pd.read_csv(path, index_col=0)
    print(f"✓ Loaded {len(df)} countries × {len(df.columns)} indicators")

    # Log data period from accompanying metadata if available
    metadata_path = path.parent / "indicators_metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        print(f"  Period: {meta.get('years', 'N/A')}  |  "
              f"Aggregation: {meta.get('aggregation', 'N/A')}")

    return df


#  Region labelling 

def add_region_labels(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Add a 'region' column derived from config country lists.
    Skips if the column already exists.
    """
    if "region" in df.columns:
        return df

    region_map = {
        code: region
        for region, codes in config["countries"].items()
        for code in codes
    }

    df = df.copy()
    df["region"] = df.index.map(region_map)

    missing = df[df["region"].isna()].index.tolist()
    if missing:
        print(f"  ⚠ No region mapping for: {missing}")

    return df


#  Missing value handling 

def report_missing_values(df: pd.DataFrame, indicator_cols: List[str]) -> None:
    """Print a concise summary of missing values per indicator."""
    print("\n[Missing Values]")
    found = False
    for col in indicator_cols:
        if col not in df.columns:
            continue
        n = df[col].isna().sum()
        if n > 0:
            print(f"  {col}: {n} missing ({n / len(df) * 100:.1f}%)")
            found = True
    if not found:
        print("  ✓ No missing values")


def impute_missing_values(
    df: pd.DataFrame, indicator_cols: List[str]
) -> pd.DataFrame:
    """
    Fill missing values using regional median with a global median fallback.

    Regional imputation is preferred because countries in the same region
    (e.g. Southeast Asia) share similar energy infrastructure and
    economic development patterns, making peers more informative than
    the global sample for energy and emissions indicators.
    """
    df = df.copy()

    for col in indicator_cols:
        if col not in df.columns or df[col].isna().sum() == 0:
            continue

        n_before = df[col].isna().sum()

        # Primary: regional median
        df[col] = df[col].fillna(df.groupby("region")[col].transform("median"))

        # Fallback: global median (handles cases where the full region is missing)
        n_after_regional = df[col].isna().sum()
        df[col] = df[col].fillna(df[col].median())

        print(f"  {col}: {n_before} missing → "
              f"{n_after_regional} after regional → 0 after global fallback")

    return df


#  Outlier handling 

def winsorize(
    df: pd.DataFrame,
    indicator_cols: List[str],
    lower: float = 0.05,
    upper: float = 0.95,
) -> pd.DataFrame:
    """
    Cap extreme values at the lower and upper percentiles.

    Winsorising at the 5th / 95th percentile prevents outliers — such as
    oil-rich economies with extreme energy import values or small island
    states with atypical emissions — from compressing variation for the
    majority of countries in the 0–100 normalised scale.
    """
    df = df.copy()

    for col in indicator_cols:
        if col not in df.columns:
            continue

        lo, hi   = df[col].quantile(lower), df[col].quantile(upper)
        n_low    = (df[col] < lo).sum()
        n_high   = (df[col] > hi).sum()

        df[col]  = df[col].clip(lo, hi)

        if n_low > 0 or n_high > 0:
            print(f"  {col}: capped {n_low} below p{int(lower*100)}, "
                  f"{n_high} above p{int(upper*100)}")

    return df


#  Direction normalisation 

def flip_lower_is_better(
    df: pd.DataFrame, directions: Dict[str, str]
) -> pd.DataFrame:
    """
    Invert indicators where a lower raw value signals greater opportunity.

    xamples:
      - modern_renewable_share:   low current modern renewables = high growth runway
      - electricity_imports:      low imports = less grid dependency (higher energy security)

    Multiplying by -1 ensures all indicators point the same direction
    (higher normalised score = better opportunity) before aggregation.
    """
    df     = df.copy()
    flipped = [
        col for col, direction in directions.items()
        if col in df.columns and direction == "lower_is_better"
    ]

    for col in flipped:
        df[col] *= -1

    if flipped:
        print(f"  Inverted {len(flipped)} indicator(s): {flipped}")

    return df


#  Normalisation 

def min_max_normalize(
    df: pd.DataFrame, indicator_cols: List[str]
) -> pd.DataFrame:
    """
    Scale all indicators to a 0–100 range.

    Formula: score = (value - min) / (max - min) × 100

    100 = best-performing country in the sample
    0   = worst-performing country in the sample
    50  = midpoint

    This puts GDP growth (%), CO2 emissions (t/capita), and governance
    indices (–2.5 to +2.5) on a common footing for weighted aggregation.
    If all countries have the same value, the indicator is set to 50.
    """
    df = df.copy()

    for col in indicator_cols:
        if col not in df.columns:
            continue

        lo, hi = df[col].min(), df[col].max()

        if lo == hi:
            
            df[col] = 50.0
        else:
            df[col] = ((df[col] - lo) / (hi - lo) * 100).round(2)

    return df


#  Validation 

def validate_normalization(
    df: pd.DataFrame, indicator_cols: List[str]
) -> bool:
    """
    Assert all indicators are within [0, 100] after normalisation.
    Returns True if all pass; logs failures otherwise.
    """
    failures = []

    for col in indicator_cols:
        if col not in df.columns:
            continue
        lo, hi = df[col].min(), df[col].max()
        if lo < -1e-10 or hi > 100 + 1e-10:
            failures.append(f"  ✗ {col}: [{lo:.2f}, {hi:.2f}] — out of bounds")

    if failures:
        print("\n[Validation — FAILED]")
        for f in failures:
            print(f)
        return False

    print("  ✓ All indicators within [0, 100]")
    return True


#  Export 

def export_normalized_data(
    df: pd.DataFrame,
    output_path: str = "data/processed/normalized_indicators.csv",
) -> None:
    """Write normalised data to CSV, creating parent directories as needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    print(f"✓ Saved normalised data → {path}")


#  Orchestrator 

def run_preprocessing(
    input_path:  str = "data/processed/indicators.csv",
    output_path: str = "data/processed/normalized_indicators.csv",
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline and return a normalised DataFrame.

    Args:
        input_path:  Path to 6-year mean data from data_ingestion.py
        output_path: Destination for normalised indicators CSV
        config_path: Path to project configuration YAML

    Returns:
        pd.DataFrame: All values in [0, 100], ready for scoring.py
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING  ·  6-year means → normalised scores")
    print("=" * 60)

    config          = load_config(config_path)
    indicator_cols  = get_all_indicator_names(config)
    directions      = get_indicator_directions(config)

    df = load_processed_data(input_path)
    df = add_region_labels(df, config)

    print(f"\n[Step 1] Missing values")
    report_missing_values(df, indicator_cols)

    print(f"\n[Step 2] Imputation (regional median → global fallback)")
    df = impute_missing_values(df, indicator_cols)

    print(f"\n[Step 3] Winsorisation (p5 / p95)")
    df = winsorize(df, indicator_cols)

    print(f"\n[Step 4] Direction alignment (invert lower_is_better)")
    df = flip_lower_is_better(df, directions)

    print(f"\n[Step 5] Min-max normalisation → [0, 100]")
    df = min_max_normalize(df, indicator_cols)

    print(f"\n[Step 6] Validation")
    validate_normalization(df, indicator_cols)

    export_normalized_data(df, output_path)

    print(f"\n{'=' * 60}")
    print(f"PREPROCESSING COMPLETE")
    print(f"  {len(df)} countries × {len(indicator_cols)} indicators → [0, 100]")
    print(f"  Next: python scoring.py")
    print(f"{'=' * 60}\n")

    return df


# Run

if __name__ == "__main__":
    df = run_preprocessing()

    # Spot-check: first 5 numeric columns, first 5 rows
    sample_cols = df.select_dtypes(include=[np.number]).columns[:5]
    print("Sample normalised values (first 5 countries):")
    print(df[sample_cols].head().round(1))

    print("\nAverage normalised scores by region:")
    print(df.groupby("region")[sample_cols].mean().round(1))