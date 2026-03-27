"""
data_ingestion.py
-----------------
Fetches all World Bank indicators.

For each indicator, both raw yearly values and 6-year means are produced.
Outputs:
    data/raw/<indicator>_raw.csv          one file per indicator
    data/raw/all_indicators_raw.csv       combined multi-index file
    data/raw/data_quality_metadata.csv    completeness report
    data/processed/indicators.csv         6-year means (input to preprocessing)
    data/processed/indicators_metadata.json
"""

import json
import yaml
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import wbgapi as wb
from tenacity import retry, stop_after_attempt, wait_exponential


# Configuration 

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load project configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_all_country_codes(config: dict) -> List[str]:
    """Flatten all regional country codes into a single list."""
    return [
        code
        for region_countries in config["countries"].values()
        for code in region_countries
    ]


#  World Bank API 

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def _fetch_from_api(
    indicator: str,
    countries: List[str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Call the World Bank API with automatic retry on failure.

    Retries up to 3 times with exponential back-off (4–10 s) to handle
    transient network errors or API rate limits.
    """
    return wb.data.DataFrame(
        indicator,
        economy=countries,
        time=range(start_year, end_year + 1),
        skipBlanks=True,
        labels=False,
    )


def fetch_indicator(
    wb_code: str,
    indicator_name: str,
    countries: List[str],
    start_year: int,
    end_year: int,
    min_obs: int = 3,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Fetch one World Bank indicator and return its 6-year mean and raw yearly data.

    Countries with fewer than `min_obs` non-null observations across the
    period are marked NaN in the mean series — they will be imputed later
    in preprocessing.py rather than silently averaged over sparse data.

    Args:
        wb_code:        World Bank indicator code (e.g. 'NY.GDP.PCAP.KD.ZG')
        indicator_name: Internal column name used throughout the pipeline
        countries:      ISO-3 country codes
        start_year:     First year of the fetch window
        end_year:       Last year of the fetch window
        min_obs:        Minimum non-null observations required to compute a mean

    Returns:
        mean_series:  pd.Series  — one value per country (6-year mean)
        raw_df:       pd.DataFrame — one column per year, one row per country
    """
    print(f"  Fetching {indicator_name} ({wb_code}) ...", end=" ")

    try:
        raw_df = _fetch_from_api(wb_code, countries, start_year, end_year)

        # Normalise year column names: 'YR2018' → 2018
        raw_df.columns     = [int(str(c).replace("YR", "")) for c in raw_df.columns]
        raw_df.index.name  = "country_code"

        # Compute mean; set to NaN where observations are too sparse
        obs_count  = raw_df.count(axis=1)
        mean_series = raw_df.mean(axis=1)
        mean_series[obs_count < min_obs] = float("nan")
        mean_series.name = indicator_name

        print(f"✓  ({raw_df.shape[1]} years, "
              f"{obs_count.ge(min_obs).sum()}/{len(countries)} countries complete)")
        return mean_series, raw_df

    except Exception as exc:
        print(f"✗  {exc}")
        # Return empty placeholders so the pipeline can continue and
        # report the gap rather than crash mid-run.
        empty_mean = pd.Series(float("nan"), index=countries, name=indicator_name)
        empty_raw  = pd.DataFrame(
            index=countries, columns=range(start_year, end_year + 1)
        )
        return empty_mean, empty_raw


#  Main ingestion 

def fetch_all_indicators(
    config: dict,
    export_raw: bool = True,
    raw_data_dir: str = "data/raw",
) -> pd.DataFrame:
    """
    Iterate over every indicator in config, fetch data, and return 6-year means.

    Args:
        config:       Loaded project configuration
        export_raw:   Write per-indicator CSVs and a combined file to disk
        raw_data_dir: Directory for raw yearly data files

    Returns:
        pd.DataFrame: rows = countries, columns = indicators (6-year means)
    """
    countries  = get_all_country_codes(config)
    start_year = config["data"]["year_range"]["start"]
    end_year   = config["data"]["year_range"]["end"]

    print(f"\n{'=' * 60}")
    print(f"DATA INGESTION  ·  {start_year}–{end_year}  ·  {len(countries)} countries")
    print(f"{'=' * 60}")

    means_list      = []   
    raw_data        = {}   
    quality_records = []   

    for dimension, indicators in config["indicators"].items():
        print(f"\n  {dimension.upper().replace('_', ' ')}")

        for indicator_name, props in indicators.items():
            wb_code = props.get("wb_indicator")

            # Skip indicators without a World Bank code (e.g. manual entries)
            if not wb_code or props.get("source") == "manual_research":
                continue

            mean_series, raw_df = fetch_indicator(
                wb_code, indicator_name, countries,
                start_year, end_year, min_obs=3,
            )

            means_list.append(mean_series)

            if export_raw:
                raw_data[indicator_name] = raw_df

            # Record completeness for the quality report
            obs_count = raw_df.count(axis=1)
            n_years   = raw_df.shape[1]
            quality_records.append({
                "indicator":          indicator_name,
                "dimension":          dimension,
                "wb_code":            wb_code,
                "period":             f"{start_year}–{end_year}",
                "avg_completeness_%": round((obs_count / n_years * 100).mean(), 1),
                "countries_complete": int((obs_count == n_years).sum()),
                "countries_partial":  int(((obs_count > 0) & (obs_count < n_years)).sum()),
                "countries_missing":  int((obs_count == 0).sum()),
            })

    #  Export raw data 
    if export_raw and raw_data:
        raw_dir = Path(raw_data_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)

        for name, df in raw_data.items():
            df.to_csv(raw_dir / f"{name}_raw.csv")

        # Combined multi-index file (indicator × year)
        combined = pd.concat(raw_data.values(), keys=raw_data.keys(), axis=1)
        combined.columns.names = ["indicator", "year"]
        combined.to_csv(raw_dir / "all_indicators_raw.csv")

        # Data quality report
        if quality_records:
            pd.DataFrame(quality_records).to_csv(
                raw_dir / "data_quality_metadata.csv", index=False
            )

        print(f"\n  Raw data written to {raw_data_dir}/")

    #  Assemble means DataFrame 
    df = pd.concat(means_list, axis=1)
    df.index.name = "country_code"

    #  Quality summary 
    if quality_records:
        qdf = pd.DataFrame(quality_records)
        print(f"\n{'=' * 60}")
        print("DATA QUALITY SUMMARY")
        print(f"{'=' * 60}")
        for dim, grp in qdf.groupby("dimension"):
            print(f"  {dim:30s} {grp['avg_completeness_%'].mean():.1f}% complete")
        print(f"\n  Overall avg completeness : "
              f"{qdf['avg_completeness_%'].mean():.1f}%")
        print(f"  Fully complete indicators: "
              f"{(qdf['countries_missing'] == 0).sum()} / {len(qdf)}")

    print(f"\n{'=' * 60}")
    print(f"INGESTION COMPLETE")
    print(f"  {len(df)} countries × {len(df.columns)} indicators (6-year means)")
    print(f"  Next: python preprocessing.py")
    print(f"{'=' * 60}\n")

    return df


#  Export 

def export_processed_data(
    df: pd.DataFrame,
    output_path: str = "data/processed/indicators.csv",
) -> None:
    """
    Write 6-year means to CSV and save a JSON metadata sidecar.

    The metadata file is read by preprocessing.py to log the data period
    and aggregation method in the preprocessing output.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)

    metadata = {
        "ingestion_timestamp": datetime.now().isoformat(),
        "years":               "see config",
        "aggregation":         "6-year mean",
        "n_countries":         len(df),
        "n_indicators":        len(df.columns),
        "indicators":          list(df.columns),
    }
    metadata_path = path.parent / "indicators_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"✓ Means saved    → {path}")
    print(f"✓ Metadata saved → {metadata_path}")


#  Validation (optional QA step) 

def validate_means_against_raw(
    df_means: pd.DataFrame,
    indicator_name: str,
    raw_data_dir: str = "data/raw",
) -> bool:
    """
    Verify that the stored 6-year mean matches the mean recomputed from raw data.

    Useful as a spot-check after ingestion to catch any column-alignment
    or index mismatch issues introduced during concatenation.
    """
    filepath = Path(raw_data_dir) / f"{indicator_name}_raw.csv"
    if not filepath.exists():
        print(f"  No raw file for {indicator_name} — skipping validation")
        return False

    raw_df     = pd.read_csv(filepath, index_col=0)
    recalc     = raw_df.mean(axis=1)
    stored     = df_means.get(indicator_name)

    if stored is None:
        print(f"  {indicator_name} not found in means DataFrame")
        return False

    common     = recalc.index.intersection(stored.index)
    delta      = (recalc.loc[common] - stored.loc[common]).abs()
    passed     = (delta < 1e-10).all()

    if passed:
        print(f"  ✓ {indicator_name} validation passed")
    else:
        mismatches = common[delta >= 1e-10].tolist()
        print(f"  ✗ {indicator_name} mismatches: {mismatches}")

    return passed


#  Run 

if __name__ == "__main__":
    config = load_config()

    df = fetch_all_indicators(config, export_raw=True)
    export_processed_data(df)

    # Spot-check the first successfully fetched indicator
    first_col = next((c for c in df.columns if df[c].notna().any()), None)
    if first_col:
        print(f"\nValidating {first_col} ...")
        validate_means_against_raw(df, first_col)