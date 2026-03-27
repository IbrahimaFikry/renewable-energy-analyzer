"""
main.py
-------
Pipeline orchestrator.

Runs all modules in sequence, then writes a plain-text executive summary.

Steps:
    1. Data Ingestion   — World Bank API fetch, 6-year means + raw export
    2. Preprocessing    — imputation, winsorisation, 0–100 normalisation
    3. Scoring          — weighted dimension scores + sensitivity analysis
    4. Clustering       — K-Means market archetypes

Usage:
    python main.py
"""

import sys
import time
import subprocess
import traceback
from pathlib import Path
from datetime import datetime

import pandas as pd


# ── Directory setup ────────────────────────────────────────────────────────

OUTPUT_DIRS = [
    "data/raw",
    "data/processed",
    "outputs",
    "outputs/charts",
]


def ensure_output_dirs() -> None:
    """Create all required output directories if they do not yet exist."""
    for d in OUTPUT_DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Output directories ready")


# ── Step runner ────────────────────────────────────────

def run_step(step_num: int, step_name: str, script_name: str) -> bool:
    """
    Execute one pipeline module as a subprocess.

    Scripts are expected in the src/ directory. stdout is streamed to the
    terminal; stderr is shown only when it contains an error.

    Returns True on success, False on failure.
    """
    script_path = Path("src") / script_name

    print(f"\n{'=' * 60}")
    print(f"STEP {step_num}  ·  {step_name}")
    print(f"{'=' * 60}")

    if not script_path.exists():
        print(f"  ✗ Script not found: {script_path}")
        return False

    t0 = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
        )

        if result.stdout:
            print(result.stdout)

        # Only surface stderr when there is a genuine error
        if result.stderr and "Error" in result.stderr:
            print(f"  Warnings / errors:\n{result.stderr}")

        elapsed = time.time() - t0

        if result.returncode == 0:
            print(f"  ✓ Completed in {elapsed:.1f} s")
            return True

        print(f"  ✗ Failed (exit code {result.returncode})")
        if result.stderr:
            print(result.stderr)
        return False

    except Exception as exc:
        print(f"  ✗ Could not execute {script_path}: {exc}")
        return False


# ── Executive summary ─────────────────────────────────────────────────

def generate_executive_summary() -> None:
    """
    Write a plain-text executive summary to outputs/executive_summary.txt.

    Reads the cluster and sensitivity CSV files produced by the pipeline.
    Skips gracefully if either file is absent.
    """
    print(f"\n{'=' * 60}")
    print("EXECUTIVE SUMMARY")
    print(f"{'=' * 60}")

    clusters_path    = Path("outputs/market_clusters.csv")
    sensitivity_path = Path("outputs/sensitivity_analysis.csv")

    if not clusters_path.exists() or not sensitivity_path.exists():
        print("  ⚠ Output files not found — skipping summary")
        return

    try:
        df  = pd.read_csv(clusters_path,    index_col="country_code")
        sens = pd.read_csv(sensitivity_path, index_col="country_code")

        #  Top 5 
        top5_lines = []
        for i, (code, row) in enumerate(df.head(5).iterrows(), 1):
            region  = row.get("region", "—")
            score   = row["total_score"]
            cluster = row.get("cluster_label", "—")
            top5_lines.append(
                f"  {i}. {code} ({region})\n"
                f"     Score: {score:.1f}/100   Archetype: {cluster}"
            )

        #  Archetypes 
        arch_lines = []
        if "cluster_label" in df.columns:
            stats = (
                df.groupby("cluster_label")["total_score"]
                .agg(["mean", "min", "max", "count"])
                .round(1)
                .sort_values("mean", ascending=False)
            )
            for label, row in stats.iterrows():
                countries = df[df["cluster_label"] == label].index.tolist()
                listed    = ", ".join(countries[:5])
                if len(countries) > 5:
                    listed += f" + {len(countries) - 5} more"
                arch_lines.append(
                    f"  {label}  (n={int(row['count'])}, "
                    f"avg {row['mean']:.1f}, range {row['min']:.1f}–{row['max']:.1f})\n"
                    f"    {listed}"
                )

        #  Regional summary 
        region_lines = []
        if "region" in df.columns:
            reg = (
                df.groupby("region")["total_score"]
                .agg(["mean", "count"])
                .round(1)
                .sort_values("mean", ascending=False)
            )
            for region, row in reg.iterrows():
                region_lines.append(
                    f"  {region}: {row['mean']:.1f} avg  "
                    f"({int(row['count'])} countries)"
                )

        #  Stability summary (from sensitivity analysis) 
        stability_lines = []
        if "stability" in sens.columns:
            stab_counts = sens["stability"].value_counts()
            stability_lines.append("  Stability across 4 investor scenarios:")
            for label, count in stab_counts.items():
                stability_lines.append(f"    {label}: {count} markets ({count/len(sens)*100:.0f}%)")
            
            # Find markets with perfect stability (rank_std == 0)
            perfect_stable = sens[sens["rank_std"] == 0].index.tolist()
            if perfect_stable:
                stability_lines.append(f"\n  Perfectly stable markets (rank unchanged across all scenarios):")
                stability_lines.append(f"    {', '.join(perfect_stable)}")

        #  Assemble 
        sep = "-" * 60

        summary = "\n".join([
            sep,
            "RENEWABLE ENERGY INVESTMENT ANALYZER",
            "Executive Summary — Emerging Markets Clean Power Opportunity",
            sep,
            f"Date:      {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Period:    2018–2023 (6-year means)",
            f"Countries: {len(df)} across 3 regions",
            f"Indicators: 13 (updated: modern renewables, fossil electricity share, fuel imports)",
            "",
            "Archetype thresholds: Ready Markets (≥70) · Transition (40–70) · Watch & Wait (<40)",
            "",
            sep,
            "TOP 5 INVESTMENT OPPORTUNITIES",
            sep,
            *top5_lines,
            "",
            sep,
            "MARKET ARCHETYPES",
            sep,
            *arch_lines,
            "",
            sep,
            "REGIONAL SUMMARY",
            sep,
            *region_lines,
            "",
            sep,
            "RANK STABILITY",
            sep,
            *stability_lines,
            "",
            sep,
            "OUTPUT FILES",
            sep,
            "  data/processed/indicators.csv             6-year means (13 indicators)",
            "  data/processed/normalized_indicators.csv  0-100 normalised scores",
            "  outputs/market_scores.csv                  Balanced scenario scores",
            "  outputs/sensitivity_analysis.csv           Cross-scenario comparison (4 scenarios)",
            "  outputs/market_clusters.csv                Cluster assignments (threshold-based)",
            "  outputs/executive_summary.txt              This file",
            sep,
        ])

        out_path = Path("outputs/executive_summary.txt")
        out_path.write_text(summary, encoding="utf-8")
        print(f"  ✓ Saved → {out_path}")
        print(f"\n{summary[:800]}\n  ...")

    except Exception as exc:
        print(f"  ⚠ Could not generate summary: {exc}")

        out_path = Path("outputs/executive_summary.txt")
        out_path.write_text(summary, encoding="utf-8")
        print(f"  ✓ Saved → {out_path}")
        print(f"\n{summary[:600]}\n  ...")

    except Exception as exc:
        print(f"  ⚠ Could not generate summary: {exc}")


# ── Pipeline ───────────────────────────────────────────────────────────────

# Ordered list of (step_number, display_name, script_filename)
PIPELINE_STEPS = [
    (1, "Data Ingestion",  "data_ingestion.py"),
    (2, "Preprocessing",   "preprocessing.py"),
    (3, "Scoring",         "scoring.py"),
    (4, "Clustering",      "clustering.py"),
]


def run_pipeline() -> None:
    """Execute all pipeline steps in order, then write the executive summary."""
    t0 = time.time()

    print(f"\n{'=' * 60}")
    print("RENEWABLE ENERGY INVESTMENT ANALYZER")
    print("Pipeline Orchestrator  ·  2018–2023  ·  20 markets  ·  13 indicators")
    print(f"{'=' * 60}")

    ensure_output_dirs()

    for step_num, step_name, script in PIPELINE_STEPS:
        if not run_step(step_num, step_name, script):
            print(f"\n  Pipeline aborted at step {step_num} — {step_name}")
            sys.exit(1)

    generate_executive_summary()

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"PIPELINE COMPLETE  ·  {elapsed:.1f} s")
    print(f"{'=' * 60}")
    print("\n  Run 'python app.py' to launch the interactive dashboard")
    print(f"{'=' * 60}\n")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        run_pipeline()
    except KeyboardInterrupt:
        print("\n  Pipeline interrupted by user")
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        sys.exit(1)