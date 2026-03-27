# Renewable Energy Investment Analyzer

A quantitative market prioritization framework for emerging-market renewable energy investment. Built for a European clean energy developer screening 20 markets across Southeast Asia, Africa, and Latin America for solar and wind deployment opportunities.

The pipeline fetches 13 World Bank indicators, normalizes them to a common 0–100 scale, scores markets across four strategic dimensions, tests rankings under four investor scenarios, and segments markets into three investment archetypes using a hybrid approach (K-Means + thresholds). An interactive Dash dashboard and a Quarto HTML report present the findings.

---

## Project Structure

```
renewable-energy-analyzer/
│
├── config/
│   └── config.yaml                  # Countries, indicators, weights, clustering
│
├── src/
│   ├── data_ingestion.py            # World Bank API fetch — 6-year means + raw export
│   ├── preprocessing.py             # Imputation, winsorisation, 0–100 normalisation
│   ├── scoring.py                   # Weighted dimension scores + sensitivity analysis
│   └── clustering.py                # K-Means market archetype segmentation
│
├── data/
│   ├── raw/                         # Per-indicator yearly CSVs + combined file
│   └── processed/                   # 6-year means and normalised indicators
│
├── outputs/
│   ├── market_scores.csv            # Balanced scenario scores
│   ├── sensitivity_analysis.csv     # Cross-scenario rank comparison
│   ├── market_clusters.csv          # Cluster assignments
│   ├── cluster_profiles.csv         # Mean dimension scores per archetype
│   └── executive_summary.txt        # Auto-generated text summary
│
├── app.py                           # Interactive Dash dashboard
├── main.py                          # Pipeline orchestrator
├── index.qmd                        # Quarto analytical report
├── footer.html                      # Shared report footer
├── requirements.txt
└── README.md
```

---

## Analytical Framework

Markets are scored across four dimensions using World Bank Open Data indicators (2018–2023 average).

### Indicators by Dimension

| Dimension | Indicator | Source Code | Direction |
|-----------|-----------|-------------|-----------|
| **Market Opportunity** | GDP per capita growth | NY.GDP.PCAP.KD.ZG | Higher → better |
| | Electricity consumption per capita | EG.USE.ELEC.KH.PC | Higher → better |
| | Population growth rate | SP.POP.GROW | Higher → better |
| | Urban population share | SP.URB.TOTL.IN.ZS | Higher → better |
| **Decarbonisation** | Energy use per capita | EG.USE.PCAP.KG.OE | Higher → better |
| | Fossil electricity share | EG.ELC.FOSL.ZS | Higher → better |
| | Modern renewable share (inverse) | EG.ELC.RNWX.ZS | Lower → better |
| | Electricity access rate | EG.ELC.ACCS.ZS | Higher → better |
| **Business Environment** | Political stability | PV.EST | Higher → better |
| | Regulatory quality | RQ.EST | Higher → better |
| | Rule of law | RL.EST | Higher → better |
| | Control of corruption | CC.EST | Higher → better |
| **Energy Security** | Net energy imports | EG.IMP.CONS.ZS | Higher → better |

### Dimensions and Baseline Weights

| Dimension | Weight | What It Captures |
|-----------|-------:|-----------------|
| Decarbonisation Opportunity | 45% | Fossil fuel dependency, modern renewable gap, carbon intensity |
| Market Opportunity | 30% | GDP growth, electricity consumption, population and urbanisation dynamics |
| Business Environment | 20% | Political stability, regulatory quality, rule of law, corruption control |
| Energy Security | 5% | Net energy imports (energy dependency) |


### Four Investor Scenarios

| Scenario | Dominant Dimension | Capital Mandate |
|----------|--------------------|----------------|
| Balanced | Decarbonisation (45%) | Neutral reference |
| Impact-First | Decarbonisation (60%) | Development finance, climate-impact funds |
| Growth-Focused | Market Opportunity (55%) | Commercial infrastructure funds, utilities |
| Risk-Averse | Business Environment (40%) | Pension funds, insurance-backed capital |


### Market Archetypes (Hybrid Approach)

Markets are segmented using a hybrid approach that combines K-Means clustering (statistical validation) with score-based thresholds (business clarity):

| Archetype | Threshold | N | Avg. Score | Markets |
|-----------|-----------|--:|----------:|---------|
| **Ready Markets** | Score ≥ 70 | 2 | 78.4 | Malaysia, Chile |
| **Transition Markets** | 40 ≤ Score < 70 | 15 | 49.3 | Thailand, Vietnam, Mexico, Indonesia, Ghana, Senegal, Côte d'Ivoire, Bangladesh, Peru, Morocco, Colombia, Philippines, Brazil, Cambodia, Tanzania |
| **Watch & Wait** | Score < 40 | 3 | 28.4 | Nigeria, Ethiopia, Kenya |

*K-Means silhouette score: 0.349 (good separation)*

--- 


## Pipeline

```
World Bank API
      │
      ▼
data_ingestion.py   →   data/raw/           (yearly values per indicator)
                    →   data/processed/     (6-year means)
      │
      ▼
preprocessing.py    →   data/processed/normalized_indicators.csv
      │
      ▼
scoring.py          →   outputs/market_scores.csv
                    →   outputs/sensitivity_analysis.csv
      │
      ▼
clustering.py       →   outputs/market_clusters.csv
                    →   outputs/cluster_profiles.csv
```

Run the full pipeline in one command:

```bash
python main.py
```

Or run individual modules:

```bash
python src/data_ingestion.py
python src/preprocessing.py
python src/scoring.py
python src/clustering.py
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- A World Bank API connection (no key required — `wbgapi` uses the public API)
- Quarto (for report rendering)

### Installation

```bash
# Clone the repository
git clone https://github.com/IbrahimaFikry/renewable-energy-analyzer.git
cd renewable-energy-analyzer

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
python main.py
```

This fetches all data, runs preprocessing, scoring, and clustering, and writes outputs to the `outputs/` directory.

### Launch the Dashboard

```bash
python app.py
```

Navigate to [http://localhost:8050](http://localhost:8050).

### Render the Report

```bash
quarto render renewable_energy_revised.qmd
```

Requires [Quarto](https://quarto.org) to be installed.

---

## Dashboard

The Dash app mirrors the structure of the Quarto report and provides five interactive tabs:

| Tab | Content |
|-----|---------|
| Executive Summary | KPI strip, full rankings chart, governance scatter, archetype cards |
| Market Rankings | Horizontal bar chart, dimensional dot plot, score decomposition, data table |
| Sensitivity Analysis | Rank stability dot plot with range lines, stability statistics table |
| Market Archetypes | PCA scatter with cluster boundaries, radar chart, archetype definitions |
| Scenario Builder | Live weight sliders, preset buttons, real-time ranking update |

---

## Configuration

All pipeline parameters are controlled through `config/config.yaml`.

**To add a country:** add the ISO-3 code under the appropriate region.

**To change dimension weights:** edit the relevant scenario under `weight_scenarios`. Weights must sum to 1.0.

**To add an indicator:** add an entry under the appropriate dimension with `wb_indicator`, `direction`, and `weight`. The pipeline picks it up automatically on the next run.

**To change the number of archetypes:** update `clustering.n_clusters` in `config.yaml`.
Note that threshold-based labels override K-Means assignments, so the final archetype counts are determined by score thresholds (Ready ≥70, Transition 40–70, Watch & Wait <40).

---

## Reproducibility

All data is retrieved programmatically from the [World Bank Open Data API](https://data.worldbank.org) via `wbgapi`. No manual data entry at any stage. Full lineage is documented in `src/data_ingestion.py`.

To reproduce from scratch:

```bash
# Clear existing outputs
rm -rf data/raw data/processed outputs/

# Re-run the full pipeline
python main.py
```

---

## Requirements

See `requirements.txt` for the full package list. Core dependencies:

- `pandas`, `numpy` — data manipulation
- `scikit-learn` — StandardScaler, KMeans, silhouette score
- `scipy` — ConvexHull for PCA cluster visualisation
- `plotly` — interactive charts in both the report and dashboard
- `dash` — interactive web dashboard
- `wbgapi` — World Bank Open Data API client
- `pyyaml` — configuration loading
- `tenacity` — retry logic for API calls
- `quarto` — HTML report rendering (install separately from [quarto.org](https://quarto.org))

---

## Author

**Ibrahima Fikry Diallo**
Operations Research & Decision Support Engineer | Data Analyst

[ibrahimafikrydiallo.com](https://ibrahimafikrydiallo.com) · [LinkedIn](https://www.linkedin.com/in/ibrahimafikrydiallo) · [GitHub](https://github.com/IbrahimaFikry)

---

*Data source: [World Bank Open Data](https://data.worldbank.org) · Analysis period: 2018–2023 · Framework v1.0*
