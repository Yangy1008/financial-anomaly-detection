# Quick Start Guide

Get the full pipeline — database build, cleaning, exploratory analysis, anomaly detection, cross-domain demo, and Power BI export — running in a few minutes.

## Prerequisites

- **Python 3.10+** (this project was built and tested on 3.13.2; the pinned dependency versions in `requirements.txt` require at least 3.10 — if you're on an older Python, see the note at the bottom of this section)
- **sqlite3** — built into Python's standard library (`import sqlite3` — no separate install needed). The `sqlite3` *command-line tool* is optional and only used if you want to inspect `data/insurance_claims.db` directly outside of Python; it ships with macOS/Linux by default.
- The three source CSVs already present in the project root: `insurance_data.csv`, `employee_data.csv`, `vendor_data.csv`.

> **On Python 3.8/3.9**: `requirements.txt` pins versions that require 3.10+ (numpy 2.x, matplotlib 3.10). To run on an older Python, install unpinned versions instead — `pip install pandas numpy scikit-learn matplotlib seaborn pyarrow` — the pipeline code itself has no 3.10-specific syntax.

## Installation

```bash
# From the project root
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

That installs pandas, numpy, scikit-learn, matplotlib, seaborn, and pyarrow at the tested versions (see `requirements.txt`).

## Running the Pipeline

Run the five scripts **in order** from the project root — each one depends on files the previous one produced. Every script is fast (seconds, not minutes) on this dataset size.

```bash
python3 python/01_data_import.py
python3 python/02_exploratory_analysis.py
python3 python/03_anomaly_detection.py
python3 python/04_framework_portability.py
python3 python/05_prepare_for_powerbi.py
```

### What each script does, and what to expect

**`01_data_import.py`** — builds `data/insurance_claims.db` from the three source CSVs using `sql/schema.sql`, cleans the data (fixes date types, handles missing values, drops sensitive/unnecessary columns), engineers `days_to_process` / `claim_to_premium_ratio` / `customer_claim_frequency`, validates data quality, and exports clean Parquet files.
> **Output**: `data/insurance_claims.db`, `output/insurance_claims_clean.parquet`, `output/agents_clean.parquet`, `output/vendors_clean.parquet`, `logs/01_data_import.log`. Console ends with `=== Pipeline completed successfully ===`.

**`02_exploratory_analysis.py`** — runs 10 analysis sections (statistics, distributions, correlations, customer segmentation, insurance-type breakdown, agent performance, geography, temporal patterns, risk segmentation, outliers).
> **Output**: 13 PNG charts in `output/eda_plots/`, 10 summary CSVs in `output/` (prefixed `eda_`), `logs/02_exploratory_analysis.log`.

**`03_anomaly_detection.py`** — scores every claim with 4 independent methods (Z-score, Isolation Forest, Local Outlier Factor, business rules), evaluates each, and combines them into a weighted ensemble score.
> **Output**: `output/anomaly_scores.parquet` (every claim, all scores), `output/top_anomalies.csv` (top 500, with explanations), `output/method_comparison.csv`, 4 PNGs in `output/anomaly_plots/`, `logs/03_anomaly_detection.log`.

**`04_framework_portability.py`** — proves the same detection engine works on other financial domains by generating synthetic payment-transaction and fund-settlement data (with true injected anomaly labels) and running the identical algorithms against them.
> **Output**: `output/portability_comparison.csv`, `output/scenario_outputs/{insurance_claims,payment_transactions,fund_operations}/`, `output/framework_adaptation_notes.txt`, `logs/04_framework_portability.log`.

**`05_prepare_for_powerbi.py`** — packages the anomaly detection results into 9 flat, dashboard-ready CSV tables plus a setup guide.
> **Output**: 9 CSVs + `PowerBI_SETUP.txt` in `output/powerbi_data/`, `logs/05_prepare_for_powerbi.log`.

### Verifying it worked

Each script prints a log banner and ends with a success message; a non-zero exit code (`echo $?` after running) means something failed — check the matching file in `logs/` for the full traceback (the console only shows INFO-level messages; the log file has DEBUG-level detail).

```bash
python3 python/01_data_import.py && echo "01 OK"
```

## Next Steps: Using the Power BI Export

After running all five scripts, `output/powerbi_data/` contains everything needed for a dashboard:

1. Open Power BI Desktop.
2. **Get Data → Folder** and point it at `output/powerbi_data/` (or **Get Data → Text/CSV** and select the 9 `.csv` files individually — everything except `PowerBI_SETUP.txt`, which is documentation, not data).
3. Open `output/powerbi_data/PowerBI_SETUP.txt` — it tells you, file by file:
   - which chart type to use for each table (bar, scatter, map, card, etc.)
   - which relationships to draw in Model view (`fact_claims` relates to `agent_performance`, `vendor_performance`, `claims_overview`, `risk_segmentation`, and `geographic_analysis` by their shared key columns)
   - which slicers to add (`insurance_type`, `state`, `risk_segmentation`, `claim_status`, `is_top_100_anomaly`)
   - data caveats to keep in mind while building visuals (what "flagged" actually means, and why a couple of business-rule base rates are misleadingly high in this dataset — both explained in more depth in `README.md` and `FRAMEWORK_MAPPING.md`)
4. Build relationships first (Model view), then build visuals — the guide's recommendations assume the relationships are already in place so cross-filtering works as expected.

For the full project narrative — problem statement, findings, and how the detection framework generalizes to other financial domains — see `README.md` and `FRAMEWORK_MAPPING.md`.
