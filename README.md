# Financial Anomaly Detection Framework

## Executive Summary

This project builds an end-to-end pipeline that ingests raw insurance claims data, cleans and enriches it, explores it statistically, and scores every claim for anomalousness using four independent, complementary detection methods (a statistical Z-score test, Isolation Forest, Local Outlier Factor, and hand-written business rules) combined into a single weighted ensemble score. The pipeline runs entirely on a SQLite database and Python (pandas, scikit-learn, matplotlib/seaborn), and every summary table it produces is exported as CSV so it can be dropped directly into Power BI, Tableau, or any other BI tool for dashboarding — no rework required.

Beyond the insurance use case, the project's second deliverable is a demonstration that the detection framework itself is domain-agnostic. `python/04_framework_portability.py` refactors the same four detection algorithms into generic, config-driven functions and re-runs them, unchanged, against two additional synthetic financial domains — payment transactions and fund settlement operations — swapping out only the column names, feature lists, and business rules per domain. A cross-check confirms the generic engine reproduces the original insurance results bit-for-bit, and `FRAMEWORK_MAPPING.md` documents exactly what changes (and what doesn't) to port the framework to a fourth domain, AML/compliance monitoring, that was mapped out but not implemented in code.

The honest headline finding is as important as the framework itself: on the real insurance data, all four detection methods score close to random (ROC-AUC ≈ 0.50) against the only available evaluation label, `incident_severity`. That is not a bug — it is evidence that claim severity and statistical anomalousness are simply uncorrelated in this dataset, and it is a reminder that unsupervised anomaly scores must be validated against real investigation outcomes before being trusted in production. The same framework, run against synthetic data with true injected anomaly labels, reaches ROC-AUC of 0.78–0.98, confirming the algorithms themselves work as intended — it was the evaluation label, not the detectors, that was weak. Every script in this pipeline documents its findings, including its own limitations, directly in its log output.

## Problem Statement

Insurance claims data is high-volume and high-dimensional: a single claim carries policy details, customer demographics, incident context, and financial terms, and a meaningful fraction of claims are large enough that even a small false-negative rate has real dollar consequences. Manual review of every claim doesn't scale, and any single detection technique has known blind spots — a hard percentile threshold misses multivariate patterns, a black-box model resists explanation to an adjuster, and a fixed business rule can quietly saturate (fire on nearly every row) if the world it was written for doesn't match the data it's applied to, as this project discovers firsthand. The goal here is a layered detection system where each method's weaknesses are covered by the others, every flagged claim comes with a plain-language explanation of *why* it was flagged, and the whole system is built so the same code can be pointed at a different financial dataset (payments, fund operations, compliance) with configuration changes rather than a rewrite.

## Technical Approach

**SQL** (`sql/schema.sql`) defines the relational schema for the three source entities — `insurance_claims`, `agents`, `vendors` — with foreign keys, `CHECK` constraints derived from the actual distinct values profiled in the source CSVs (not guessed), and indexes on the columns the pipeline and any downstream BI tool will filter/join on most (`loss_dt`, `agent_id`, `claim_amount`, etc.). SQLite was chosen for zero-setup portability; the schema is standard enough to port to Postgres/MySQL with minor syntax changes if needed.

**Python** (`python/01`–`04`) does everything downstream of the schema: loading the CSVs into SQLite, cleaning and feature-engineering the data, exploratory analysis with statistical summaries and charts, the four-method anomaly detection engine and its evaluation, and the cross-domain portability demonstration. Every script is runnable standalone, logs to both console and `logs/`, and fails gracefully (a failure in one analysis section or detection method is logged and skipped rather than crashing the whole run).

**Power BI** (or any BI tool) is the intended presentation layer for this pipeline's outputs, not something built into this repository. Every table in `output/` that starts with `eda_summary_*`, `portability_comparison.csv`, `method_comparison.csv`, and `top_anomalies.csv` is a flat, tool-agnostic CSV designed to be loaded directly into a BI tool for dashboarding (claims-by-state maps, agent performance scorecards, anomaly trend lines, etc.) without any reshaping. No `.pbix` file is included — see **Future Improvements** below.

## Data Overview and Quality Notes

Three source CSVs, profiled directly (not assumed) before the schema and cleaning logic were written:

| Source file | Rows | Description |
|---|---|---|
| `insurance_data.csv` | 10,000 | Claims: policy, customer, incident, and financial details |
| `employee_data.csv` | 1,200 | Insurance agents |
| `vendor_data.csv` | 600 | Third-party service/repair vendors |

Notable data-quality findings from profiling, all handled explicitly in `01_data_import.py`:
- **`AUTHORITY_CONTACTED` contains the literal string `'None'`** (1,945 rows) and **`CUSTOMER_EDUCATION_LEVEL` contains the literal string `'NA'`** (529 rows) — both are legitimate category values, not missing data. pandas' default CSV parsing treats both strings as null by default, which would have silently corrupted them; the import script explicitly disables that behavior (`keep_default_na=False`).
- **`VENDOR_ID` is empty on 3,245 of 10,000 claims** — a claim genuinely having no vendor is a valid business state, not missing data, so it is left as `NULL` rather than imputed.
- Sensitive fields (SSN, bank routing/account numbers) are dropped entirely during cleaning; they carry no analytical value for anomaly detection and shouldn't be retained.
- Overall data quality is otherwise very clean: no duplicate transaction IDs, no negative amounts, no referential-integrity violations against `agents`/`vendors`.

## Key Findings

- **State mismatch is the norm, not the exception, in this dataset**: 93.6% of claims have a policyholder state that differs from the incident state. Used naively as a fraud rule it flags almost everything (see below) — a reminder to always check a rule's base rate before trusting it.
- **Claim amounts are structurally large relative to premiums**: the median `claim_to_premium_ratio` is 74.5x, and even the 5th percentile is ~12x — so a "claim > 10x premium" rule, which sounds like a sensible red flag, actually fires on 97.6% of all claims in this data.
- **Claim amounts are bounded, not heavy-tailed, within each insurance type**: the maximum Z-score observed anywhere in the dataset is 1.76σ, so the classic 3-sigma statistical outlier test flags zero claims. This is a genuine property of the (evidently synthetic) data, not a bug.
- **All four detection methods land on the ROC random diagonal (AUC ≈ 0.50)** against the only available evaluation label (`incident_severity == 'Total Loss'`). This means severity and statistical anomalousness are uncorrelated here — the metrics measure "agreement with severity," not fraud-detection skill, and should not be read as real-world performance.
- **The same framework, evaluated against true labels on synthetic data, performs well**: ROC-AUC reaches 0.91 (Isolation Forest, payments), 0.98 (ensemble, fund operations) — strong evidence the detection algorithms themselves are sound, and that the insurance evaluation numbers reflect a proxy-label problem rather than a modeling problem.
- **Life insurance claims dominate by dollar value**: average claim $54,386 vs. a dataset-wide average of $16,564, despite Mobile insurance having a similar claim *count*. Insurance type is the single strongest driver of claim scale (correlation-matrix and EDA bar charts both confirm this).
- **Claims cluster on Saturdays and around 3 PM**, and peaked in December 2020 (828 claims) — worth investigating whether this reflects genuine seasonal incident patterns or a reporting/data-generation artifact.
- Approval rate is stable at ~95% across every insurance type, risk segment, and the large majority of agents — a small number of agents sit well outside that band (see `output/eda_plots/agent_performance.png`), which is exactly the kind of pattern this framework is built to surface for follow-up.

## How to Run

**Requirements**: Python 3.10+, `sqlite3` CLI (optional, for manual DB inspection), and the packages `pandas`, `numpy`, `pyarrow`, `matplotlib`, `seaborn`, `scikit-learn`.

```bash
pip install pandas numpy pyarrow matplotlib seaborn scikit-learn
```

Run the pipeline in order from the project root — each script depends on the previous one's output:

```bash
# 1. Build the SQLite database from the source CSVs, clean, feature-engineer, validate, export to Parquet
python3 python/01_data_import.py

# 2. Exploratory analysis: statistics, distributions, correlations, segmentation, plots
python3 python/02_exploratory_analysis.py

# 3. Core anomaly detection: 4 methods, evaluation, ensemble, top-500 shortlist
python3 python/03_anomaly_detection.py

# 4. Cross-domain portability demonstration (synthetic payments + fund operations)
python3 python/04_framework_portability.py
```

Each script prints a progress log to the console and writes a detailed copy to `logs/0N_<script_name>.log`. All generated data (`data/insurance_claims.db`) and results (`output/`) are reproducible from the three source CSVs — nothing in `data/` or `output/` needs to be committed to version control.

To inspect the database directly:
```bash
sqlite3 data/insurance_claims.db "SELECT * FROM insurance_claims LIMIT 5;"
```

## Project Structure

```
financial-anomaly-detection/
├── README.md                        - this file
├── FRAMEWORK_MAPPING.md              - cross-domain portability reference
├── insurance_data.csv                - source: 10,000 claims
├── employee_data.csv                 - source: 1,200 agents
├── vendor_data.csv                   - source: 600 vendors
├── sql/
│   └── schema.sql                    - SQLite schema: 3 tables, FKs, CHECK constraints, indexes
├── python/
│   ├── 01_data_import.py             - build DB, clean, feature-engineer, validate, export to Parquet
│   ├── 02_exploratory_analysis.py    - statistics, distributions, correlations, segmentation, 10 EDA sections
│   ├── 03_anomaly_detection.py       - 4-method detection engine, evaluation, ensemble scoring
│   └── 04_framework_portability.py   - generic engine + payments/funds synthetic scenarios
├── data/                             - generated: data/insurance_claims.db (gitignore-able)
├── logs/                             - generated: one log file per script
└── output/                           - generated: all analysis results
    ├── *_clean.parquet               - cleaned insurance_claims / agents / vendors tables
    ├── eda_summary_*.csv             - EDA summary tables (BI-tool ready)
    ├── eda_plots/                    - 13 PNG charts from the EDA stage
    ├── anomaly_scores.parquet        - every claim scored by all 4 methods + ensemble
    ├── top_anomalies.csv             - top 500 anomalies with plain-language explanations
    ├── method_comparison.csv         - precision/recall/F1/ROC-AUC per detection method
    ├── anomaly_plots/                - score distributions, ROC curves, confusion matrices
    ├── portability_comparison.csv    - all methods x all 3 domains, side by side
    ├── framework_adaptation_notes.txt - what changes per domain, and why
    └── scenario_outputs/             - per-domain scores/top-anomalies/comparison (insurance, payments, funds)
```

## Results and Metrics

**Insurance claims (real data, evaluated against the `incident_severity` proxy label — see caveat above):**

| Method | Precision | Recall | F1 | ROC-AUC | Flagged |
|---|---|---|---|---|---|
| Z-score | 0.000 | 0.000 | 0.000 | 0.497 | 0 (0.0%) |
| Isolation Forest | 0.344 | 0.051 | 0.088 | 0.505 | 500 (5.0%) |
| Local Outlier Factor | 0.348 | 0.051 | 0.089 | 0.507 | 500 (5.0%) |
| Rule-based | 0.339 | 0.998 | 0.506 | 0.502 | 9,985 (99.9%) |

**Cross-domain portability (synthetic data, evaluated against true injected anomaly labels):**

| Domain | Best method | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Payment transactions | Ensemble | 0.518 | 0.739 | 0.609 | 0.948 |
| Fund operations | Ensemble | 0.672 | 0.840 | 0.747 | 0.984 |

Full per-method, per-domain results (including confusion matrix counts) are in `output/method_comparison.csv` and `output/portability_comparison.csv`.

## Technologies Used

- **SQLite** — embedded relational database, zero-setup schema enforcement (foreign keys, CHECK constraints)
- **Python 3** — pipeline orchestration and analysis
- **pandas / NumPy** — data cleaning, feature engineering, aggregation
- **PyArrow / Parquet** — columnar storage for the cleaned, analysis-ready datasets
- **scikit-learn** — `IsolationForest`, `LocalOutlierFactor`, `StandardScaler`, classification metrics
- **matplotlib / seaborn** — all charts, styled with a validated colorblind-safe palette
- **Power BI** (or equivalent) — intended presentation layer for the exported CSV summary tables (not included as a built artifact — see below)

## Future Improvements

- **Build the actual Power BI (or Tableau/Looker) dashboard** on top of the exported CSVs — this repo produces BI-ready data but no `.pbix`/dashboard file yet.
- **Replace the severity proxy label with real investigation outcomes** (confirmed fraud/no-fraud, or adjuster override decisions) if that data ever becomes available — the entire evaluation section of `03_anomaly_detection.py` is built to swap in a real label with a one-line change.
- **Re-tune or retire the rule-based method for insurance specifically** — both of its business rules are saturated on this dataset (97.6% and 93.6% base rates); either tighter, dataset-specific thresholds or a different rule pair would make it a real contributor instead of a near-constant "flag everything" signal.
- **Implement the AML/compliance domain** mapped out (but not coded) in `FRAMEWORK_MAPPING.md`, and validate the framework against a fourth real or synthetic dataset.
- **Add a supervised layer**: once real labels exist, the four unsupervised scores in `anomaly_scores.parquet` become natural input features to a supervised classifier (e.g. gradient boosting) rather than an end product in their own right.
- **Automate the pipeline** (e.g. a scheduled job re-running `01`→`04` as new claims data arrives) rather than the current manual, notebook-style invocation.
