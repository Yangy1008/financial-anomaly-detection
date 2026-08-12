#!/usr/bin/env python3
"""
Packages the results of 03_anomaly_detection.py into a set of flat,
Power-BI-ready CSV tables plus a setup guide explaining how to wire them
into a dashboard.

Design: one wide fact table (fact_claims) at claim grain, carrying every
dimension key (agent_id, vendor_id, insurance_type, state, risk_segmentation)
so it can be sliced any way in Power BI, plus a small set of pre-aggregated
summary tables for the specific visuals requested. All "flagged" counts
across every table refer to the SAME top-100-by-ensemble-score set (loaded
from top_anomalies.csv), so numbers are consistent whichever table a
dashboard reader looks at.

Output:
    output/powerbi_data/*.csv        - 9 import-ready tables
    output/powerbi_data/PowerBI_SETUP.txt - which file for which visual, relationships, filters
    logs/05_prepare_for_powerbi.log
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
POWERBI_DIR = OUTPUT_DIR / "powerbi_data"
LOG_PATH = BASE_DIR / "logs" / "05_prepare_for_powerbi.log"

SCORES_PATH = OUTPUT_DIR / "anomaly_scores.parquet"
TOP_ANOMALIES_PATH = OUTPUT_DIR / "top_anomalies.csv"
AGENTS_PATH = OUTPUT_DIR / "agents_clean.parquet"
VENDORS_PATH = OUTPUT_DIR / "vendors_clean.parquet"

TOP_N_DASHBOARD = 100  # every "flagged" figure across every exported table refers to this same set

logger = logging.getLogger("powerbi_prep")


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(console)
    logger.addHandler(file_handler)


def print_section(title: str) -> None:
    logger.info("\n" + f" {title} ".center(78, "="))


def save_csv(df: pd.DataFrame, filename: str) -> None:
    POWERBI_DIR.mkdir(parents=True, exist_ok=True)
    out_path = POWERBI_DIR / filename
    df.to_csv(out_path, index=False)
    logger.info("Saved -> %s (%d rows, %d cols)", out_path, len(df), df.shape[1])


# ---------------------------------------------------------------------------
# Step 1: load anomaly detection results
# ---------------------------------------------------------------------------
def load_data() -> tuple:
    for path in (SCORES_PATH, TOP_ANOMALIES_PATH, AGENTS_PATH, VENDORS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run 01_data_import.py and 03_anomaly_detection.py first.")

    scores = pd.read_parquet(SCORES_PATH)
    top_anomalies = pd.read_csv(TOP_ANOMALIES_PATH)  # already sorted by ensemble_score desc (03's top-500 shortlist)
    agents = pd.read_parquet(AGENTS_PATH)
    vendors = pd.read_parquet(VENDORS_PATH)
    logger.info(
        "Loaded %d scored claims, %d shortlisted anomalies, %d agents, %d vendors",
        len(scores), len(top_anomalies), len(agents), len(vendors),
    )

    # The single, consistent "flagged" definition used everywhere below
    top_ids = set(top_anomalies.sort_values("ensemble_score", ascending=False).head(TOP_N_DASHBOARD)["transaction_id"])
    scores["is_top_100_anomaly"] = scores["transaction_id"].isin(top_ids)
    logger.info("Defined is_top_100_anomaly from the top %d of top_anomalies.csv by ensemble_score", TOP_N_DASHBOARD)

    return scores, top_anomalies, agents, vendors


# ---------------------------------------------------------------------------
# fact_claims: the backbone table every other table can relate to
# ---------------------------------------------------------------------------
def build_fact_claims(scores: pd.DataFrame) -> pd.DataFrame:
    # customer_name / address dropped here on top of 01's PII cleanup - a
    # dashboard export is more widely shared than the raw pipeline output,
    # so claim-grain PII is minimized further; customer_id remains as the
    # slicer/join key.
    cols = [
        "transaction_id", "customer_id", "agent_id", "vendor_id",
        "txn_date_time", "loss_dt", "report_dt",
        "insurance_type", "claim_status", "incident_severity", "risk_segmentation",
        "state", "incident_state", "city", "incident_city",
        "claim_amount", "premium_amount", "claim_to_premium_ratio", "days_to_process",
        "age", "tenure", "no_of_family_members", "employment_status", "marital_status",
        "house_type", "social_class", "customer_education_level",
        "any_injury", "police_report_available", "authority_contacted", "incident_hour_of_the_day",
        "zscore_score", "iso_score", "lof_score", "rule_score", "ensemble_score",
        "zscore_flag", "iso_flag", "lof_flag", "rule_flag", "is_top_100_anomaly",
    ]
    fact = scores[cols].copy()
    logger.info("fact_claims: %d rows, %d columns (PII fields customer_name/address excluded)", len(fact), fact.shape[1])
    return fact


# ---------------------------------------------------------------------------
# claims_overview: by insurance_type
# ---------------------------------------------------------------------------
def build_claims_overview(scores: pd.DataFrame) -> pd.DataFrame:
    g = scores.groupby("insurance_type")
    overview = g.agg(
        total_claims=("transaction_id", "count"),
        approved_count=("claim_status", lambda s: (s == "A").sum()),
        denied_count=("claim_status", lambda s: (s == "D").sum()),
        avg_claim_amount=("claim_amount", "mean"),
        median_claim_amount=("claim_amount", "median"),
        avg_premium_amount=("premium_amount", "mean"),
        total_claim_amount=("claim_amount", "sum"),
        flagged_count=("is_top_100_anomaly", "sum"),
    ).reset_index()
    overview["approval_rate"] = overview["approved_count"] / overview["total_claims"]
    overview["flagged_rate"] = overview["flagged_count"] / overview["total_claims"]
    overview = overview.sort_values("total_claim_amount", ascending=False)
    return overview


# ---------------------------------------------------------------------------
# kpi_summary: single-row headline numbers for card visuals
# ---------------------------------------------------------------------------
def build_kpi_summary(scores: pd.DataFrame, agents: pd.DataFrame, vendors: pd.DataFrame) -> pd.DataFrame:
    row = {
        "total_claims": len(scores),
        "total_agents": len(agents),
        "total_vendors": len(vendors),
        "approved_count": int((scores["claim_status"] == "A").sum()),
        "denied_count": int((scores["claim_status"] == "D").sum()),
        "overall_approval_rate": (scores["claim_status"] == "A").mean(),
        "avg_claim_amount": scores["claim_amount"].mean(),
        "total_claim_value": scores["claim_amount"].sum(),
        "total_premium_value": scores["premium_amount"].sum(),
        "high_risk_claim_count": int((scores["risk_segmentation"] == "H").sum()),
        "flagged_anomaly_count": int(scores["is_top_100_anomaly"].sum()),
        "flagged_anomaly_rate": scores["is_top_100_anomaly"].mean(),
        "avg_ensemble_score": scores["ensemble_score"].mean(),
        "data_start_date": str(scores["loss_dt"].min().date()),
        "data_end_date": str(scores["loss_dt"].max().date()),
    }
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# anomaly_dashboard: top 100 flagged claims with explanations
# ---------------------------------------------------------------------------
def build_anomaly_dashboard(top_anomalies: pd.DataFrame, agents: pd.DataFrame) -> pd.DataFrame:
    top100 = top_anomalies.sort_values("ensemble_score", ascending=False).head(TOP_N_DASHBOARD).copy()
    if "agent_name" not in top100.columns:
        top100 = top100.merge(agents[["agent_id", "agent_name"]], on="agent_id", how="left")
    top100.insert(0, "rank", range(1, len(top100) + 1))
    cols = [
        "rank", "transaction_id", "customer_id", "agent_id", "agent_name", "vendor_id",
        "insurance_type", "claim_amount", "premium_amount", "claim_to_premium_ratio",
        "incident_severity", "claim_status", "state", "incident_state",
        "zscore_score", "iso_score", "lof_score", "rule_score", "ensemble_score", "explanation",
    ]
    return top100[[c for c in cols if c in top100.columns]]


# ---------------------------------------------------------------------------
# agent_performance: per-agent dimension + performance metrics
# ---------------------------------------------------------------------------
def build_agent_performance(scores: pd.DataFrame, agents: pd.DataFrame) -> pd.DataFrame:
    g = scores.groupby("agent_id")
    perf = g.agg(
        claim_count=("transaction_id", "count"),
        approved_count=("claim_status", lambda s: (s == "A").sum()),
        denied_count=("claim_status", lambda s: (s == "D").sum()),
        avg_claim_amount=("claim_amount", "mean"),
        total_claim_amount=("claim_amount", "sum"),
        flagged_count=("is_top_100_anomaly", "sum"),
        avg_ensemble_score=("ensemble_score", "mean"),
    ).reset_index()
    perf["approval_rate"] = perf["approved_count"] / perf["claim_count"]

    dim_cols = [c for c in ["agent_id", "agent_name", "city", "state", "date_of_joining"] if c in agents.columns]
    perf = agents[dim_cols].merge(perf, on="agent_id", how="left")
    for col in ["claim_count", "approved_count", "denied_count", "flagged_count"]:
        perf[col] = perf[col].fillna(0).astype(int)
    n_no_claims = (perf["claim_count"] == 0).sum()
    if n_no_claims:
        logger.info("agent_performance: %d agents have zero claims in this dataset", n_no_claims)
    return perf.sort_values("claim_count", ascending=False)


# ---------------------------------------------------------------------------
# vendor_performance: per-vendor dimension + performance metrics
# ---------------------------------------------------------------------------
def build_vendor_performance(scores: pd.DataFrame, vendors: pd.DataFrame) -> pd.DataFrame:
    n_no_vendor = scores["vendor_id"].isna().sum()
    logger.info(
        "vendor_performance: %d of %d claims (%.1f%%) have no vendor assigned and are excluded from these aggregates",
        n_no_vendor, len(scores), 100 * n_no_vendor / len(scores),
    )
    g = scores.dropna(subset=["vendor_id"]).groupby("vendor_id")
    perf = g.agg(
        claim_count=("transaction_id", "count"),
        avg_claim_amount=("claim_amount", "mean"),
        total_claim_amount=("claim_amount", "sum"),
        flagged_count=("is_top_100_anomaly", "sum"),
        avg_ensemble_score=("ensemble_score", "mean"),
    ).reset_index()

    dim_cols = [c for c in ["vendor_id", "vendor_name", "city", "state"] if c in vendors.columns]
    perf = vendors[dim_cols].merge(perf, on="vendor_id", how="left")
    for col in ["claim_count", "flagged_count"]:
        perf[col] = perf[col].fillna(0).astype(int)
    return perf.sort_values("claim_count", ascending=False)


# ---------------------------------------------------------------------------
# geographic_analysis: by policyholder state
# ---------------------------------------------------------------------------
def build_geographic_analysis(scores: pd.DataFrame) -> pd.DataFrame:
    g = scores.groupby("state")
    geo = g.agg(
        claim_count=("transaction_id", "count"),
        avg_claim_amount=("claim_amount", "mean"),
        total_claim_amount=("claim_amount", "sum"),
        flagged_count=("is_top_100_anomaly", "sum"),
        avg_ensemble_score=("ensemble_score", "mean"),
    ).reset_index()
    geo["flagged_rate"] = geo["flagged_count"] / geo["claim_count"]
    geo["pct_of_total_claims"] = geo["claim_count"] / len(scores)
    return geo.sort_values("claim_count", ascending=False)


# ---------------------------------------------------------------------------
# temporal_analysis: by month and by ISO week, long format
# ---------------------------------------------------------------------------
def build_temporal_analysis(scores: pd.DataFrame) -> pd.DataFrame:
    df = scores.copy()
    # year-month (not bare month name/number) so 2020 and 2021 aren't conflated -
    # this dataset spans 2020-05 through 2021-06, confirmed during EDA.
    df["month"] = df["loss_dt"].dt.to_period("M").astype(str)
    # ISO year-week (e.g. "2020-W23") - unambiguous and sorts correctly as text
    df["week"] = df["loss_dt"].dt.strftime("%G-W%V")

    frames = []
    for period_type, col in [("month", "month"), ("week", "week")]:
        g = df.groupby(col).agg(
            claim_count=("transaction_id", "count"),
            avg_claim_amount=("claim_amount", "mean"),
            total_claim_amount=("claim_amount", "sum"),
            flagged_count=("is_top_100_anomaly", "sum"),
        ).reset_index().rename(columns={col: "period_value"})
        g.insert(0, "period_type", period_type)
        frames.append(g)

    temporal = pd.concat(frames, ignore_index=True)
    return temporal


# ---------------------------------------------------------------------------
# risk_segmentation: by L/M/H
# ---------------------------------------------------------------------------
def build_risk_segmentation(scores: pd.DataFrame) -> pd.DataFrame:
    order = ["L", "M", "H"]
    g = scores.groupby("risk_segmentation")
    risk = g.agg(
        claim_count=("transaction_id", "count"),
        approved_count=("claim_status", lambda s: (s == "A").sum()),
        avg_claim_amount=("claim_amount", "mean"),
        avg_premium_amount=("premium_amount", "mean"),
        flagged_count=("is_top_100_anomaly", "sum"),
        avg_ensemble_score=("ensemble_score", "mean"),
    ).reindex(order).reset_index()
    risk["approval_rate"] = risk["approved_count"] / risk["claim_count"]
    risk["flagged_rate"] = risk["flagged_count"] / risk["claim_count"]
    return risk


# ---------------------------------------------------------------------------
# Setup guide
# ---------------------------------------------------------------------------
def write_setup_guide(path: Path) -> None:
    guide = """POWER BI SETUP GUIDE
Financial Anomaly Detection Framework - Insurance Claims Dashboard
====================================================================

All 9 CSVs in this folder are ready to import as-is: Power BI Desktop ->
Get Data -> Text/CSV -> select all 9 files (or Get Data -> Folder to import
the whole output/powerbi_data/ directory at once).

--------------------------------------------------------------------
1. WHICH FILE FOR WHICH VISUAL
--------------------------------------------------------------------

fact_claims.csv  (10,000 rows - the backbone table)
    Use for: any visual that needs claim-level detail, drill-through pages,
    or custom measures (e.g. "% of claims over $50k").
    Recommended visuals: Table/Matrix (detail view), Scatter chart
    (claim_amount vs ensemble_score, colored by insurance_type), Card
    visuals built from DAX measures over this table.

claims_overview.csv  (6 rows, one per insurance_type)
    Recommended visuals: Clustered column chart (avg_claim_amount by
    insurance_type), 100% stacked bar (approved_count vs denied_count by
    insurance_type), Donut chart (total_claim_amount share by type).

kpi_summary.csv  (1 row - headline numbers)
    Recommended visuals: Card / KPI visuals only - one card per column
    (total_claims, overall_approval_rate, flagged_anomaly_count,
    total_claim_value). Place these at the top of the dashboard as the
    at-a-glance summary strip.

anomaly_dashboard.csv  (100 rows - the top-100 shortlist with explanations)
    Recommended visuals: Table visual sorted by `rank`, with `explanation`
    as a wide text column so adjusters can read why each claim was
    flagged. Pair with a Scatter chart (claim_amount vs ensemble_score)
    to visually separate the shortlist from the bulk of claims.

agent_performance.csv  (1,200 rows, one per agent)
    Recommended visuals: Scatter chart (avg_claim_amount on X, approval_rate
    on Y, size = claim_count) to spot agents that are outliers on both
    volume and approval behavior at once. Bar chart of Top N agents by
    claim_count or flagged_count.

vendor_performance.csv  (rows = vendors that appear on at least one claim)
    Recommended visuals: same pattern as agent_performance - scatter for
    outlier detection, bar chart for Top N by flagged_count. Note: ~32%
    of claims have no vendor at all (see PowerBI note below) - this table
    only covers claims that DO have one.

geographic_analysis.csv  (one row per state)
    Recommended visuals: Filled Map or Shape Map visual (state on
    Location, claim_count or avg_claim_amount on color saturation), plus a
    bar chart of Top 15 states by claim_count for a non-map alternative.

temporal_analysis.csv  (month + week rows, long format)
    IMPORTANT: this table stacks two different granularities in one file
    via the `period_type` column. Before building a chart, add a filter/
    slicer on period_type = "month" for a monthly trend line, or
    period_type = "week" for a weekly view - do NOT chart both at once,
    the x-axis values from each granularity will interleave incorrectly.
    Recommended visuals: Line chart (period_value on X, claim_count or
    avg_claim_amount on Y), one page per granularity.

risk_segmentation.csv  (3 rows: L / M / H)
    Recommended visuals: Clustered column chart (avg_claim_amount and
    approval_rate side by side, one series each, NOT on a dual axis - use
    two small charts or a small-multiples layout instead of a combo chart
    with two y-scales). Donut chart for claim_count share by segment.

--------------------------------------------------------------------
2. RECOMMENDED SLICERS (add these once, pin to every dashboard page)
--------------------------------------------------------------------
    - insurance_type       (fact_claims / claims_overview)
    - state                (fact_claims / geographic_analysis)
    - risk_segmentation    (fact_claims / risk_segmentation)
    - claim_status         (fact_claims)
    - is_top_100_anomaly   (fact_claims) - toggle "show flagged only"
      across every visual on the page in one click
    - Date range on loss_dt (fact_claims) - requires the Date table below

--------------------------------------------------------------------
3. DATA RELATIONSHIPS TO ESTABLISH (Model view)
--------------------------------------------------------------------
    fact_claims[agent_id]           -> agent_performance[agent_id]        (many-to-one)
    fact_claims[vendor_id]          -> vendor_performance[vendor_id]      (many-to-one;
                                        blank vendor_id claims will simply not match - expected)
    fact_claims[insurance_type]     -> claims_overview[insurance_type]    (many-to-one)
    fact_claims[risk_segmentation]  -> risk_segmentation[risk_segmentation] (many-to-one)
    fact_claims[state]              -> geographic_analysis[state]         (many-to-one)

    kpi_summary.csv and anomaly_dashboard.csv are standalone - kpi_summary
    has no join key (it's a single row of measures), and anomaly_dashboard
    is a curated shortlist rather than a dimension. Optionally relate
    anomaly_dashboard[agent_id] -> agent_performance[agent_id] if you want
    selecting an agent to filter the shortlist table.

    temporal_analysis.csv has no direct relationship to fact_claims (its
    period_value is a formatted string, not a real date). For proper
    calendar-aware filtering (quarter/year hierarchies, "last 30 days"
    slicers, etc.), enable Power BI's auto date/time on fact_claims[loss_dt]
    or build a dedicated Date table (Modeling -> New Table) and relate it
    to fact_claims[loss_dt] - then build monthly/weekly visuals directly
    off the fact table instead of temporal_analysis.csv where possible.

--------------------------------------------------------------------
4. KNOWN DATA CHARACTERISTICS TO KEEP IN MIND WHILE DASHBOARDING
--------------------------------------------------------------------
    - "Flagged" means "in the top 100 claims by ensemble anomaly score" -
      it is NOT a confirmed-fraud label. See 03_anomaly_detection.py's
      evaluation caveats before presenting this as a fraud rate to
      stakeholders.
    - state != incident_state on 93.6% of ALL claims in this dataset -
      it is the norm here, not a red flag by itself; don't build a single-
      metric "state mismatch = suspicious" card without the other signals.
    - vendor_id is blank on ~32% of claims by design (no vendor involved) -
      vendor_performance.csv only covers the other ~68%.
"""
    path.write_text(guide)
    logger.info("Saved -> %s", path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> int:
    setup_logging(LOG_PATH)
    logger.info("=== Preparing Power BI export ===")

    try:
        scores, top_anomalies, agents, vendors = load_data()
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    print_section("Building tables")
    tables = {
        "fact_claims.csv": build_fact_claims(scores),
        "claims_overview.csv": build_claims_overview(scores),
        "kpi_summary.csv": build_kpi_summary(scores, agents, vendors),
        "anomaly_dashboard.csv": build_anomaly_dashboard(top_anomalies, agents),
        "agent_performance.csv": build_agent_performance(scores, agents),
        "vendor_performance.csv": build_vendor_performance(scores, vendors),
        "geographic_analysis.csv": build_geographic_analysis(scores),
        "temporal_analysis.csv": build_temporal_analysis(scores),
        "risk_segmentation.csv": build_risk_segmentation(scores),
    }

    print_section("Exporting CSVs")
    for filename, df in tables.items():
        save_csv(df, filename)

    print_section("Writing setup guide")
    write_setup_guide(POWERBI_DIR / "PowerBI_SETUP.txt")

    logger.info(
        "=== Power BI export complete: %d CSV files + setup guide in %s ===",
        len(tables), POWERBI_DIR,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
