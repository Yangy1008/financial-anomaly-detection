#!/usr/bin/env python3
"""
Build the insurance claims SQLite database from source CSVs, load it back,
clean it, engineer analysis features, validate data quality, and export
analysis-ready Parquet files.

Pipeline stages:
    0. Build/refresh the SQLite database from the source CSVs (sql/schema.sql)
    1. Load all three tables from SQLite into DataFrames
    2. Clean (dtypes, missing values, drop/consolidate unnecessary columns)
    3. Engineer derived features on insurance_claims
    4. Validate data quality
    5. Export cleaned/enriched tables to Parquet
"""

import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SCHEMA_PATH = BASE_DIR / "sql" / "schema.sql"
DB_PATH = BASE_DIR / "data" / "insurance_claims.db"
OUTPUT_DIR = BASE_DIR / "output"
LOG_PATH = BASE_DIR / "logs" / "01_data_import.log"

CSV_SOURCES = {
    "agents": BASE_DIR / "employee_data.csv",
    "vendors": BASE_DIR / "vendor_data.csv",
    "insurance_claims": BASE_DIR / "insurance_data.csv",
}

DATE_COLUMNS = {
    "agents": ["date_of_joining"],
    "vendors": [],
    "insurance_claims": ["txn_date_time", "policy_eff_dt", "loss_dt", "report_dt"],
}

# Columns dropped entirely: sensitive PII/banking data not needed for
# anomaly analytics, plus street-level address detail that's too granular
# to be useful (city/state/postal_code are retained for geo analysis).
COLUMNS_TO_DROP = {
    "agents": ["address_line1", "address_line2", "emp_routing_number", "emp_acct_number"],
    "vendors": ["address_line1", "address_line2"],
    "insurance_claims": [
        "address_line1", "address_line2", "ssn", "routing_number", "acct_number",
    ],
}

# pandas' default na_values list includes the literal strings 'NA' and
# 'None', which are legitimate category labels in this dataset
# (CUSTOMER_EDUCATION_LEVEL='NA', AUTHORITY_CONTACTED='None'). Only a truly
# empty cell should be treated as missing.
NA_VALUES = [""]

REQUIRED_CLAIM_COLUMNS = [
    "transaction_id", "customer_id", "policy_number",
    "loss_dt", "report_dt", "claim_amount", "premium_amount", "agent_id",
]

VALID_CATEGORIES = {
    "insurance_type": {"Property", "Mobile", "Health", "Life", "Travel", "Motor"},
    "claim_status": {"A", "D"},
    "risk_segmentation": {"L", "M", "H"},
    "incident_severity": {"Total Loss", "Major Loss", "Minor Loss"},
    "authority_contacted": {"Police", "Ambulance", "Other", "None"},
}

logger = logging.getLogger("data_import")


class DataQualityError(Exception):
    """Raised when validation finds a critical, unrecoverable data problem."""


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Stage 0: build the SQLite database from source CSVs
# ---------------------------------------------------------------------------
def build_database(db_path: Path, schema_path: Path) -> None:
    logger.info("Building database at %s from %s", db_path, schema_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    schema_sql = schema_path.read_text()

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(schema_sql)  # schema.sql drops/recreates each table
        conn.commit()

        # Load agents and vendors before insurance_claims so that FK
        # constraints on agent_id/vendor_id are satisfiable on insert.
        for table in ("agents", "vendors", "insurance_claims"):
            csv_path = CSV_SOURCES[table]
            if not csv_path.exists():
                raise FileNotFoundError(f"Source CSV not found: {csv_path}")

            df = pd.read_csv(csv_path, keep_default_na=False, na_values=NA_VALUES)
            df.columns = [c.lower() for c in df.columns]
            df.to_sql(table, conn, if_exists="append", index=False)
            logger.info("Loaded %d rows into '%s' from %s", len(df), table, csv_path.name)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Stage 1: load from SQLite
# ---------------------------------------------------------------------------
def load_from_database(db_path: Path) -> dict:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    logger.info("Loading tables from %s", db_path)
    conn = sqlite3.connect(db_path)
    try:
        tables = {}
        for table in ("agents", "vendors", "insurance_claims"):
            tables[table] = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            logger.info("Loaded %d rows, %d columns from '%s'",
                        len(tables[table]), tables[table].shape[1], table)
    finally:
        conn.close()
    return tables


# ---------------------------------------------------------------------------
# Stage 2: cleaning
# ---------------------------------------------------------------------------
def clean_table(name: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Convert date columns to datetime ---
    for col in DATE_COLUMNS.get(name, []):
        if col in df.columns:
            before_na = df[col].isna().sum()
            df[col] = pd.to_datetime(df[col], errors="coerce")
            after_na = df[col].isna().sum()
            if after_na > before_na:
                logger.warning(
                    "[%s] %d values in '%s' could not be parsed as dates and became NaT",
                    name, after_na - before_na, col,
                )

    # --- Consolidate address into a single field, then drop raw columns ---
    if "address_line1" in df.columns:
        line1 = df["address_line1"].fillna("")
        line2 = df.get("address_line2", pd.Series("", index=df.index)).fillna("")
        df["address"] = (line1 + " " + line2).str.strip().replace("", np.nan)

    drop_cols = [c for c in COLUMNS_TO_DROP.get(name, []) if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.info("[%s] dropped unnecessary/sensitive columns: %s", name, drop_cols)

    # --- Handle missing values ---
    for col in ("city", "incident_city"):
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing:
                df[col] = df[col].fillna("Unknown")
                logger.info("[%s] filled %d missing '%s' values with 'Unknown'", name, n_missing, col)

    # vendor_id is intentionally left null where present: an absent vendor
    # is a meaningful business state (no vendor involved), not missing data.
    if name == "insurance_claims" and "vendor_id" in df.columns:
        n_no_vendor = df["vendor_id"].isna().sum()
        logger.info("[%s] %d claims have no vendor assigned (left as NULL)", name, n_no_vendor)

    # Any remaining nulls in numeric columns: impute with the column median
    # and log it, since none are expected from the profiled source data.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        n_missing = df[col].isna().sum()
        if n_missing:
            median = df[col].median()
            df[col] = df[col].fillna(median)
            logger.warning("[%s] imputed %d missing '%s' values with median=%s", name, n_missing, col, median)

    return df


# ---------------------------------------------------------------------------
# Stage 3: feature engineering
# ---------------------------------------------------------------------------
def engineer_features(claims: pd.DataFrame) -> pd.DataFrame:
    claims = claims.copy()

    # days_to_process: turnaround time between loss and report
    claims["days_to_process"] = (claims["report_dt"] - claims["loss_dt"]).dt.days
    n_negative = (claims["days_to_process"] < 0).sum()
    if n_negative:
        logger.warning(
            "%d claims report_dt precede loss_dt (negative days_to_process) - "
            "left in place for anomaly review, not corrected",
            n_negative,
        )

    # claim_to_premium_ratio: guard against premium_amount == 0
    zero_premium = (claims["premium_amount"] == 0).sum()
    if zero_premium:
        logger.warning("%d claims have premium_amount == 0; claim_to_premium_ratio set to NaN", zero_premium)
    claims["claim_to_premium_ratio"] = np.where(
        claims["premium_amount"] > 0,
        claims["claim_amount"] / claims["premium_amount"],
        np.nan,
    )

    # customer_claim_frequency: total claim count for that customer_id,
    # attached to every row belonging to that customer
    claims["customer_claim_frequency"] = claims.groupby("customer_id")["transaction_id"].transform("count")

    logger.info(
        "Engineered features: days_to_process (median=%.1f), "
        "claim_to_premium_ratio (median=%.2f), customer_claim_frequency (max=%d)",
        claims["days_to_process"].median(),
        claims["claim_to_premium_ratio"].median(),
        claims["customer_claim_frequency"].max(),
    )
    return claims


# ---------------------------------------------------------------------------
# Stage 4: validation
# ---------------------------------------------------------------------------
def validate_data(claims: pd.DataFrame, agents: pd.DataFrame, vendors: pd.DataFrame) -> None:
    errors = []
    warnings = []

    # Required fields present and non-null
    for col in REQUIRED_CLAIM_COLUMNS:
        if col not in claims.columns:
            errors.append(f"required column '{col}' missing from insurance_claims")
        elif claims[col].isna().any():
            errors.append(f"required column '{col}' contains {claims[col].isna().sum()} null value(s)")

    # Primary key uniqueness
    dupes = claims["transaction_id"].duplicated().sum()
    if dupes:
        errors.append(f"{dupes} duplicate transaction_id value(s) found")

    # Value ranges
    if (claims["claim_amount"] < 0).any():
        errors.append("negative claim_amount value(s) found")
    if (claims["premium_amount"] < 0).any():
        errors.append("negative premium_amount value(s) found")
    if (claims["days_to_process"] < 0).any():
        warnings.append(f"{(claims['days_to_process'] < 0).sum()} claim(s) reported before the loss date")

    # Referential integrity
    unknown_agents = ~claims["agent_id"].isin(agents["agent_id"])
    if unknown_agents.any():
        errors.append(f"{unknown_agents.sum()} claim(s) reference an agent_id not present in agents")

    vendor_mask = claims["vendor_id"].notna()
    unknown_vendors = vendor_mask & ~claims["vendor_id"].isin(vendors["vendor_id"])
    if unknown_vendors.any():
        errors.append(f"{unknown_vendors.sum()} claim(s) reference a vendor_id not present in vendors")

    # Category domains
    for col, allowed in VALID_CATEGORIES.items():
        if col not in claims.columns:
            continue
        invalid = set(claims[col].dropna().unique()) - allowed
        if invalid:
            errors.append(f"column '{col}' contains unexpected categories: {sorted(invalid)}")

    # Row-count sanity check against expected data volumes
    expected_counts = {"insurance_claims": 10_000, "agents": 1_200, "vendors": 600}
    actual_counts = {"insurance_claims": len(claims), "agents": len(agents), "vendors": len(vendors)}
    for table, expected in expected_counts.items():
        actual = actual_counts[table]
        if actual < 0.9 * expected:
            warnings.append(f"'{table}' has {actual} rows, well below the expected ~{expected}")

    for w in warnings:
        logger.warning("Data quality warning: %s", w)

    if errors:
        for e in errors:
            logger.error("Data quality error: %s", e)
        raise DataQualityError(f"{len(errors)} critical data quality issue(s) found; see log for details")

    logger.info("Validation passed: %d claims, %d agents, %d vendors, 0 critical issues",
                actual_counts["insurance_claims"], actual_counts["agents"], actual_counts["vendors"])


# ---------------------------------------------------------------------------
# Stage 5: export
# ---------------------------------------------------------------------------
def export_to_parquet(tables: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        out_path = output_dir / f"{name}_clean.parquet"
        df.to_parquet(out_path, engine="pyarrow", index=False, compression="snappy")
        logger.info("Exported '%s' -> %s (%d rows, %.1f KB)",
                    name, out_path, len(df), out_path.stat().st_size / 1024)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> int:
    setup_logging(LOG_PATH)
    logger.info("=== Starting insurance claims data import pipeline ===")

    try:
        build_database(DB_PATH, SCHEMA_PATH)
        raw = load_from_database(DB_PATH)

        cleaned = {name: clean_table(name, df) for name, df in raw.items()}
        cleaned["insurance_claims"] = engineer_features(cleaned["insurance_claims"])

        validate_data(cleaned["insurance_claims"], cleaned["agents"], cleaned["vendors"])

        export_to_parquet(cleaned, OUTPUT_DIR)

    except DataQualityError as e:
        logger.error("Pipeline halted due to data quality issues: %s", e)
        return 1
    except FileNotFoundError as e:
        logger.error("Pipeline halted - missing file: %s", e)
        return 1
    except Exception:
        logger.exception("Pipeline failed with an unexpected error")
        return 1

    logger.info("=== Pipeline completed successfully ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
