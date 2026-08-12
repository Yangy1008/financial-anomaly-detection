#!/usr/bin/env python3
"""
Exploratory data analysis over the cleaned insurance claims dataset produced
by 01_data_import.py.

Runs ten independent analysis sections (statistics, distributions,
correlations, customer segmentation, insurance-type breakdown, agent
performance, geography, temporal patterns, risk segmentation, outliers).
Each section is isolated with its own error handling so one failing section
does not prevent the rest of the report from being produced.

Output:
    output/eda_plots/*.png   - all charts
    output/eda_*.csv         - summary tables
    console + logs/          - printed statistics and insights
"""

import logging
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
PLOTS_DIR = OUTPUT_DIR / "eda_plots"
LOG_PATH = BASE_DIR / "logs" / "02_exploratory_analysis.log"

CLAIMS_PATH = OUTPUT_DIR / "insurance_claims_clean.parquet"
AGENTS_PATH = OUTPUT_DIR / "agents_clean.parquet"
VENDORS_PATH = OUTPUT_DIR / "vendors_clean.parquet"

logger = logging.getLogger("eda")

# ---------------------------------------------------------------------------
# Palette (validated categorical/diverging/status roles - see dataviz skill)
# Fixed hue order for identity encoding; never cycled or reassigned per-plot.
# ---------------------------------------------------------------------------
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS_LINE = "#c3c2b7"

CATEGORICAL = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]
DIVERGING_NEG, DIVERGING_MID, DIVERGING_POS = "#2a78d6", "#f0efec", "#e34948"
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "diverging_blue_red", [DIVERGING_NEG, DIVERGING_MID, DIVERGING_POS]
)
STATUS_CRITICAL = "#d03b3b"


def configure_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS_LINE,
        "axes.labelcolor": INK_SECONDARY,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK_PRIMARY,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "figure.dpi": 110,
    })


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
# I/O helpers
# ---------------------------------------------------------------------------
def load_data() -> tuple:
    for path in (CLAIMS_PATH, AGENTS_PATH, VENDORS_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run python/01_data_import.py first to generate it."
            )
    claims = pd.read_parquet(CLAIMS_PATH)
    agents = pd.read_parquet(AGENTS_PATH)
    vendors = pd.read_parquet(VENDORS_PATH)
    logger.info("Loaded %d claims, %d agents, %d vendors", len(claims), len(agents), len(vendors))
    return claims, agents, vendors


def save_fig(fig, filename: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot -> %s", out_path)


def save_csv(df: pd.DataFrame, filename: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / filename
    df.to_csv(out_path, index=df.index.name is not None)
    logger.info("Saved summary table -> %s (%d rows)", out_path, len(df))


def print_section(title: str) -> None:
    banner = f" {title} "
    logger.info("\n" + banner.center(78, "="))


# ---------------------------------------------------------------------------
# 1. Statistical summaries
# ---------------------------------------------------------------------------
def section_statistical_summary(claims: pd.DataFrame) -> pd.DataFrame:
    print_section("1. Statistical Summary: claim_amount, premium_amount, days_to_process")

    cols = ["claim_amount", "premium_amount", "days_to_process", "claim_to_premium_ratio"]
    stats = claims[cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    stats["skew"] = claims[cols].skew()
    stats["kurtosis"] = claims[cols].kurtosis()
    stats["missing"] = claims[cols].isna().sum()

    logger.info("\n%s", stats.round(2).to_string())
    save_csv(stats.round(4), "eda_summary_key_columns.csv")
    return stats


# ---------------------------------------------------------------------------
# 2. Distribution analysis
# ---------------------------------------------------------------------------
def section_distributions(claims: pd.DataFrame) -> None:
    print_section("2. Distribution Analysis: histograms and box plots")

    targets = ["claim_amount", "premium_amount", "days_to_process"]
    for col in targets:
        data = claims[col].dropna()
        fig, (ax_hist, ax_box) = plt.subplots(
            1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [3, 1]}
        )
        ax_hist.hist(data, bins=40, color=CATEGORICAL[0], edgecolor=SURFACE, linewidth=0.3)
        ax_hist.set_title(f"Distribution of {col}")
        ax_hist.set_xlabel(col)
        ax_hist.set_ylabel("count")

        box = ax_box.boxplot(
            data, vert=True, patch_artist=True,
            boxprops=dict(facecolor=CATEGORICAL[0], edgecolor=INK_SECONDARY),
            medianprops=dict(color=INK_PRIMARY, linewidth=1.5),
            flierprops=dict(marker="o", markerfacecolor=STATUS_CRITICAL,
                             markeredgecolor=STATUS_CRITICAL, markersize=3, alpha=0.5),
        )
        ax_box.set_title("Box plot")
        ax_box.set_xticks([])

        save_fig(fig, f"distribution_{col}.png")
        logger.info(
            "%s: mean=%.2f median=%.2f std=%.2f min=%.2f max=%.2f",
            col, data.mean(), data.median(), data.std(), data.min(), data.max(),
        )


# ---------------------------------------------------------------------------
# 3. Correlation analysis
# ---------------------------------------------------------------------------
def section_correlations(claims: pd.DataFrame) -> pd.DataFrame:
    print_section("3. Correlation Analysis")

    numeric_cols = [
        "claim_amount", "premium_amount", "age", "tenure", "no_of_family_members",
        "days_to_process", "claim_to_premium_ratio", "customer_claim_frequency",
        "incident_hour_of_the_day", "any_injury", "police_report_available",
    ]
    numeric_cols = [c for c in numeric_cols if c in claims.columns]
    corr = claims[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap=DIVERGING_CMAP, vmin=-1, vmax=1, center=0,
        square=True, linewidths=0.5, linecolor=SURFACE,
        cbar_kws={"label": "Pearson correlation"}, ax=ax,
        annot_kws={"size": 7, "color": INK_PRIMARY},
    )
    ax.set_title("Correlation Matrix - Numeric Features")
    save_fig(fig, "correlation_heatmap.png")

    strong = (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .rename("correlation")
        .reset_index()
        .rename(columns={"level_0": "feature_a", "level_1": "feature_b"})
    )
    strong = strong.reindex(strong["correlation"].abs().sort_values(ascending=False).index)
    logger.info("Top correlations (by |r|):\n%s", strong.head(10).round(3).to_string(index=False))

    save_csv(corr.round(4), "eda_correlation_matrix.csv")
    return corr


# ---------------------------------------------------------------------------
# 4. Customer segmentation
# ---------------------------------------------------------------------------
def section_customer_segmentation(claims: pd.DataFrame) -> pd.DataFrame:
    print_section("4. Customer Segmentation by Claim Frequency and Amount")

    cust = claims.groupby("customer_id").agg(
        claim_count=("transaction_id", "count"),
        total_claim_amount=("claim_amount", "sum"),
        avg_claim_amount=("claim_amount", "mean"),
    ).reset_index()

    freq_labels = ["Low", "Medium", "High"]
    amount_labels = ["Low", "Medium", "High"]
    # rank-based bins (duplicates='drop') so ties don't break qcut when many
    # customers share the same claim_count (e.g. everyone at exactly 1 claim)
    cust["frequency_segment"] = pd.qcut(
        cust["claim_count"].rank(method="first"), q=3, labels=freq_labels
    )
    cust["amount_segment"] = pd.qcut(
        cust["total_claim_amount"].rank(method="first"), q=3, labels=amount_labels
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    for i, seg in enumerate(freq_labels):
        subset = cust[cust["frequency_segment"] == seg]
        ax.scatter(
            subset["claim_count"], subset["total_claim_amount"],
            s=18, alpha=0.6, color=CATEGORICAL[i], label=f"{seg} frequency",
        )
    ax.set_xlabel("claim count per customer")
    ax.set_ylabel("total claim amount per customer")
    ax.set_title("Customer Segmentation: Frequency vs. Total Claim Amount")
    ax.legend(frameon=False, title="Segment")
    save_fig(fig, "customer_segmentation.png")

    segment_summary = cust.groupby(["frequency_segment", "amount_segment"], observed=True).agg(
        customers=("customer_id", "count"),
        avg_claim_count=("claim_count", "mean"),
        avg_total_claim_amount=("total_claim_amount", "mean"),
    ).reset_index()
    logger.info("Segment cross-tab:\n%s", segment_summary.round(2).to_string(index=False))

    save_csv(cust, "eda_customer_segments.csv")
    save_csv(segment_summary, "eda_customer_segment_summary.csv")
    return cust


# ---------------------------------------------------------------------------
# 5. Insurance type analysis
# ---------------------------------------------------------------------------
def section_insurance_type_analysis(claims: pd.DataFrame) -> pd.DataFrame:
    print_section("5. Insurance Type Analysis")

    by_type = claims.groupby("insurance_type").agg(
        claim_count=("transaction_id", "count"),
        avg_claim_amount=("claim_amount", "mean"),
        median_claim_amount=("claim_amount", "median"),
        avg_premium_amount=("premium_amount", "mean"),
        approval_rate=("claim_status", lambda s: (s == "A").mean()),
    ).sort_values("avg_claim_amount", ascending=False)
    logger.info("By insurance type:\n%s", by_type.round(2).to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    order = by_type.index.tolist()
    colors = [CATEGORICAL[i % len(CATEGORICAL)] for i in range(len(order))]
    ax.bar(order, by_type["avg_claim_amount"], color=colors, width=0.6)
    ax.set_ylabel("average claim amount")
    ax.set_title("Average Claim Amount by Insurance Type")
    ax.tick_params(axis="x", rotation=30)
    save_fig(fig, "insurance_type_avg_claim.png")

    severity_ct = pd.crosstab(claims["insurance_type"], claims["incident_severity"], normalize="index")
    fig, ax = plt.subplots(figsize=(9, 5))
    severity_ct = severity_ct[["Minor Loss", "Major Loss", "Total Loss"]]
    bottom = np.zeros(len(severity_ct))
    for i, severity in enumerate(severity_ct.columns):
        ax.bar(
            severity_ct.index, severity_ct[severity], bottom=bottom,
            color=CATEGORICAL[i], label=severity, width=0.6,
            edgecolor=SURFACE, linewidth=1,
        )
        bottom += severity_ct[severity].values
    ax.set_ylabel("share of claims")
    ax.set_title("Incident Severity Distribution by Insurance Type")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(frameon=False, title="Severity", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(fig, "insurance_type_severity.png")

    save_csv(by_type.reset_index(), "eda_summary_by_insurance_type.csv")
    return by_type


# ---------------------------------------------------------------------------
# 6. Agent performance metrics
# ---------------------------------------------------------------------------
def section_agent_performance(claims: pd.DataFrame, agents: pd.DataFrame) -> pd.DataFrame:
    print_section("6. Agent Performance Metrics")

    by_agent = claims.groupby("agent_id").agg(
        claim_count=("transaction_id", "count"),
        approval_rate=("claim_status", lambda s: (s == "A").mean()),
        avg_claim_amount=("claim_amount", "mean"),
        avg_claim_to_premium_ratio=("claim_to_premium_ratio", "mean"),
    ).reset_index()
    by_agent = by_agent.merge(agents[["agent_id", "agent_name"]], on="agent_id", how="left")

    logger.info(
        "Agent-level approval rate: mean=%.2f%%, std=%.2f%%, min=%.2f%%, max=%.2f%%",
        by_agent["approval_rate"].mean() * 100, by_agent["approval_rate"].std() * 100,
        by_agent["approval_rate"].min() * 100, by_agent["approval_rate"].max() * 100,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(by_agent["approval_rate"], bins=20, color=CATEGORICAL[0], edgecolor=SURFACE)
    axes[0].set_title("Distribution of Agent Approval Rates")
    axes[0].set_xlabel("approval rate")
    axes[0].xaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    axes[1].scatter(
        by_agent["avg_claim_amount"], by_agent["approval_rate"],
        s=14, alpha=0.5, color=CATEGORICAL[0],
    )
    axes[1].set_xlabel("average claim amount")
    axes[1].set_ylabel("approval rate")
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    axes[1].set_title("Avg Claim Amount vs. Approval Rate (per agent)")
    save_fig(fig, "agent_performance.png")

    top_by_volume = by_agent.sort_values("claim_count", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top_by_volume["agent_name"], top_by_volume["claim_count"], color=CATEGORICAL[0])
    ax.invert_yaxis()
    ax.set_xlabel("claim count")
    ax.set_title("Top 15 Agents by Claim Volume")
    save_fig(fig, "agent_top15_by_volume.png")

    save_csv(by_agent, "eda_summary_by_agent.csv")
    return by_agent


# ---------------------------------------------------------------------------
# 7. Geographic patterns
# ---------------------------------------------------------------------------
def section_geographic_patterns(claims: pd.DataFrame) -> pd.DataFrame:
    print_section("7. Geographic Patterns")

    by_state = claims.groupby("state").agg(
        claim_count=("transaction_id", "count"),
        avg_claim_amount=("claim_amount", "mean"),
        total_claim_amount=("claim_amount", "sum"),
    ).sort_values("claim_count", ascending=False)
    logger.info("Top 10 states by claim count:\n%s", by_state.head(10).round(2).to_string())

    mismatch = (claims["state"] != claims["incident_state"]).mean()
    logger.info("Claims where policyholder state != incident state: %.1f%%", mismatch * 100)

    top_states = by_state.head(15)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(top_states.index, top_states["claim_count"], color=CATEGORICAL[0], width=0.6)
    ax.set_ylabel("claim count")
    ax.set_title("Top 15 States by Claim Count")
    ax.tick_params(axis="x", rotation=45)
    save_fig(fig, "claims_by_state.png")

    save_csv(by_state.reset_index(), "eda_summary_by_state.csv")
    return by_state


# ---------------------------------------------------------------------------
# 8. Temporal patterns
# ---------------------------------------------------------------------------
def section_temporal_patterns(claims: pd.DataFrame) -> pd.DataFrame:
    print_section("8. Temporal Patterns")

    df = claims.copy()
    # Grouping by calendar month name (not year-month) would wrongly merge
    # June 2020 with June 2021 - this dataset spans 2020-05 through 2021-06.
    df["year_month"] = df["loss_dt"].dt.to_period("M").astype(str)
    df["day_of_week"] = df["loss_dt"].dt.day_name()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    by_month = df.groupby("year_month")["transaction_id"].count().rename("claim_count")
    by_dow = (
        df.groupby("day_of_week")["transaction_id"].count()
        .reindex(dow_order).rename("claim_count")
    )
    by_hour = df.groupby("incident_hour_of_the_day")["transaction_id"].count().rename("claim_count")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(by_month.index, by_month.values, color=CATEGORICAL[0], marker="o", markersize=3)
    axes[0].set_title("Claims by Month")
    axes[0].tick_params(axis="x", rotation=90)

    axes[1].bar(by_dow.index, by_dow.values, color=CATEGORICAL[0], width=0.6)
    axes[1].set_title("Claims by Day of Week")
    axes[1].tick_params(axis="x", rotation=45)

    axes[2].bar(by_hour.index, by_hour.values, color=CATEGORICAL[0], width=0.7)
    axes[2].set_title("Claims by Hour of Day")
    axes[2].set_xlabel("hour")
    save_fig(fig, "temporal_patterns.png")

    logger.info("Peak month: %s (%d claims)", by_month.idxmax(), by_month.max())
    logger.info("Peak day of week: %s (%d claims)", by_dow.idxmax(), by_dow.max())
    logger.info("Peak hour: %d:00 (%d claims)", by_hour.idxmax(), by_hour.max())

    temporal = pd.concat([
        by_month.rename_axis("period_value").reset_index().assign(period_type="month"),
        by_dow.rename_axis("period_value").reset_index().assign(period_type="day_of_week"),
        by_hour.rename_axis("period_value").reset_index().assign(period_type="hour"),
    ], ignore_index=True)[["period_type", "period_value", "claim_count"]]
    save_csv(temporal, "eda_temporal_patterns.csv")
    return temporal


# ---------------------------------------------------------------------------
# 9. Risk segmentation analysis
# ---------------------------------------------------------------------------
def section_risk_segmentation(claims: pd.DataFrame) -> pd.DataFrame:
    print_section("9. Risk Segmentation Analysis (L/M/H)")

    order = ["L", "M", "H"]
    by_risk = claims.groupby("risk_segmentation").agg(
        claim_count=("transaction_id", "count"),
        avg_claim_amount=("claim_amount", "mean"),
        avg_premium_amount=("premium_amount", "mean"),
        avg_claim_to_premium_ratio=("claim_to_premium_ratio", "mean"),
        approval_rate=("claim_status", lambda s: (s == "A").mean()),
    ).reindex(order)
    logger.info("By risk segmentation:\n%s", by_risk.round(2).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(order, by_risk["avg_claim_amount"], color=CATEGORICAL[:3], width=0.5)
    axes[0].set_title("Average Claim Amount by Risk Segment")
    axes[0].set_xlabel("risk segmentation (Low / Medium / High)")

    axes[1].bar(order, by_risk["approval_rate"], color=CATEGORICAL[:3], width=0.5)
    axes[1].set_title("Approval Rate by Risk Segment")
    axes[1].set_xlabel("risk segmentation (Low / Medium / High)")
    axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    save_fig(fig, "risk_segmentation.png")

    save_csv(by_risk.reset_index(), "eda_summary_by_risk_segment.csv")
    return by_risk


# ---------------------------------------------------------------------------
# 10. Outlier identification
# ---------------------------------------------------------------------------
def iqr_outlier_mask(series: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return (series < lower) | (series > upper)


def section_outliers(claims: pd.DataFrame) -> pd.DataFrame:
    print_section("10. Outlier Identification")

    df = claims.copy()
    df["outlier_claim_amount"] = iqr_outlier_mask(df["claim_amount"])
    df["outlier_ratio"] = False
    ratio_notna = df["claim_to_premium_ratio"].notna()
    df.loc[ratio_notna, "outlier_ratio"] = iqr_outlier_mask(df.loc[ratio_notna, "claim_to_premium_ratio"])
    df["is_outlier"] = df["outlier_claim_amount"] | df["outlier_ratio"]

    n_outliers = df["is_outlier"].sum()
    logger.info(
        "Outliers (1.5x IQR rule): %d claim_amount, %d claim_to_premium_ratio, %d combined (%.1f%% of claims)",
        df["outlier_claim_amount"].sum(), df["outlier_ratio"].sum(), n_outliers,
        100 * n_outliers / len(df),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].boxplot(
        df["claim_amount"].dropna(), vert=True, patch_artist=True,
        boxprops=dict(facecolor=CATEGORICAL[0], edgecolor=INK_SECONDARY),
        medianprops=dict(color=INK_PRIMARY, linewidth=1.5),
        flierprops=dict(marker="o", markerfacecolor=STATUS_CRITICAL,
                         markeredgecolor=STATUS_CRITICAL, markersize=4, alpha=0.6),
    )
    axes[0].set_xticks([])
    axes[0].set_ylabel("claim_amount")
    axes[0].set_title("Claim Amount - Outliers Flagged")

    normal = df[~df["is_outlier"]]
    flagged = df[df["is_outlier"]]
    axes[1].scatter(normal["claim_amount"], normal["claim_to_premium_ratio"],
                     s=10, alpha=0.35, color=CATEGORICAL[0], label="normal")
    axes[1].scatter(flagged["claim_amount"], flagged["claim_to_premium_ratio"],
                     s=16, alpha=0.8, color=STATUS_CRITICAL, label="outlier")
    axes[1].set_xlabel("claim_amount")
    axes[1].set_ylabel("claim_to_premium_ratio")
    axes[1].set_title("Outliers: Claim Amount vs. Claim-to-Premium Ratio")
    axes[1].legend(frameon=False)
    save_fig(fig, "outliers.png")

    outlier_cols = [
        "transaction_id", "customer_id", "agent_id", "vendor_id", "insurance_type",
        "claim_amount", "premium_amount", "claim_to_premium_ratio",
        "claim_status", "incident_severity", "outlier_claim_amount", "outlier_ratio",
    ]
    top_outliers = flagged[outlier_cols].sort_values("claim_amount", ascending=False)
    save_csv(top_outliers, "eda_outliers.csv")
    return top_outliers


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_section(name: str, func, *args) -> bool:
    try:
        func(*args)
        return True
    except Exception:
        logger.error("Section '%s' failed - continuing with remaining sections", name)
        logger.debug(traceback.format_exc())
        return False


def main() -> int:
    setup_logging(LOG_PATH)
    configure_style()
    logger.info("=== Starting exploratory data analysis ===")

    try:
        claims, agents, vendors = load_data()
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    sections = [
        ("statistical_summary", section_statistical_summary, (claims,)),
        ("distributions", section_distributions, (claims,)),
        ("correlations", section_correlations, (claims,)),
        ("customer_segmentation", section_customer_segmentation, (claims,)),
        ("insurance_type_analysis", section_insurance_type_analysis, (claims,)),
        ("agent_performance", section_agent_performance, (claims, agents)),
        ("geographic_patterns", section_geographic_patterns, (claims,)),
        ("temporal_patterns", section_temporal_patterns, (claims,)),
        ("risk_segmentation", section_risk_segmentation, (claims,)),
        ("outliers", section_outliers, (claims,)),
    ]

    results = {name: run_section(name, func, *args) for name, func, args in sections}

    failed = [name for name, ok in results.items() if not ok]
    if failed:
        logger.warning("Completed with %d failed section(s): %s", len(failed), failed)
    else:
        logger.info("All %d sections completed successfully", len(sections))

    logger.info("=== Exploratory data analysis finished. Plots: %s | Tables: %s ===", PLOTS_DIR, OUTPUT_DIR)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
