#!/usr/bin/env python3
"""
Core anomaly detection module for the insurance claims dataset produced by
01_data_import.py / 02_exploratory_analysis.py.

Implements four independent detection methods, evaluates each against a
proxy label, combines them into a weighted ensemble score, and exports the
scored dataset, a top-500 anomaly shortlist with explanations, and a
method comparison report.

*** IMPORTANT CAVEAT ABOUT EVALUATION ***
This dataset has no verified fraud/anomaly ground truth. Per the task spec,
`incident_severity == 'Total Loss'` is used as a PROXY label purely so
precision/recall/F1/ROC-AUC are computable and the four methods can be
compared on a common yardstick. A "Total Loss" claim is not necessarily
fraudulent or even suspicious - it is a legitimate severity category that
~34% of claims fall into. The metrics below therefore measure "agreement
with high severity", not "fraud detection accuracy". Treat them as relative
method comparisons, not real-world performance estimates.

Output:
    output/anomaly_scores.parquet   - every claim with all 4 method scores + ensemble score
    output/top_anomalies.csv        - top 500 by ensemble score, with plain-language explanations
    output/method_comparison.csv    - precision/recall/F1/ROC-AUC/confusion matrix per method
    output/anomaly_plots/*.png      - score distributions, breakdowns, ROC curves, confusion matrices
    logs/03_anomaly_detection.log   - detailed run log
"""

import logging
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
PLOTS_DIR = OUTPUT_DIR / "anomaly_plots"
LOG_PATH = BASE_DIR / "logs" / "03_anomaly_detection.log"

CLAIMS_PATH = OUTPUT_DIR / "insurance_claims_clean.parquet"
AGENTS_PATH = OUTPUT_DIR / "agents_clean.parquet"

RANDOM_STATE = 42          # fixes IsolationForest's internal sampling so results are reproducible
ASSUMED_ANOMALY_RATE = 0.05  # business assumption: ~5% of claims warrant investigation
TOP_N = 500

# Ensemble weights. Statistical/Isolation Forest/LOF get roughly equal say;
# rule_based is intentionally down-weighted - see method_rule_based() for
# why the state-mismatch component is far less selective than expected.
ENSEMBLE_WEIGHTS = {
    "zscore": 0.30,
    "isolation_forest": 0.30,
    "lof": 0.25,
    "rule_based": 0.15,
}

logger = logging.getLogger("anomaly_detection")

# ---------------------------------------------------------------------------
# Palette (validated categorical/status roles - see dataviz skill)
# ---------------------------------------------------------------------------
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, GRID, AXIS_LINE = "#fcfcfb", "#e1e0d9", "#c3c2b7"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
METHOD_COLORS = {
    "zscore": CATEGORICAL[0],
    "isolation_forest": CATEGORICAL[1],
    "lof": CATEGORICAL[2],
    "rule_based": CATEGORICAL[3],
    "ensemble": CATEGORICAL[6],
}
METHOD_LABELS = {
    "zscore": "Z-score",
    "isolation_forest": "Isolation Forest",
    "lof": "Local Outlier Factor",
    "rule_based": "Rule-based",
    "ensemble": "Ensemble",
}


def configure_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS_LINE, "axes.labelcolor": INK_SECONDARY, "text.color": INK_PRIMARY,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.grid": True, "axes.axisbelow": True, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlecolor": INK_PRIMARY, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "figure.dpi": 110,
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


def print_section(title: str) -> None:
    logger.info("\n" + f" {title} ".center(78, "="))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_data() -> tuple:
    for path in (CLAIMS_PATH, AGENTS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run 01_data_import.py first.")
    claims = pd.read_parquet(CLAIMS_PATH)
    agents = pd.read_parquet(AGENTS_PATH)
    logger.info("Loaded %d claims, %d agents", len(claims), len(agents))
    return claims, agents


def save_fig(fig, filename: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / filename
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved plot -> %s", out_path)


def minmax_scale_series(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if hi - lo < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Proxy label
# ---------------------------------------------------------------------------
def build_proxy_label(claims: pd.DataFrame) -> pd.Series:
    label = (claims["incident_severity"] == "Total Loss").astype(int)
    logger.info(
        "Proxy label built from incident_severity == 'Total Loss': %d positive / %d total (%.1f%%)",
        label.sum(), len(label), 100 * label.mean(),
    )
    return label


# ---------------------------------------------------------------------------
# Method 1: Statistical - Z-score
# ---------------------------------------------------------------------------
def method_zscore(claims: pd.DataFrame) -> pd.DataFrame:
    """
    Statistical method: per-insurance-type Z-score on claim_amount.

    Z = (x - group_mean) / group_std measures how many standard deviations
    a claim sits from the AVERAGE CLAIM OF ITS OWN INSURANCE TYPE. A global
    (ungrouped) z-score would just rediscover "this is a Life policy" -
    Life claims average ~$54k vs Mobile ~$400 - rather than "this claim is
    unusual for its type", so we standardize within each insurance_type
    group. |z| > 3 is the classic three-sigma rule of thumb for a
    statistical outlier under an assumed-normal distribution.
    """
    group_mean = claims.groupby("insurance_type")["claim_amount"].transform("mean")
    group_std = claims.groupby("insurance_type")["claim_amount"].transform("std").replace(0, np.nan)
    z = ((claims["claim_amount"] - group_mean) / group_std).fillna(0)

    flag = z.abs() > 3
    # Linear 0-1 mapping: the |z|=3 flag threshold sits at score 0.5, giving
    # headroom up to |z|=6 (score 1.0) so the ensemble can still rank
    # severity among flagged claims rather than saturating them all to 1.0.
    score = (z.abs() / 6).clip(0, 1)

    out = claims[["transaction_id"]].copy()
    out["claim_zscore"] = z
    out["zscore_flag"] = flag
    out["zscore_score"] = score
    logger.info("Z-score: %d claims flagged (|z| > 3), %.1f%% of dataset", flag.sum(), 100 * flag.mean())
    logger.info("Z-score: max |z| observed = %.2f (per insurance_type)", z.abs().max())
    if flag.sum() == 0:
        logger.warning(
            "Z-score flagged 0 claims: within every insurance_type, claim_amount never "
            "exceeds ~%.1fσ from its group mean. This dataset's claim amounts look "
            "bounded/near-uniform per type rather than heavy-tailed like real-world claims "
            "data - the classic 3-sigma rule genuinely finds nothing here, it is not a bug.",
            z.abs().max(),
        )
    return out


# ---------------------------------------------------------------------------
# Method 2: Isolation Forest over temporal + amount features
# ---------------------------------------------------------------------------
def method_isolation_forest(claims: pd.DataFrame) -> pd.DataFrame:
    """
    Isolation Forest applied to temporal and financial-magnitude features:
    days_since_start (position in the observed timeline), days_to_process,
    incident_hour_of_the_day, claim_amount, premium_amount, and
    claim_to_premium_ratio.

    Isolation Forest is not a sequential/autocorrelation-aware time-series
    model (it does not model trend or seasonality the way ARIMA would) -
    each claim is treated as an independent record, which is appropriate
    here since claims from unrelated customers have no reason to be
    autocorrelated. It isolates anomalies by recursively splitting on
    random features/thresholds: outliers sit in sparse regions of feature
    space and need far fewer splits to isolate than typical points, so a
    short average path length across the forest's trees signals anomaly.
    Using temporal + amount features together lets it flag claims that are
    processed unusually (fast/slow, odd hour) in combination with unusual
    amounts - patterns a single-column z-score can't see.
    """
    features = claims[[
        "loss_dt", "days_to_process", "incident_hour_of_the_day",
        "claim_amount", "premium_amount", "claim_to_premium_ratio",
    ]].copy()
    features["days_since_start"] = (features["loss_dt"] - features["loss_dt"].min()).dt.days
    X = features.drop(columns=["loss_dt"])
    X_scaled = StandardScaler().fit_transform(X)

    model = IsolationForest(
        n_estimators=200, contamination=ASSUMED_ANOMALY_RATE, random_state=RANDOM_STATE
    )
    pred = model.fit_predict(X_scaled)          # -1 = outlier, 1 = inlier
    raw = -model.score_samples(X_scaled)         # higher score_samples = more normal, so negate

    out = claims[["transaction_id"]].copy()
    out["iso_flag"] = pred == -1
    out["iso_score"] = minmax_scale_series(pd.Series(raw, index=claims.index))
    logger.info(
        "Isolation Forest: %d claims flagged (contamination=%.0f%%), features=%s",
        out["iso_flag"].sum(), ASSUMED_ANOMALY_RATE * 100, list(X.columns),
    )
    return out


# ---------------------------------------------------------------------------
# Method 3: Local Outlier Factor over customer + claim profile features
# ---------------------------------------------------------------------------
def method_lof(claims: pd.DataFrame) -> pd.DataFrame:
    """
    Local Outlier Factor (LOF) applied to a broader multi-dimensional
    profile: age, tenure, no_of_family_members, claim_amount,
    premium_amount, claim_to_premium_ratio, days_to_process, any_injury,
    police_report_available.

    Unlike Isolation Forest's temporal focus, LOF compares each claim's
    LOCAL density of neighbors (in this multi-feature space) to the local
    density of its neighbors' neighbors. A point in a much sparser
    neighborhood than its neighbors gets a high outlier factor - this
    catches claims that are only unusual in COMBINATION (e.g. a young,
    short-tenure customer with an unusually large claim relative to
    premium) even when no single feature is extreme on its own, which is
    exactly the kind of complex, multi-dimensional pattern a univariate
    z-score misses. Distance-based, so all features are standardized first.
    """
    feature_cols = [
        "age", "tenure", "no_of_family_members", "claim_amount", "premium_amount",
        "claim_to_premium_ratio", "days_to_process", "any_injury", "police_report_available",
    ]
    X = claims[feature_cols]
    X_scaled = StandardScaler().fit_transform(X)

    model = LocalOutlierFactor(n_neighbors=20, contamination=ASSUMED_ANOMALY_RATE)
    pred = model.fit_predict(X_scaled)                       # -1 = outlier, 1 = inlier
    raw = -model.negative_outlier_factor_                     # more negative = more abnormal, so negate

    out = claims[["transaction_id"]].copy()
    out["lof_flag"] = pred == -1
    out["lof_score"] = minmax_scale_series(pd.Series(raw, index=claims.index))
    logger.info(
        "LOF: %d claims flagged (contamination=%.0f%%), features=%s",
        out["lof_flag"].sum(), ASSUMED_ANOMALY_RATE * 100, feature_cols,
    )
    return out


# ---------------------------------------------------------------------------
# Method 4: Rule-based business logic
# ---------------------------------------------------------------------------
def method_rule_based(claims: pd.DataFrame) -> pd.DataFrame:
    """
    Two hand-written business rules, OR'd into a single flag:
      1. claim_to_premium_ratio > 10  - the claim is worth more than 10x
         the annual premium paid for it, a classic "too good to be true"
         red flag adjusters use.
      2. state != incident_state      - the policyholder's address state
         differs from where the incident occurred.

    CAVEAT: in this dataset BOTH rules are saturated. State mismatch fires
    on 93.6% of claims (confirmed during EDA - plausibly because
    Travel/Motor claims are often filed away from home), and claim_amount
    is structurally so much larger than a single premium payment that
    claim_to_premium_ratio > 10 fires on ~97.6% of claims too (even the 5th
    percentile ratio is ~12). Neither rule is a rare-event signal here, so
    the OR'd rule_flag ends up firing on almost every claim - a "flag
    everything" degenerate result, not a bug. It is included (per spec) but
    weighted at 70/30 (ratio/state) in the composite rule_score below, and
    down-weighted overall in the ensemble - see ENSEMBLE_WEIGHTS.
    """
    ratio_rule = claims["claim_to_premium_ratio"] > 10
    state_mismatch_rule = claims["state"] != claims["incident_state"]

    out = claims[["transaction_id"]].copy()
    out["ratio_rule"] = ratio_rule
    out["state_mismatch_rule"] = state_mismatch_rule
    out["rule_flag"] = ratio_rule | state_mismatch_rule
    out["rule_score"] = 0.7 * ratio_rule.astype(float) + 0.3 * state_mismatch_rule.astype(float)

    logger.info(
        "Rule-based: ratio>10 flags %d (%.1f%%), state-mismatch flags %d (%.1f%%), "
        "combined OR flags %d (%.1f%%)",
        ratio_rule.sum(), 100 * ratio_rule.mean(),
        state_mismatch_rule.sum(), 100 * state_mismatch_rule.mean(),
        out["rule_flag"].sum(), 100 * out["rule_flag"].mean(),
    )
    for rule_name, rule_mask in [("claim_to_premium_ratio > 10", ratio_rule),
                                  ("state ≠ incident_state", state_mismatch_rule)]:
        if rule_mask.mean() > 0.5:
            logger.warning(
                "Rule '%s' fires on %.1f%% of claims in this dataset - it is saturated, "
                "not a rare-event signal, and contributes little discriminative power alone.",
                rule_name, 100 * rule_mask.mean(),
            )
    if out["rule_flag"].mean() > 0.9:
        logger.warning(
            "Combined rule-based flag fires on %.1f%% of claims - as an OR of two "
            "near-saturated rules this is close to a 'flag everything' strategy on this "
            "dataset. Its high recall / low precision against the proxy label reflects "
            "that, not genuine discriminative skill.",
            100 * out["rule_flag"].mean(),
        )
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_method(name: str, y_true: pd.Series, y_score: pd.Series, y_pred: pd.Series) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = np.nan

    logger.info(
        "[%s] precision=%.3f recall=%.3f f1=%.3f roc_auc=%.3f | "
        "confusion matrix: TN=%d FP=%d FN=%d TP=%d | flagged=%d (%.1f%%)",
        METHOD_LABELS[name], precision, recall, f1, roc_auc, tn, fp, fn, tp,
        int(y_pred.sum()), 100 * y_pred.mean(),
    )
    return {
        "method": METHOD_LABELS[name], "precision": precision, "recall": recall, "f1": f1,
        "roc_auc": roc_auc, "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "n_flagged": int(y_pred.sum()), "flagged_rate": float(y_pred.mean()),
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def plot_score_distributions(scored: pd.DataFrame) -> None:
    methods = ["zscore", "isolation_forest", "lof", "rule_based", "ensemble"]
    score_cols = {"zscore": "zscore_score", "isolation_forest": "iso_score",
                  "lof": "lof_score", "rule_based": "rule_score", "ensemble": "ensemble_score"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, m in enumerate(methods):
        axes[i].hist(scored[score_cols[m]], bins=30, color=METHOD_COLORS[m], edgecolor=SURFACE)
        axes[i].set_title(f"{METHOD_LABELS[m]} score distribution")
        axes[i].set_xlabel("anomaly score (0-1)")
    axes[-1].axis("off")
    save_fig(fig, "score_distributions.png")


def plot_roc_curves(scored: pd.DataFrame, y_true: pd.Series) -> None:
    score_cols = {"zscore": "zscore_score", "isolation_forest": "iso_score",
                  "lof": "lof_score", "rule_based": "rule_score"}
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for m, col in score_cols.items():
        fpr, tpr, _ = roc_curve(y_true, scored[col])
        auc = roc_auc_score(y_true, scored[col])
        ax.plot(fpr, tpr, color=METHOD_COLORS[m], linewidth=2,
                label=f"{METHOD_LABELS[m]} (AUC={auc:.2f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color=INK_MUTED, linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves vs. Proxy Label (incident_severity = Total Loss)")
    ax.legend(frameon=False, loc="lower right")
    save_fig(fig, "roc_curves.png")


def plot_confusion_matrices(comparison_rows: list) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    for ax, row in zip(axes, comparison_rows):
        cm = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"],
            annot_kws={"size": 11},
        )
        ax.set_title(row["method"])
    save_fig(fig, "confusion_matrices.png")


def plot_top_anomaly_breakdown(top: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    by_type = top["insurance_type"].value_counts()
    axes[0].bar(by_type.index, by_type.values, color=CATEGORICAL[0], width=0.6)
    axes[0].set_title("Top-500 Anomalies by Insurance Type")
    axes[0].tick_params(axis="x", rotation=30)

    by_severity = top["incident_severity"].value_counts().reindex(
        ["Minor Loss", "Major Loss", "Total Loss"]
    )
    axes[1].bar(by_severity.index, by_severity.values, color=CATEGORICAL[1], width=0.6)
    axes[1].set_title("Top-500 Anomalies by Severity")
    axes[1].tick_params(axis="x", rotation=30)

    by_agent = top["agent_name"].value_counts().head(15)
    axes[2].barh(by_agent.index, by_agent.values, color=CATEGORICAL[2])
    axes[2].invert_yaxis()
    axes[2].set_title("Top 15 Agents by Anomaly Count")
    save_fig(fig, "top_anomalies_breakdown.png")


# ---------------------------------------------------------------------------
# Explanations for the top-500 shortlist
# ---------------------------------------------------------------------------
def build_explanation(row: pd.Series) -> str:
    parts = []
    if row["zscore_flag"]:
        parts.append(f"Z-score {row['claim_zscore']:.1f}σ vs {row['insurance_type']} average")
    if row["iso_flag"]:
        parts.append(f"Isolation Forest anomaly (score {row['iso_score']:.2f})")
    if row["lof_flag"]:
        parts.append(f"LOF density anomaly (score {row['lof_score']:.2f})")
    if row["ratio_rule"]:
        parts.append(f"claim/premium ratio {row['claim_to_premium_ratio']:.1f}x (>10x threshold)")
    if row["state_mismatch_rule"]:
        parts.append(f"policy state {row['state']} ≠ incident state {row['incident_state']}")
    if not parts:
        return "Elevated ensemble score without any single method crossing its flag threshold"
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> int:
    setup_logging(LOG_PATH)
    configure_style()
    logger.info("=== Starting anomaly detection ===")
    logger.warning(
        "Evaluation uses incident_severity=='Total Loss' as a PROXY label - "
        "there is no verified fraud ground truth in this dataset. Metrics "
        "below compare methods to each other, not to real fraud outcomes."
    )

    try:
        claims, agents = load_data()
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    y_true = build_proxy_label(claims)

    print_section("Running detection methods")
    method_funcs = {
        "zscore": method_zscore,
        "isolation_forest": method_isolation_forest,
        "lof": method_lof,
        "rule_based": method_rule_based,
    }
    method_results = {}
    for name, func in method_funcs.items():
        try:
            method_results[name] = func(claims)
        except Exception:
            logger.error("Method '%s' failed - excluding it from the ensemble", name)
            logger.debug(traceback.format_exc())

    if not method_results:
        logger.error("All detection methods failed; aborting.")
        return 1

    scored = claims.copy()
    for name, result in method_results.items():
        scored = scored.merge(result, on="transaction_id", how="left")
    scored["proxy_label"] = y_true.values

    print_section("Evaluating methods against proxy label")
    flag_score_cols = {
        "zscore": ("zscore_flag", "zscore_score"),
        "isolation_forest": ("iso_flag", "iso_score"),
        "lof": ("lof_flag", "lof_score"),
        "rule_based": ("rule_flag", "rule_score"),
    }
    comparison_rows = []
    for name in method_results:
        flag_col, score_col = flag_score_cols[name]
        row = evaluate_method(name, scored["proxy_label"], scored[score_col], scored[flag_col].astype(int))
        comparison_rows.append(row)
    comparison_df = pd.DataFrame(comparison_rows)

    print_section("Ensemble scoring")
    active_weights = {k: v for k, v in ENSEMBLE_WEIGHTS.items() if k in method_results}
    weight_sum = sum(active_weights.values())
    if not np.isclose(weight_sum, 1.0):
        logger.info("Renormalizing ensemble weights (missing method(s)): %s", active_weights)
        active_weights = {k: v / weight_sum for k, v in active_weights.items()}

    scored["ensemble_score"] = 0.0
    for name, weight in active_weights.items():
        _, score_col = flag_score_cols[name]
        scored["ensemble_score"] += weight * scored[score_col]
    logger.info("Ensemble weights used: %s", {k: round(v, 3) for k, v in active_weights.items()})

    # Cross-method agreement: how correlated are the four scores with each
    # other? Low correlation confirms the methods are catching different
    # kinds of anomalies rather than all rediscovering the same claims.
    score_cols_present = [flag_score_cols[n][1] for n in method_results]
    corr = scored[score_cols_present].corr()
    logger.info("Pairwise correlation between method scores:\n%s", corr.round(2).to_string())

    top = scored.nlargest(TOP_N, "ensemble_score").copy()
    top = top.merge(agents[["agent_id", "agent_name"]], on="agent_id", how="left")
    top["explanation"] = top.apply(build_explanation, axis=1)
    logger.info(
        "Top %d anomalies by ensemble score: mean claim_amount=%.2f (dataset mean=%.2f), "
        "proxy-positive rate within top %d = %.1f%%",
        TOP_N, top["claim_amount"].mean(), scored["claim_amount"].mean(), TOP_N,
        100 * top["proxy_label"].mean(),
    )

    print_section("Generating visualizations")
    for plot_name, plot_func, plot_args in [
        ("score_distributions", plot_score_distributions, (scored,)),
        ("roc_curves", plot_roc_curves, (scored, scored["proxy_label"])),
        ("confusion_matrices", plot_confusion_matrices, (comparison_rows,)),
        ("top_anomaly_breakdown", plot_top_anomaly_breakdown, (top,)),
    ]:
        try:
            plot_func(*plot_args)
        except Exception:
            logger.error("Plot '%s' failed", plot_name)
            logger.debug(traceback.format_exc())

    print_section("Exporting results")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(OUTPUT_DIR / "anomaly_scores.parquet", index=False)
    logger.info("Saved -> %s (%d rows)", OUTPUT_DIR / "anomaly_scores.parquet", len(scored))

    explanation_cols = [
        "transaction_id", "customer_id", "agent_id", "agent_name", "vendor_id",
        "insurance_type", "claim_amount", "premium_amount", "claim_to_premium_ratio",
        "incident_severity", "claim_status", "state", "incident_state",
        "zscore_score", "iso_score", "lof_score", "rule_score", "ensemble_score", "explanation",
    ]
    top[explanation_cols].to_csv(OUTPUT_DIR / "top_anomalies.csv", index=False)
    logger.info("Saved -> %s (%d rows)", OUTPUT_DIR / "top_anomalies.csv", len(top))

    comparison_df.to_csv(OUTPUT_DIR / "method_comparison.csv", index=False)
    logger.info("Saved -> %s", OUTPUT_DIR / "method_comparison.csv")
    logger.info("\n%s", comparison_df.round(3).to_string(index=False))

    mean_auc = comparison_df["roc_auc"].mean()
    if mean_auc < 0.55:
        logger.warning(
            "All methods land close to AUC=0.50 (mean=%.3f) against the proxy label - "
            "'Total Loss' severity is essentially UNCORRELATED with statistical "
            "anomalousness in this dataset. This is a real finding, not a modeling "
            "failure: it means severity is a poor proxy for 'unusual claim' here, and "
            "any of these scores should be validated against real investigation "
            "outcomes before being used to prioritize claims in production.",
            mean_auc,
        )

    logger.info("=== Anomaly detection finished ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
