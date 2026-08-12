# Framework Mapping: Cross-Domain Portability

This document is the reference for how the anomaly detection framework built for insurance claims (`python/03_anomaly_detection.py`) transfers to other financial domains. Domains 1–3 are implemented and tested in `python/04_framework_portability.py`; every number quoted below comes from actually running that code, not estimation. Domain 4 (AML/Compliance) is a design blueprint — laid out in the same structure and level of detail, but **not implemented in code**, so it is clearly marked as such throughout.

The underlying claim: **one detection engine, four config-driven building blocks.**

```
compute_group_zscore(df, amount_col, group_col, threshold)   -> statistical outliers, scaled to the group's own distribution
compute_isolation_forest(df, feature_cols, contamination)    -> multivariate outliers via random recursive partitioning
compute_lof(df, feature_cols, contamination, n_neighbors)    -> multivariate outliers via local density comparison
compute_rule_based(rules, weights)                            -> hand-written business rules, OR'd into a flag, weighted into a score
```

Everything domain-specific lives in a `CONFIG` dict passed into these four functions plus a shared `run_scenario()` orchestrator. Porting to a new domain means writing a new `CONFIG`, not new detection code.

---

## Domain 1: Insurance Claims (current, implemented)

### Data characteristics
- 10,000 real claims, 37 columns after cleaning (`output/insurance_claims_clean.parquet`), spanning 6 insurance types (Property, Mobile, Health, Life, Travel, Motor) with wildly different claim scales — Life averages $54,386/claim, Mobile averages $407/claim.
- No verified fraud/investigation label exists in the source data — the closest available signal is `incident_severity` (Minor/Major/Total Loss), used throughout as an explicit **proxy label**, not ground truth.
- Claim amounts are bounded within each insurance type rather than heavy-tailed (max Z-score observed anywhere: 1.76σ) — a property of this specific dataset that materially affects which methods find anything.

### Feature engineering approach
- `days_to_process = report_dt - loss_dt` — the claim's processing-delay gap.
- `claim_to_premium_ratio = claim_amount / premium_amount` — normalizes claim size against what the customer actually paid.
- `customer_claim_frequency` — claim count per customer (present in the schema; in this particular dataset every customer has exactly one claim, so it carries no signal here, but the column exists for datasets with repeat customers).

### Detection thresholds

| Component | Config | Value |
|---|---|---|
| Z-score group | `group_col` | `insurance_type` |
| Z-score flag | `threshold` | `\|z\| > 3` (classic 3-sigma) |
| Isolation Forest / LOF | `contamination` | 5% (`ASSUMED_ANOMALY_RATE`) |
| Rule 1 | `claim_to_premium_ratio > 10` | fires on **97.6%** of claims |
| Rule 2 | `state != incident_state` | fires on **93.6%** of claims |
| Ensemble weights | zscore / iso / lof / rule | 0.30 / 0.30 / 0.25 / 0.15 |

### Key insights
- **Both business rules are saturated.** They sounded like sensible fraud heuristics in isolation, but on this dataset's actual scale (premiums are structurally tiny relative to claims; state mismatch turns out to be the norm, not the exception, likely because Travel/Motor claims are often filed away from home) they fire on nearly every row. The rule-based method's F1 = 0.506 is a "flag almost everyone" degenerate result, not genuine skill — visible immediately in its confusion matrix (`TN=8, FP=6602`).
- **All four methods land on the ROC random diagonal (AUC ≈ 0.50–0.51)** against the severity proxy label. This is the most important finding in the whole project: it means severity is uncorrelated with statistical anomalousness here, and these metrics should never be read as real-world fraud-detection performance.
- Z-score literally flags **zero** claims — not a bug, but proof the data is bounded/near-uniform per type rather than heavy-tailed like real-world claims data typically is.

---

## Domain 2: Payment Systems (implemented, synthetic data)

### Data transformation needed
No real payment data existed for this project, so `04_framework_portability.py` generates 8,000 synthetic card transactions with realistic structure (800 accounts, 8 merchant categories, lognormal amounts scaled per category, card-issuing vs. merchant country) and **3 injected true-label anomaly patterns**: amount spikes (10–40x normal), cross-border card/merchant mismatches bundled with a larger amount, and velocity bursts (many transactions for one account in a tight time window). Because the labels are injected by the generator, precision/recall/F1 here are measured against **true labels**, not a proxy — a meaningfully stronger evaluation than Domain 1's.

### Feature mapping

| Insurance concept | Payment equivalent | Why |
|---|---|---|
| `claim_amount` | `transaction_amount` | the core magnitude column |
| `insurance_type` (z-score group) | `merchant_category` | groups with comparable spend scale |
| `claim_to_premium_ratio` | `amount_to_account_avg_ratio` (`transaction_amount / account's own average`) | payments have no "premium" to divide by, so the ratio is computed against the account's own historical baseline instead |
| `days_to_process` (temporal feature) | `txn_count_last_1h` (rolling velocity) | **payments have no processing-delay gap** — transactions are near-instant, so the framework substitutes velocity as its temporal signal instead of forcing a delay concept that doesn't exist |
| `state != incident_state` (mismatch rule) | `card_country != merchant_country` | direct structural analog — a cross-border mismatch rule |
| age / tenure / family (LOF demographics) | `merchant_risk_score`, `transaction_hour` | no customer demographics in a payment record, so LOF instead draws on merchant-category risk and time-of-day |

### Why thresholds change
The Z-score rule (`\|z\| > 3`) is applied with the **identical threshold** as insurance, but the dollar amount it implies is wildly different by merchant category — from ~$498 for Restaurant to ~$13,257 for Travel (`implied_z3_dollar_threshold`, logged per run). No manual retuning was needed: the statistic is scale-invariant by construction. What *did* need to change is the rule-based thresholds themselves, because they're expressed in domain terms, not abstract statistics — `amount_to_account_avg_ratio > 5` (not the insurance ratio's `> 10`, since account-relative deviation is a smaller-magnitude signal than claim/premium) and a country-code inequality instead of a state-code inequality.

### Expected — and observed — improvements
Because these rules were built with realistic, *low* base rates (amount-spike fires on 2.1% of transactions, geo-mismatch on 7.9% — nothing like insurance's 93–97% saturation) and evaluation uses true labels:

| Method | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Z-score | 1.000 | 0.257 | 0.409 | 0.797 |
| Isolation Forest | 0.430 | 0.614 | 0.506 | 0.909 |
| LOF | 0.198 | 0.282 | 0.232 | 0.787 |
| Rule-based | 0.224 | 0.604 | 0.326 | 0.771 |
| **Ensemble** | **0.518** | **0.739** | **0.609** | **0.948** |

The ensemble beats every individual method here — it did not for insurance — directly supporting the conclusion that Domain 1's weak numbers were a proxy-label problem, not a framework problem.

### Pseudocode — what actually changes
```python
PAYMENT_CONFIG = {
    "amount_col": "transaction_amount",                       ### CHANGE ### was claim_amount
    "group_col": "merchant_category",                          ### CHANGE ### was insurance_type
    "z_threshold": 3.0,                                         # unchanged
    "iso_features": [
        "days_since_start", "transaction_hour",
        "txn_count_last_1h",                                    ### CHANGE ### velocity replaces days_to_process
        "transaction_amount", "amount_to_account_avg_ratio",
    ],
    "lof_features": [
        "transaction_amount", "amount_to_account_avg_ratio",
        "txn_count_last_1h", "merchant_risk_score",             ### CHANGE ### merchant risk replaces demographics
        "transaction_hour",
    ],
    "rules": lambda df: {
        "amount_spike_rule": df["amount_to_account_avg_ratio"] > 5,     ### CHANGE ### threshold + baseline column
        "geo_mismatch_rule": df["card_country"] != df["merchant_country"],  ### CHANGE ### column names only
    },
    "rule_weights": {"amount_spike_rule": 0.7, "geo_mismatch_rule": 0.3},  # unchanged weighting scheme
}
# compute_group_zscore(...), compute_isolation_forest(...), compute_lof(...), compute_rule_based(...)
# are called EXACTLY as they are for insurance - only PAYMENT_CONFIG differs.
```

---

## Domain 3: Fund Operations (implemented, synthetic data)

### Settlement-specific features
5,000 synthetic settlement records (50 funds, 5 instrument types — Equity/Bond/FX/Derivative/Repo — 150 counterparties) with 3 injected true-label anomaly patterns: counterparty concentration risk (high-risk counterparty + oversized settlement), severe settlement delay, and settlement-currency mismatch bundled with a larger amount. `settlement_amount` is the amount column, scaled by instrument type ($500K median for Equity up to $10M+ for Repo) — an order of magnitude larger than insurance claims or payment transactions, reflecting real institutional settlement sizes.

### Timing considerations
`settlement_days = settlement_date - trade_date` is the **direct structural analog of insurance's `days_to_process`** (loss → report) — both are a processing-delay gap fed straight to Isolation Forest. But funds diverge from both other domains on time-of-day: **no `transaction_hour`-equivalent feature is used**, because settlements happen in batch cycles (T+0 to T+3), not tied to intraday timing the way a retail claim or card swipe is. This is a case where the framework correctly *drops* a feature type rather than forcing an analog that doesn't exist for the domain.

### Risk dimension changes
`counterparty_risk_rating` (Low/Medium/High) is a direct structural echo of insurance's `risk_segmentation` (L/M/H) — both encode a categorical risk tier, numeric-encoded (1/2/3) for LOF. `fund_concentration_ratio = settlement_amount / fund's own average settlement` plays the same role as insurance's `claim_to_premium_ratio` and payments' `amount_to_account_avg_ratio`: a domain-appropriate "how unusual is this relative to its own baseline" feature, present in every domain in some form.

### Pseudocode — what actually changes
```python
FUND_CONFIG = {
    "amount_col": "settlement_amount",                          ### CHANGE ### was claim_amount
    "group_col": "instrument_type",                              ### CHANGE ### was insurance_type
    "z_threshold": 3.0,                                          # unchanged
    "iso_features": [
        "days_since_start", "settlement_days",                   ### CHANGE ### trade->settlement gap, direct analog
        "settlement_amount", "fund_concentration_ratio",         #             of days_to_process
    ],                                                            # NOTE: no hour-of-day feature - see Timing above
    "lof_features": [
        "settlement_amount", "settlement_days",
        "counterparty_risk_score",                                ### CHANGE ### numeric-encoded L/M/H risk tier
        "fund_concentration_ratio",
    ],
    "rules": lambda df: {
        "settlement_delay_rule": df["settlement_days"] > 5,               ### CHANGE ### delay threshold, no "premium" concept
        "currency_mismatch_rule": df["settlement_currency"] != df["fund_base_currency"],  ### CHANGE ### column names only
    },
    "rule_weights": {"settlement_delay_rule": 0.6, "currency_mismatch_rule": 0.4},  ### CHANGE ### re-balanced weights
}
```

### Results
Base rates for both rules stayed realistically low (settlement-delay 1.3%, currency-mismatch 1.3%), so the rule-based method alone reached **precision 1.000 / recall 0.665** — tied with payments' Z-score for the best precision of any method in the project, but with far higher recall — and the ensemble reached **F1 = 0.747, ROC-AUC = 0.984**, the strongest ensemble result in the whole project.

---

## Domain 4: Compliance / AML (design blueprint — not implemented)

This domain was mapped out to test whether the framework's pattern (ratio/spike rule + mismatch/high-risk rule, a temporal-gap-or-velocity feature, a categorical risk-tier dimension) continues to hold for a fourth, structurally different domain — anti-money-laundering transaction monitoring. No code was written or run for this section; it is included to show the mapping process generalizes, not to claim results.

### Data characteristics (anticipated)
AML monitoring operates on account-level transaction streams rather than discrete claims or settlements: wires, cash deposits/withdrawals, and ACH transfers, each carrying a counterparty, a jurisdiction, and (crucially) a **regulatory reporting threshold** (e.g. the $10,000 Currency Transaction Report threshold in the US) that has no equivalent in any of the first three domains.

### Feature mapping

| Insurance concept | AML equivalent | Why |
|---|---|---|
| `claim_amount` | `transaction_amount` | core magnitude column |
| `insurance_type` (z-score group) | `transaction_type` (wire / cash / ACH / check) | groups with comparable typical size and regulatory treatment |
| `claim_to_premium_ratio` | `amount_to_account_avg_ratio` | same account-relative-deviation pattern as payments |
| `days_to_process` (temporal) | `txn_count_last_24h` **and** `pct_of_threshold` (`transaction_amount / $10,000`) | AML's signature pattern is *structuring* — multiple transactions kept just under the reporting threshold in a short window — so the temporal feature must capture velocity **and** proximity-to-threshold together, not just velocity alone as in payments |
| `state != incident_state` (mismatch rule) | `counterparty_country in HIGH_RISK_JURISDICTIONS` | evolves the mismatch pattern into a **watchlist-membership** rule rather than a simple inequality, since AML risk is about specific known-risk jurisdictions/entities, not any cross-border difference |
| `risk_segmentation` (L/M/H) | `customer_risk_rating` (Low/Medium/High/PEP) | same categorical risk-tier pattern as funds' `counterparty_risk_rating`, extended with a Politically-Exposed-Person flag |

### Why thresholds would change
Reporting-threshold proximity is domain-specific in a way none of the first three domains required: `pct_of_threshold` only makes sense because a specific dollar cutoff ($10,000) is defined by regulation, not by a comparison to the entity's own history. This is the one place the mapping needs a genuinely new feature category (threshold-proximity) rather than a renamed existing one — worth flagging as a limit on how far "just rename the columns" portability goes.

### Pseudocode — proposed configuration (NOT implemented or tested)
```python
### DESIGN BLUEPRINT - NOT RUN ###
AML_CONFIG = {
    "amount_col": "transaction_amount",
    "group_col": "transaction_type",
    "z_threshold": 3.0,
    "iso_features": [
        "days_since_account_opened", "txn_count_last_24h",
        "pct_of_reporting_threshold",           # NEW feature category - no direct analog in domains 1-3
        "transaction_amount", "amount_to_account_avg_ratio",
    ],
    "lof_features": [
        "transaction_amount", "amount_to_account_avg_ratio",
        "customer_risk_score", "txn_count_last_24h",
        "counterparty_country_risk_score",
    ],
    "rules": lambda df: {
        "structuring_rule": (
            (df["pct_of_reporting_threshold"].between(0.8, 1.0))
            & (df["txn_count_last_24h"] >= 3)
        ),                                       # multiple near-threshold transactions in one day
        "high_risk_jurisdiction_rule": df["counterparty_country"].isin(HIGH_RISK_JURISDICTIONS),
    },
    "rule_weights": {"structuring_rule": 0.6, "high_risk_jurisdiction_rule": 0.4},
}
# Same compute_group_zscore / compute_isolation_forest / compute_lof / compute_rule_based
# calls as every other domain. Before trusting this config on real data, the base rate of
# BOTH rules must be checked first - Domain 1 already proved that skipping this step turns
# a plausible-sounding rule into a "flag everything" false signal.
```

---

## Conclusion: The Framework Is Genuinely Reusable

Three lines of evidence, not just an architectural claim:

1. **Bit-identical cross-check.** Domain 1, re-run through the generic engine built for Domains 2–3, reproduces `03_anomaly_detection.py`'s original F1 scores exactly (verified programmatically in `04_framework_portability.py`, not eyeballed). The generic functions are a faithful refactor of the original bespoke code, not a coincidentally similar reimplementation.
2. **Zero changes to detection code across three implemented domains.** `compute_group_zscore`, `compute_isolation_forest`, `compute_lof`, and `compute_rule_based` are called identically for insurance, payments, and funds. Every difference between domains lives in a config dict — visible, in the actual source, as `### CHANGE ###` comments.
3. **A recurring structural pattern held across all four domains**, including the unimplemented one: every domain decomposes into a ratio/spike rule plus a mismatch/high-risk rule, some form of temporal signal (a delay gap, a velocity count, or both), and a categorical risk tier. Even where the pattern had to bend — Domain 4's threshold-proximity feature has no equivalent in Domains 1–3 — it bent by *adding* a feature category within the same four-function structure, not by requiring new detection logic.

The framework's real limitation is not portability — it's that a business rule's usefulness depends entirely on its base rate in the specific dataset it's applied to, and that unsupervised anomaly scores are only as trustworthy as the label used to evaluate them. Both lessons were learned empirically in Domain 1 and confirmed by contrast in Domains 2–3, and both should be checked explicitly before this framework — or the Domain 4 blueprint — is pointed at any new real dataset.
