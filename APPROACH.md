# Design Decisions & Tradeoffs

## Why SQL first, Python second?

I could have done this all in pandas, but:

SQL forces me to think about data structure before analysis. If I can't model it cleanly 
in SQL, that's a signal the problem is unclear.

Schema design is your first defense against garbage-in-garbage-out. By the time I got to 
Python, the data quality was already validated.

Separates concerns. SQL = "what data do we have?" Python = "what do we see in the data?"

When this approach fails: If the data was already messy or came from 20 different sources. 
Then I'd start in pandas to explore, then design the schema backward.

---

## Why these 4 detection methods? And why not others?

Chose:
- Z-score (statistical simplicity)
- Isolation Forest (handles nonlinear patterns)
- Local Outlier Factor (density-based, good for clusters)
- Rule-based (domain-specific knowledge)

Rejected:
- Autoencoders (overkill for this data; can't explain anomalies)
- Markov chains (better for behavioral sequences, not point anomalies)
- Clustering (doesn't flag anomalies, just groups them)

Key insight: The "best" method depends on whether anomalies are rare points or weird 
patterns. Insurance and payments are mostly rare points. Fund ops are both. That's why 
the framework starts to break for compliance (which is pure patterns).

---

## What we didn't do (and why)

No AutoML or hyperparameter tuning: The framework is proof-of-concept, not production. 
If this goes live, tuning matters. For portfolio value, clarity matters more.

No online learning or retraining logic: I wanted to understand static anomalies first. 
Drift detection is a separate problem that deserves its own project.

No synthetic minority class balancing: There is no "normal" class to oversample. 
Anomalies are defined structurally, not by class imbalance.

---

## Framework limitations I discovered

Breaks for AML compliance: The framework detects statistical anomalies. AML detects 
behavioral anomalies (a pattern of small transactions that adds up). These need graph 
analysis and temporal state machines, not univariate scoring.

Doesn't handle seasonal data well: If your baseline changes by season (which fund flows do), 
we'd need to either train per-season models or add a seasonal component. The current Z-score 
approach will fail spectacularly in off-season.

Assumes independent transactions: If X buying from Y is only suspicious because Y sells to Z, 
we're missing relational context. The framework would need a graph layer.

---

## If I rebuilt this

Separate "clean" anomaly detection from "explain anomalies"—right now they're mixed.

Add a seasonal/trend decomposition layer before scoring.

Build modular scoring so you can swap methods without rebuilding.

Add a "confidence interval" output, not just a 0-1 flag.
