# Financial Anomaly Detection Framework

## Why

I was interviewing at PTSB and Invesco in mid-2026. Every technical interview 
hit the same wall: "Your analysis is solid. Does it work for **our** domain?"

I realized I didn't have an answer. I had insurance expertise from China Life, 
but I'd never actually *proven* the same detection logic works across different 
financial domains.

So I built this to test one hypothesis: **anomaly detection is domain-agnostic. 
Only parameters change.**

Insurance → Payments → Funds. Same question: "What's unusual?" Different data.

---

## What it does

Builds an anomaly detection system in three parts:

1. **SQL schema** — normalizes three source CSVs (claims, agents, vendors) 
   and preserves data quality issues (like 32% missing vendor IDs) as meaningful signals
   
2. **Python pipeline** — 5 scripts that run sequentially:
   - Clean and structure the data
   - Exploratory analysis (13 charts, 10 summary tables)
   - 4 detection methods (Z-score, Isolation Forest, LOF, rule-based)
   - Ensemble scoring across all methods
   - Export to Power BI
   
3. **Framework portability** — prove the same code works on synthetic payment 
   and fund data with only parameter changes (no code rewrites)

---

## What I learned

**Design thinking:**
- Start with schema, not code. If you can't model it in SQL, the problem is unclear.
- Detection ≠ explanation. Flagging an anomaly and explaining *why* are different problems.

**Technical limits:**
- Z-score breaks on multi-modal data (insurance types have different claim distributions).
- Isolation Forest is fast but opaque—you don't know *why* something is anomalous.
- LOF solves context, but costs O(n²) time. Sampling would be needed at scale.
- Rule-based wins on explainability. A $50k claim when premium is $100 *is* wrong.

**Domain insight:**
- Insurance is mostly point anomalies (rare individual claims).
- Payments are rare points + unusual sequences (same person buying weird combinations).
- Fund ops is both + relationships (counterparty risk, settlement networks).
- AML compliance needs behavioral patterns, not statistical anomalies. Different problem entirely.

---

## Tools & approach

**Technologies:**
- SQL (SQLite) for schema + data validation
- Python (pandas, scikit-learn) for analysis
- Power BI for visualization

**Development:**
Used Claude Code as an accelerator for boilerplate. I owned:
- Architecture decisions (SQL-first, 4-method ensemble)
- All anomaly detection logic
- Validation against real data patterns
- Framework adaptation testing

Every detection method was manually tested against insurance, payment, and fund data.

---

## Conclusion

**What works:**
The framework successfully identifies statistical anomalies across three financial domains 
using the same core methods. Detection accuracy (AUC 0.78-0.98) jumps when there are 
real anomaly labels (payments, funds) vs. when we use severity as proxy (insurance).

**Real finding:**
The insurance dataset is either synthetic or extremely clean. No method achieves 
>0.50 AUC using severity as ground truth. That's honest data quality—not a bug.

**Applies to:**
✅ Insurance/payments (point anomalies)
✅ Fund settlements (multi-dimensional scoring)
❌ AML compliance (needs behavioral, not statistical)

---

## What I'd change

If I rebuilt this:

- **Separate detection from explanation** — right now they're mixed. 
  Flagging is fast (Z-score), but explaining why takes 4 methods.
  
- **Add seasonal decomposition** — funds have annual patterns. 
  Current approach would fail spectacularly off-season.
  
- **Modular method selection** — swap detection algorithms without 
  rewriting downstream scoring.
  
- **Confidence intervals, not binary flags** — "95% confident this is anomalous" 
  is more useful than "yes/no."

The framework works, but it's a foundation, not a finished product.

---

## Try it

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline (each script depends on previous output)
python python/01_data_import.py          # SQL + clean data
python python/02_exploratory_analysis.py # 13 plots, summaries
python python/03_anomaly_detection.py    # 4 methods, ensemble score
python python/04_framework_portability.py # Prove it works on payment/fund data
python python/05_prepare_for_powerbi.py  # Export for visualization
```

View the results:
- Cleaned data: `output/*.parquet`
- Analysis plots: `output/eda_plots/`, `output/anomaly_plots/`
- Power BI data: `output/powerbi_data/`
- Full logs: `logs/`

---

## Transparency

Built with Claude Code for scaffolding, but validated and iterated by hand. 
The detection logic, architecture choices, and framework adaptation are my work. 
See APPROACH.md for design rationale.

---

## License

MIT
