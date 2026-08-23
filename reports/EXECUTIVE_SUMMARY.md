# RetailPulse - Executive Summary

_Generated automatically by the pipeline. Warehouse: `retailpulse.db`._

## 1. Can we trust the data?
- Data quality score: **99.92/100** across 24 automated checks.
- 7 checks failed; 2,439 rows were quarantined rather than silently dropped.

## 2. Who are the customers?
- Segmented **11,994** identified customers into RFM segments.
- The top 1% of customers produce **10.3%** of revenue; the top 20% produce **59.3%**.
- Month-1 repeat rate is **29.4%**.

## 3. What are they worth?
- Modelled 12-month customer value: **Rs 32,573,251** across the book.
- 3,684 customers are more likely dead than alive (P(alive) < 0.30).
- Holdout check: predicted 9735.4 transactions vs 9602.0 actual (1.39% error).

## 4. Who is about to leave?
- Out-of-time ROC-AUC **0.8294**, PR-AUC 0.9363 (trained 2024-06-30, tested 2024-09-30).
- The riskiest decile churns at **1.318x** the base rate (the ceiling is 1.322x, i.e. a decile of pure churners - with a base rate this high, ranking has little room to run).
- Most efficient campaign: contact the top 5% at **4.497x ROI** (99.8% of them really do churn).
- Largest total return: contact the top 100% for Rs 555,266 net.

## 5. What sells together?
- 88 association rules mined from 832 frequent itemsets.
- Strongest: Dish Wash Gel -> Scrub Pad (lift 37.2).

## 6. What happens next?
- 28-day revenue forecast: **Rs 6,932,162**.
- Backtested across 4 walk-forward folds, the hybrid model is **25.14% more accurate** (MASE) than a seasonal-naive baseline.

## 7. What went wrong in the stores?
- 14 incidents flagged (3 dips, 11 spikes).
- Against the 28 incidents the simulator injected: recall 0.4643, precision 0.9286.

---

Run `python -m retailpulse dashboard` for the interactive version.