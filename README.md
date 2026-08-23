# RetailPulse

**An end-to-end retail analytics platform: it simulates a supermarket chain, cleans and
warehouses the data, and then answers the five questions a retail business actually asks.**

> Who are my best customers? · What are they worth? · Who is about to leave?
> · What sells together? · How much will I sell next month?

One command builds everything:

```bash
pip install -e .
python -m retailpulse all          # ~2 minutes, fully deterministic
python -m retailpulse dashboard    # interactive Streamlit app
```

---

## The idea that makes this project different

Most analytics portfolios download a CSV, fit a model, and print an accuracy number that
nobody can check. **This one builds its own data and hides the answers inside it.**

The simulator does not generate random noise. It draws customer purchase timing from the
*exact* statistical process the lifetime-value model later tries to recover, plants a fixed
list of product pairs that are bought together far more often than chance, injects store
outages on known dates, and corrupts a known number of rows in known ways.

So every claim in this repository has a checkable answer:

| Claim | How it is verified | Result |
|---|---|---|
| The data-quality engine finds real defects | Compare against the exact number of defects injected | **Every category matched exactly** |
| The CLV model is correctly specified | Recover the four BG/NBD parameters from data drawn from that process | r **0.695** vs 0.700 · α **5.97** vs 6.00 · dropout **0.201** vs 0.200 |
| "Probability alive" means something | Score it against each customer's *hidden* alive/dead flag | **AUC 0.92** |
| The basket miner finds real affinities | Check whether it rediscovers the planted pairs | **8 of 8 found**, confidences within ~0.03 of the planted probabilities |
| The anomaly detector works | Compare alerts to the injected incidents | **93% precision** (14 alerts, 1 false alarm) |
| The forecast is useful | Walk-forward backtest against a seasonal-naive baseline | **MASE 0.79** — 25% better than naive |
| The churn model does not leak | Train at one date, test at a strictly later one | **ROC-AUC 0.829** out of time |

These are not aspirations in a README — they are assertions in the test suite. `pytest`
fails if the BG/NBD likelihood is wrong, if a market-basket rule stops being found, or if
the forecast stops beating the naive baseline.

---

## Architecture

```
                    ┌──────────────────────────────────────┐
                    │  1. SIMULATE  (generate/)            │
                    │  BG/NBD customers · Gamma-Gamma spend│
                    │  festivals · promos · planted pairs  │
                    │  injected outages · injected defects │
                    └──────────────────┬───────────────────┘
                                       │  data/raw/*.csv  (bronze)
                    ┌──────────────────▼───────────────────┐
                    │  2. VALIDATE + LOAD  (etl/)          │
                    │  24 declarative checks across 6      │
                    │  quality dimensions → quarantine     │
                    │  → star schema in SQLite  (silver/gold)
                    └──────────────────┬───────────────────┘
                                       │  dim_* / fact_sales / mart_*
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
┌───────▼────────┐   ┌─────────────────▼──────────┐   ┌───────────────▼────────┐
│ 3. ANALYTICS   │   │ 4. MACHINE LEARNING        │   │ 5. MONITORING          │
│ RFM segments   │   │ BG/NBD + Gamma-Gamma CLV   │   │ market-factor model     │
│ cohort retention│  │ churn (out-of-time GBM)    │   │ + negative-binomial     │
│ Apriori baskets│   │ Holt-Winters + GBM hybrid  │   │ tail test + BH-FDR      │
└───────┬────────┘   └─────────────────┬──────────┘   └───────────────┬────────┘
        └──────────────────────────────┼──────────────────────────────┘
                    ┌──────────────────▼───────────────────┐
                    │  6. REPORT  (viz/ + dashboard/)      │
                    │  Streamlit + Plotly · 8 tabs         │
                    │  EXECUTIVE_SUMMARY.md                │
                    └──────────────────────────────────────┘
```

Every stage reads from the warehouse rather than from the stage before it in memory, so any
stage can be re-run on its own:

```bash
python -m retailpulse etl        # rebuild the warehouse
python -m retailpulse clv        # just refit lifetime value
python -m retailpulse forecast   # just the demand forecast
```

---

## The models, in plain language

### Customer lifetime value — BG/NBD + Gamma-Gamma *(written from scratch)*

**The problem.** In a shop, nobody cancels. A customer who has not visited for three months
is either gone or just slow, and you cannot tell by looking.

**The idea.** Describe every customer with two hidden dials:

- *how often they buy while they are still a customer* — spread across the population as a
  Gamma distribution;
- *the chance they quietly quit after any given purchase* — spread as a Beta distribution.

Then each customer collapses to three numbers — how many repeat visits, how long ago the
last one was, and how long we have known them — and the four population parameters fall out
of maximum likelihood. `scipy` does the optimisation; the log-likelihood, the survival
probability and the conditional expectations are all implemented here
([`analytics/clv.py`](src/retailpulse/analytics/clv.py)), including the Gaussian
hypergeometric term.

**Gamma-Gamma** then predicts *spend* per visit, and does the sensible thing with thin
evidence: a customer with one basket is predicted mostly by the market average, a customer
with fifty is predicted by themselves.

**Does it work?** Fitted on data generated by a *known* process, it recovered r = 0.695
(true 0.700) and α = 5.97 (true 6.00). Its "probability alive" separates the customers who
really were still active from the ones who really had left with **AUC 0.92** — and that flag
exists nowhere in the data the model sees. On a six-month holdout it predicted 9,735
transactions against 9,602 actual: **1.4% off**.

### Demand forecasting — Holt-Winters + a gradient-boosted correction *(written from scratch)*

**Stage 1.** Triple exponential smoothing tracks three things and updates them every day:
*where are we*, *which way are we heading*, and *what does this weekday usually do*.
Seasonality is multiplicative, because Saturday is "+38%", not "+X rupees". The three
smoothing rates are fitted by minimising one-step-ahead error — nothing is hard-coded.

**Stage 2.** Smoothing cannot know about Diwali. So a gradient-boosted tree is trained on
what Holt-Winters *got wrong*, using only what a planner genuinely has in advance: the
calendar, the festival diary, the promo plan, and the state of the series at the moment the
forecast is made.

**Stage 3.** Walk-forward backtesting — stand in the past, forecast 28 days, score against
what happened, roll forward, repeat.

| Model | MAPE | sMAPE | MASE |
|---|---|---|---|
| Seasonal naive ("same as last week") | 16.9% | 15.4% | 1.056 |
| Holt-Winters alone | 14.2% | 12.9% | 0.902 |
| **Hybrid** | **12.6%** | **11.5%** | **0.790** |

MASE below 1.0 means "better than the naive baseline". The hybrid is **25% better**.

### Market basket — Apriori *(written from scratch)*

Every basket becomes a row in a boolean matrix. The algorithm walks up by size: find single
products that clear the support floor, join pairs that share a prefix, and — the key step —
discard any candidate with an infrequent subset before counting it. That pruning is what
stops the search exploding. Support is counted by intersecting the parents' bit-vectors, so
each level reuses the one below instead of rescanning the data.

Out of 832 frequent itemsets it produced 88 rules, and it found **all eight planted pairs**:

| Rule | Confidence found | Confidence planted | Lift |
|---|---|---|---|
| Dish Wash Gel → Scrub Pad | 0.577 | 0.57 | 37.2× |
| Bread Loaf → Butter 500g | 0.626 | 0.60 | 12.9× |
| Baby Diapers → Baby Wipes | 0.694 | 0.68 | 11.2× |
| Filter Coffee Powder → Full Cream Milk | 0.623 | 0.62 | 20.5× |
| Shampoo → Hair Conditioner | 0.587 | 0.58 | 18.5× |
| Instant Noodles → Tomato Ketchup | 0.553 | 0.55 | 17.6× |
| Green Tea → Honey | 0.512 | 0.49 | 10.0× |
| Phone Charger → USB Cable | 0.541 | 0.52 | 6.9× |

The largest gap between a discovered confidence and the probability actually used by the
simulator is **0.026**.

*Lift* is the number to read: how many times more often two things are bought together than
if shoppers chose independently.

### Churn — gradient boosting, evaluated honestly

Features are built strictly as of a snapshot date; the label looks 90 days *forward* from
that snapshot. Training uses one snapshot, testing a strictly later one — a real out-of-time
split, not a random one that lets the model peek at its own future.

- **ROC-AUC 0.829**, PR-AUC 0.936, Brier 0.134, well calibrated across all ten buckets.
- The riskiest decile churns at **1.318×** the base rate. That sounds modest until you notice
  the ceiling is **1.322×** — with a 75.6% base churn rate, a decile of *pure* churners is all
  1.32× can ever mean. The model is essentially at the maximum.
- A logistic-regression baseline scores 0.830. **The gradient booster buys nothing here** —
  the signal is dominated by recency and how overdue a customer is relative to their own
  habit. That is reported rather than hidden, because it is the honest finding.
- Turned into money: contacting the top 5% returns **4.5× ROI** at 99.8% precision.

### Anomaly detection — market factor + negative-binomial tail test

A "3 standard deviations from the mean" detector fires every Saturday and every Diwali,
because those days genuinely *are* far from the mean. So the structure comes out first:

```
sales[store, day] = level[store] × market[day] × surprise[store, day]
```

`market[day]` is the median across all stores of how each store is doing against its own
baseline — the chain moving together. A festival lifts every store, so it lands in
`market` and never reaches the alert list. A stockout hits one store, so it survives into
`surprise`.

Then the sharper question: *if this store were normal today, how likely is a day this bad?*
The count is tested against a negative binomial and the p-values go through
**Benjamini-Hochberg** FDR control, so the alert list has a bounded false-discovery rate
rather than a bounded per-test error. At ~17 baskets a day per store, Poisson noise is far
too large for a z-score to reason about correctly — this is the part that took the detector
from 5% precision to **93%**.

Against 28 injected incidents: 13 caught, 1 false alarm. The misses are honest — a 2-day,
40% dip at a small store is genuinely below the noise floor at this volume, and no threshold
recovers it without drowning the list in false alarms.

---

## Data quality: the boring part done properly

24 declarative checks across six dimensions — completeness, uniqueness, validity,
consistency, referential integrity, timeliness. Rows failing a **critical** check are moved
to a quarantine table **with the reason attached**, never silently dropped, so every row is
either in the warehouse or accounted for in quarantine.

The simulator injects a known number of duplicates, nulls, malformed emails, negative
quantities, impossible prices and orphaned rows. The test suite asserts the engine finds
**exactly** those numbers.

One subtlety worth noting: uniqueness is a *set* property. Quarantining both copies of a
duplicate would throw away a genuine record along with the accidental one, so only the
surplus copy is quarantined — and the reason is read off the original evaluation, because
re-checking uniqueness on a subset that contains one copy of each duplicate would declare
it clean.

---

## What one run produces

```
Data quality score      99.92 / 100     24 checks, 2,439 rows quarantined
Warehouse               601,691 fact rows across 12 stores, 80 products, 3 years
Customers segmented     11,994          top 1% = 10.3% of revenue, top 20% = 59.3%
Lifetime value          Rs 3.26 crore modelled over 12 months
Churn                   ROC-AUC 0.829 out of time
Basket rules            88 rules from 832 frequent itemsets
Forecast                28-day projection, MASE 0.790
Incidents               14 flagged, 93% precision
Total runtime           ~148 seconds  (including data generation)
```

A one-page brief is written to [`reports/EXECUTIVE_SUMMARY.md`](reports/EXECUTIVE_SUMMARY.md)
on every run, and every intermediate table lands in `data/outputs/` as CSV.

---

## Dashboard

`python -m retailpulse dashboard` opens eight tabs — Overview, Data quality, Customers,
Lifetime value, Churn, Basket, Forecast, Incidents. Every number is read back from the
warehouse and `data/outputs/`, so the dashboard can never disagree with the pipeline.

Charts follow one palette with colour assigned by the job it does: a fixed categorical order
for identity, a single hue light-to-dark for magnitude, and reserved status colours for
incidents. One y-axis per chart, always.

---

## Testing

```bash
pytest              # 77 tests, ~25 seconds
pytest -m slow      # plus the full end-to-end pipeline run
```

The suite is not smoke tests. It includes:

- **Algorithms against hand-worked examples** — Apriori supports on five baskets you can
  count by eye; every rule metric checked against its definition; forecast metrics checked
  against arithmetic done by hand.
- **Statistical recovery** — BG/NBD recovers the parameters that generated the data;
  Holt-Winters recovers a planted weekly shape; two independent products score lift ≈ 1.
- **Leakage traps** — deleting every row after the snapshot must not change a single churn
  feature; no backtest fold may train on data from its own test window.
- **Reconciliation** — every pre-aggregated mart must sum to the fact table exactly.
- **The dashboard** — Streamlit's `AppTest` runs the real app and fails on any exception
  in any tab.

Three real bugs were caught this way while building: a set-property check being wrongly
re-evaluated on a subset, seasonal decomposition silently skipping zero-sales days (exactly
the days a stockout produces), and Streamlit's own chart theme quietly overriding the
palette.

---

## Layout

```
src/retailpulse/
├── config.py               all tunable parameters in one place
├── pipeline.py             stage orchestration, timing, executive report
├── cli.py                  python -m retailpulse <command>
├── generate/               the simulator and its planted ground truth
│   ├── synthetic.py
│   └── calendar_effects.py festivals, paydays, weekday rhythm
├── etl/
│   ├── quality.py          the declarative check engine
│   ├── transform.py        bronze → silver → gold, quarantine
│   └── load.py             SQLite warehouse + indexes
├── analytics/
│   ├── rfm.py              segmentation
│   ├── cohort.py           retention triangles
│   ├── basket.py           Apriori + association rules
│   ├── clv.py              BG/NBD + Gamma-Gamma
│   └── anomaly.py          market factor + NB tail test + BH-FDR
├── ml/
│   ├── features.py         point-in-time feature engineering
│   ├── churn.py            model, metrics, targeting simulation
│   └── forecast.py         Holt-Winters + hybrid + backtesting
└── viz/                    palette and reusable Plotly builders
dashboard/app.py            Streamlit app
tests/                      77 tests
```

## Requirements

Python 3.10+, and `pandas`, `numpy`, `scipy`, `scikit-learn`, `plotly`, `streamlit`.
No statistical modelling library is used for the algorithms above — BG/NBD, Gamma-Gamma,
Apriori, Holt-Winters, the decomposition and the FDR control are all implemented in this
repository. `scikit-learn` supplies the gradient boosting; `scipy` supplies the optimiser
and special functions.

All generated data is reproducible from `seed = 42` and is gitignored — the pipeline
rebuilds it identically.

## License

MIT
