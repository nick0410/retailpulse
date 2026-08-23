"""Churn features/labels and the anomaly detector.

The churn tests care mostly about *leakage*, because a leaking churn model
looks brilliant and is worthless. The anomaly tests check the two ideas the
detector rests on: the market factor cancels chain-wide events, and
Benjamini-Hochberg keeps the alert list honest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from retailpulse.analytics.anomaly import (anomaly_events, benjamini_hochberg, detect_anomalies,
                                           estimate_dispersion, evaluate_against_truth,
                                           robust_zscore)
from retailpulse.ml.churn import decile_lift, evaluate, ks_statistic, targeting_simulation, train_churn_model
from retailpulse.ml.features import build_churn_label, build_customer_features, build_training_frame


# --------------------------------------------------------------------------
# Labels and leakage
# --------------------------------------------------------------------------
@pytest.fixture
def toy_fact() -> pd.DataFrame:
    """Three customers with deliberately different futures."""
    rows = [
        # returns inside the horizon -> not churned
        ("C1", "2024-01-05"), ("C1", "2024-02-10"), ("C1", "2024-04-01"),
        # goes quiet after the snapshot -> churned
        ("C2", "2024-01-08"), ("C2", "2024-02-20"),
        # comes back only long after the horizon -> churned
        ("C3", "2024-01-02"), ("C3", "2024-02-01"), ("C3", "2024-09-15"),
    ]
    df = pd.DataFrame(rows, columns=["customer_id", "date"])
    df["date"] = pd.to_datetime(df["date"])
    df["line_amount"] = 500.0
    df["gross_margin"] = 150.0
    df["discount_amount"] = 0.0
    df["line_id"] = np.arange(len(df))
    df["store_id"] = "ST001"
    df["product_id"] = "P0001"
    df["category"] = "Groceries"
    df["channel"] = "in_store"
    df["transaction_id"] = ["T%d" % i for i in range(len(df))]
    return df


def test_churn_label_uses_only_the_forward_window(toy_fact):
    label = build_churn_label(toy_fact, pd.Timestamp("2024-03-01"), horizon_days=90)
    returned = set(label["customer_id"])
    assert "C1" in returned          # bought on 1 April, inside the window
    assert "C2" not in returned      # never came back
    assert "C3" not in returned      # came back in September, far outside


def test_features_ignore_everything_after_the_snapshot(toy_fact):
    """Deleting post-snapshot rows must not change a single feature value."""
    dim = pd.DataFrame({"customer_id": ["C1", "C2", "C3"],
                        "loyalty_tier": ["Gold", "Silver", "Bronze"],
                        "age_band": ["25-34"] * 3,
                        "preferred_channel": ["app"] * 3,
                        "customer_region": ["West"] * 3,
                        "signup_date": pd.to_datetime(["2023-06-01"] * 3)})
    snapshot = pd.Timestamp("2024-03-01")

    full = build_customer_features(toy_fact, dim, snapshot)
    truncated = build_customer_features(toy_fact[toy_fact["date"] <= snapshot], dim, snapshot)
    pd.testing.assert_frame_equal(full.sort_values("customer_id").reset_index(drop=True),
                                  truncated.sort_values("customer_id").reset_index(drop=True))


def test_training_frame_excludes_customers_with_no_history(fact, star):
    snapshot = pd.to_datetime(fact["date"]).max() - pd.Timedelta(days=120)
    frame = build_training_frame(fact, star["dim_customer"], snapshot,
                                 horizon_days=90, min_history_days=30)
    assert (frame["tenure_days"] >= 30).all()
    assert frame["churned"].isin([0, 1]).all()
    assert 0.0 < frame["churned"].mean() < 1.0, "degenerate labels"
    # A churned customer must have no future trips, by definition.
    assert (frame.loc[frame["churned"] == 1, "future_trips"] == 0).all()
    assert (frame.loc[frame["churned"] == 0, "future_trips"] > 0).all()


def test_recency_and_overdue_ratio_agree(fact, star):
    snapshot = pd.to_datetime(fact["date"]).max() - pd.Timedelta(days=120)
    feats = build_customer_features(fact, star["dim_customer"], snapshot)
    assert (feats["recency_days"] >= 0).all()
    assert (feats["tenure_days"] >= feats["recency_days"]).all()
    regular = feats[feats["gap_mean_days"] > 0]
    expected = regular["recency_days"] / regular["gap_mean_days"]
    assert np.allclose(regular["overdue_ratio"], expected)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def test_evaluate_reports_a_perfect_and_a_useless_model():
    y = np.array([0, 0, 1, 1])
    assert evaluate(y, np.array([0.1, 0.2, 0.8, 0.9]))["roc_auc"] == 1.0
    assert evaluate(y, np.array([0.9, 0.8, 0.2, 0.1]))["roc_auc"] == 0.0
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


def test_decile_lift_is_ordered_and_complete():
    rng = np.random.default_rng(0)
    n = 5000
    score = rng.random(n)
    y = (rng.random(n) < score).astype(int)   # score genuinely predicts y
    lift = decile_lift(y, score)
    assert len(lift) == 10
    assert lift["customers"].sum() == n
    assert lift["lift"].iat[0] > 1.0 > lift["lift"].iat[-1]
    assert lift["cumulative_churners_captured"].iat[-1] == pytest.approx(1.0)


def test_targeting_simulation_is_arithmetically_consistent():
    scored = pd.DataFrame({
        "customer_id": [f"C{i}" for i in range(100)],
        "churned": [1] * 40 + [0] * 60,
        "churn_probability": np.linspace(1.0, 0.0, 100),
        "monetary_total": np.full(100, 10_000.0),
    })
    sim = targeting_simulation(scored, contact_cost=40.0, save_rate=0.25,
                               margin_rate=0.30, horizon_scale=0.25)
    row = sim[sim["target_pct"] == 0.20].iloc[0]
    assert row["customers_contacted"] == 20
    assert row["churners_in_target"] == 20          # the top 20 are all churners
    assert row["campaign_cost"] == pytest.approx(20 * 40.0)
    assert row["expected_margin_saved"] == pytest.approx(0.25 * 20 * 10_000 * 0.25 * 0.30)
    assert row["net_benefit"] == pytest.approx(row["expected_margin_saved"] - row["campaign_cost"])


def test_churn_model_beats_chance_out_of_time(fact, star):
    end = pd.to_datetime(fact["date"]).max()
    train = build_training_frame(fact, star["dim_customer"], end - pd.Timedelta(days=180))
    test = build_training_frame(fact, star["dim_customer"], end - pd.Timedelta(days=90))
    result = train_churn_model(train, test, compute_importance=False)

    assert result.metrics["roc_auc"] > 0.65, result.metrics
    assert result.metrics["brier"] < 0.25
    assert result.scored["churn_probability"].between(0, 1).all()
    assert set(result.scored["risk_band"]) <= {"Low", "Medium", "High", "Critical"}


# --------------------------------------------------------------------------
# Anomaly detection
# --------------------------------------------------------------------------
def test_robust_zscore_matches_its_definition():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = robust_zscore(x)
    # median 3, MAD 1 -> z = 0.6745 * (x - 3)
    assert np.allclose(z.to_numpy(), 0.6745 * (x.to_numpy() - 3.0))


def test_robust_zscore_survives_a_degenerate_series():
    assert np.allclose(robust_zscore(pd.Series([5.0] * 10)).to_numpy(), 0.0)


def test_benjamini_hochberg_controls_the_list():
    # 5 obviously significant p-values buried in 995 null ones.
    rng = np.random.default_rng(0)
    p = np.concatenate([np.full(5, 1e-8), rng.uniform(0, 1, 995)])
    flagged = benjamini_hochberg(p, q=0.01)
    assert flagged[:5].all(), "clear signals were not flagged"
    assert flagged.sum() < 20, "far too many nulls survived FDR control"


def test_benjamini_hochberg_flags_nothing_when_there_is_nothing():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 1000)     # pure noise
    assert benjamini_hochberg(p, q=0.01).sum() == 0


def test_dispersion_detects_pure_poisson():
    rng = np.random.default_rng(2)
    lam = np.full(4000, 25.0)
    y = rng.poisson(lam).astype(float)
    assert np.isinf(estimate_dispersion(y, lam)) or estimate_dispersion(y, lam) > 100


def _panel(rng, n_stores=8, n_days=400, base=30.0, weekly=None) -> pd.DataFrame:
    weekly = weekly if weekly is not None else np.array([0.9, 0.9, 0.95, 1.0, 1.15, 1.3, 1.25])
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    rows = []
    for s in range(n_stores):
        lam = base * weekly[np.arange(n_days) % 7]
        rows.append(pd.DataFrame({"date": dates, "store_id": f"ST{s:03d}",
                                  "transactions": rng.poisson(lam).astype(float)}))
    return pd.concat(rows, ignore_index=True)


def test_detector_finds_an_injected_single_store_stockout():
    rng = np.random.default_rng(4)
    panel = _panel(rng)
    victim, window = "ST003", slice(200, 204)
    mask = panel["store_id"] == victim
    idx = panel[mask].index[window]
    panel.loc[idx, "transactions"] *= 0.2      # a four-day stockout

    scored = detect_anomalies(panel, value_col="transactions", method="nb_tail", fdr_q=0.01)
    events = anomaly_events(scored, value_col="transactions")
    hit = events[(events["store_id"] == victim) & (events["direction"] == "dip")]
    assert not hit.empty, "a four-day 80% stockout went unnoticed"


def test_a_chain_wide_event_is_not_an_incident():
    """Every store spikes together - that is a festival, not a store problem."""
    rng = np.random.default_rng(6)
    panel = _panel(rng)
    festival = panel["date"].isin(pd.date_range("2023-08-01", periods=3))
    panel.loc[festival, "transactions"] *= 2.5

    scored = detect_anomalies(panel, value_col="transactions", method="nb_tail", fdr_q=0.01)
    events = anomaly_events(scored, value_col="transactions")
    if not events.empty:
        overlapping = events[(events["start_date"] <= pd.Timestamp("2023-08-03"))
                             & (events["end_date"] >= pd.Timestamp("2023-08-01"))]
        assert len(overlapping) <= 1, \
            f"the market factor failed to absorb a chain-wide event: {len(overlapping)} alerts"


def test_a_quiet_panel_produces_almost_no_alerts():
    rng = np.random.default_rng(8)
    scored = detect_anomalies(_panel(rng), value_col="transactions",
                              method="nb_tail", fdr_q=0.01)
    assert scored["is_anomaly"].sum() <= 5, "false-alarm rate is out of control"


def test_evaluation_against_truth_counts_overlaps():
    detected = pd.DataFrame({
        "store_id": ["ST001", "ST002"],
        "start_date": pd.to_datetime(["2024-01-10", "2024-05-01"]),
        "end_date": pd.to_datetime(["2024-01-12", "2024-05-02"]),
    })
    truth = pd.DataFrame({
        "store_id": ["ST001", "ST003"],
        "start_date": pd.to_datetime(["2024-01-11", "2024-07-01"]),
        "end_date": pd.to_datetime(["2024-01-13", "2024-07-03"]),
    })
    result = evaluate_against_truth(detected, truth)
    assert result["caught"] == 1 and result["injected"] == 2
    assert result["recall"] == pytest.approx(0.5)
    assert result["false_alarms"] == 1
    assert result["precision"] == pytest.approx(0.5)
