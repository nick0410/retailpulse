"""BG/NBD and Gamma-Gamma.

The headline test simulates customers from the *exact* process the model
assumes, then checks that maximum likelihood recovers the parameters that
generated them. If that fails, the likelihood is wrong - no amount of nice
output would save it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from dataclasses import replace

from retailpulse.analytics.clv import (BetaGeoFitter, GammaGammaFitter,
                                       customer_lifetime_value,
                                       summary_from_transactions, validate_holdout)
from retailpulse.generate.synthetic import _build_customers, _simulate_customer_transactions


@pytest.fixture(scope="module")
def bgnbd_sample(sim_config):
    """Customers drawn straight from BG/NBD with known parameters."""
    cfg = replace(sim_config, n_customers=3500, start_date="2021-01-01", end_date="2024-06-30")
    rng = np.random.default_rng(11)
    customers = _build_customers(rng, cfg)
    txns, truth = _simulate_customer_transactions(rng, cfg, customers)
    fact = txns.rename(columns={"date": "date"}).copy()
    fact["line_id"] = np.arange(len(fact))
    fact["line_amount"] = 1000.0
    fact["gross_margin"] = 300.0
    return cfg, fact, truth


@pytest.fixture(scope="module")
def fitted_bgnbd(bgnbd_sample):
    cfg, fact, truth = bgnbd_sample
    summary = summary_from_transactions(fact, observation_end=pd.Timestamp(cfg.end_date))
    model = BetaGeoFitter().fit(summary["frequency"], summary["recency"], summary["T"])
    return cfg, summary, model, truth


# --------------------------------------------------------------------------
# Summary construction
# --------------------------------------------------------------------------
def test_summary_counts_repeat_purchases_only():
    """frequency is repeat visits; the first purchase is a birth, not an event."""
    fact = pd.DataFrame({
        "customer_id": ["C1"] * 3 + ["C2"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01", "2024-01-10"]),
        "line_amount": [100.0, 200.0, 300.0, 500.0],
        "line_id": [1, 2, 3, 4],
        "gross_margin": [10.0, 20.0, 30.0, 50.0],
    })
    s = summary_from_transactions(fact, observation_end=pd.Timestamp("2024-03-01"))
    c1 = s[s["customer_id"] == "C1"].iloc[0]
    assert c1["frequency"] == 2                       # three visits, two repeats
    assert c1["recency"] == pytest.approx(31 / 7)     # Jan 1 -> Feb 1
    assert c1["T"] == pytest.approx(60 / 7)           # Jan 1 -> Mar 1
    assert c1["monetary_value"] == pytest.approx(250.0)  # mean of the two repeats

    c2 = s[s["customer_id"] == "C2"].iloc[0]
    assert c2["frequency"] == 0
    assert c2["monetary_value"] == 0.0


def test_same_day_lines_collapse_into_one_trip():
    fact = pd.DataFrame({
        "customer_id": ["C1"] * 4,
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01", "2024-02-01"]),
        "line_amount": [100.0, 100.0, 100.0, 400.0],
        "line_id": [1, 2, 3, 4],
        "gross_margin": [10.0] * 4,
    })
    s = summary_from_transactions(fact, observation_end=pd.Timestamp("2024-03-01"))
    assert s["frequency"].iat[0] == 1  # two trips -> one repeat


# --------------------------------------------------------------------------
# Parameter recovery
# --------------------------------------------------------------------------
def test_bgnbd_recovers_the_generating_parameters(fitted_bgnbd):
    cfg, _summary, model, _truth = fitted_bgnbd
    p = model.params

    # Purchase-rate mean r/alpha is the well-identified quantity; the
    # individual r and alpha are only identified up to their ratio in small
    # samples, so it gets the tighter bound.
    assert p.r / p.alpha == pytest.approx(cfg.bgnbd_r / cfg.bgnbd_alpha, rel=0.20)
    assert a_over_ab(p.a, p.b) == pytest.approx(
        a_over_ab(cfg.bgnbd_a, cfg.bgnbd_b), rel=0.30)
    assert p.r == pytest.approx(cfg.bgnbd_r, rel=0.35)
    assert p.alpha == pytest.approx(cfg.bgnbd_alpha, rel=0.35)


def a_over_ab(a: float, b: float) -> float:
    """Mean dropout probability of the Beta(a, b) mixing distribution."""
    return a / (a + b)


def test_probability_alive_tracks_the_hidden_truth(fitted_bgnbd):
    """P(alive) must rank real survivors above the customers who really left."""
    from sklearn.metrics import roc_auc_score

    _cfg, summary, model, truth = fitted_bgnbd
    p_alive = model.conditional_probability_alive(
        summary["frequency"], summary["recency"], summary["T"])
    merged = summary[["customer_id"]].assign(p_alive=p_alive).merge(truth, on="customer_id")
    auc = roc_auc_score(merged["true_alive_at_end"], merged["p_alive"])
    assert auc > 0.80, f"P(alive) barely separates alive from dead (AUC {auc:.3f})"


def test_probability_alive_is_a_probability(fitted_bgnbd):
    _cfg, summary, model, _truth = fitted_bgnbd
    p = model.conditional_probability_alive(
        summary["frequency"], summary["recency"], summary["T"])
    assert np.all((p >= 0) & (p <= 1))
    # A customer who has never repeated cannot be declared dead by this model.
    never_repeated = summary["frequency"] == 0
    assert np.allclose(p[never_repeated.to_numpy()], 1.0)


def test_being_overdue_lowers_probability_alive(fitted_bgnbd):
    """Same history, longer silence -> lower P(alive). This is the core claim."""
    _cfg, _summary, model, _truth = fitted_bgnbd
    freq = np.array([10.0, 10.0, 10.0])
    T = np.array([100.0, 100.0, 100.0])
    recency = np.array([95.0, 60.0, 25.0])  # last seen recently -> long ago
    p = model.conditional_probability_alive(freq, recency, T)
    assert p[0] > p[1] > p[2]


def test_expected_transactions_grow_with_the_horizon(fitted_bgnbd):
    _cfg, summary, model, _truth = fitted_bgnbd
    x = summary["frequency"].to_numpy()[:200]
    t_x = summary["recency"].to_numpy()[:200]
    T = summary["T"].to_numpy()[:200]
    short = model.conditional_expected_transactions(4.0, x, t_x, T)
    long = model.conditional_expected_transactions(52.0, x, t_x, T)
    assert np.all(short >= -1e-9)
    assert np.all(long >= short - 1e-9)


# --------------------------------------------------------------------------
# Gamma-Gamma
# --------------------------------------------------------------------------
def test_gamma_gamma_shrinks_thin_evidence_towards_the_market():
    """One observed basket should barely move the estimate; fifty should."""
    rng = np.random.default_rng(3)
    n = 3000
    nu = rng.gamma(shape=4.0, scale=1 / 800.0, size=n)
    freq = rng.integers(1, 25, size=n).astype(float)
    monetary = np.array([rng.gamma(5.0 * f, 1 / (nu[i] * f)) / 1.0
                         for i, f in enumerate(freq)]) / freq
    model = GammaGammaFitter().fit(freq, monetary)

    population = model.params.v * model.params.p / (model.params.q - 1)
    extreme = population * 4

    thin = model.conditional_expected_average_profit(np.array([1.0]), np.array([extreme]))[0]
    thick = model.conditional_expected_average_profit(np.array([50.0]), np.array([extreme]))[0]
    assert abs(thin - population) < abs(thick - population)
    assert population < thin < thick <= extreme


def test_gamma_gamma_needs_repeat_customers():
    with pytest.raises(ValueError):
        GammaGammaFitter().fit(np.zeros(50), np.zeros(50))


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------
def test_clv_output_is_well_formed(fact):
    summary = summary_from_transactions(fact)
    bgf = BetaGeoFitter().fit(summary["frequency"], summary["recency"], summary["T"])
    ggf = GammaGammaFitter().fit(summary["frequency"], summary["monetary_value"])
    clv = customer_lifetime_value(bgf, ggf, summary, months=12)

    assert len(clv) == len(summary)
    assert (clv["clv_12m"] >= 0).all()
    assert clv["prob_alive"].between(0, 1).all()
    assert clv["clv_12m"].is_monotonic_decreasing  # sorted best-first
    assert clv["clv_decile"].nunique() == 10


def test_holdout_validation_is_reasonably_accurate(fact):
    """Fit on the past, predict a window the model has never seen."""
    end = pd.to_datetime(fact["date"]).max()
    calibration_end = end - pd.Timedelta(days=120)
    summary = summary_from_transactions(fact, observation_end=calibration_end,
                                        calibration_end=calibration_end)
    bgf = BetaGeoFitter().fit(summary["frequency"], summary["recency"], summary["T"])
    result = validate_holdout(bgf, fact, calibration_end, end)

    assert result["customers_scored"] > 100
    # Aggregate demand is what a planner acts on; 25% is a generous bound that
    # still fails loudly if the model is broken.
    assert abs(result["aggregate_error_pct"]) < 25.0
    assert result["correlation"] > 0.35
