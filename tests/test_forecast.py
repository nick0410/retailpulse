"""Holt-Winters, the hybrid forecaster, and the honesty of the backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from retailpulse.ml.forecast import (HoltWinters, HybridForecaster, build_future_known_features,
                                     forecast_metrics, seasonal_naive_forecast,
                                     walk_forward_backtest)


# --------------------------------------------------------------------------
# Holt-Winters on a series whose answer is known
# --------------------------------------------------------------------------
@pytest.fixture
def known_series() -> tuple[np.ndarray, np.ndarray]:
    """level 1000, +2/day trend, a fixed weekly shape, a little noise."""
    n = 420
    rng = np.random.default_rng(5)
    weekly = np.array([0.85, 0.88, 0.92, 0.97, 1.15, 1.35, 1.28])
    t = np.arange(n)
    clean = (1000 + 2.0 * t) * weekly[t % 7]
    noisy = clean * (1 + rng.normal(0, 0.02, n))
    return noisy, clean


def test_holt_winters_learns_the_weekly_shape(known_series):
    y, _clean = known_series
    model = HoltWinters(season_period=7, seasonal="multiplicative", damped=False).fit(y)
    season = model.state.season
    # Multiplicative indices are normalised to average 1.
    assert season.mean() == pytest.approx(1.0, abs=0.05)
    # Saturday (index 5) is the planted peak; Monday (index 0) the trough.
    assert season.argmax() == 5
    assert season.argmin() == 0


def test_holt_winters_forecast_is_accurate_on_a_clean_series(known_series):
    y, _clean = known_series
    train, test = y[:-28], y[-28:]
    model = HoltWinters(7, "multiplicative", damped=False).fit(train)
    prediction = model.forecast(28)
    mape = np.mean(np.abs(prediction - test) / test) * 100
    assert mape < 6.0, f"MAPE {mape:.2f}% on a series with a known structure"


def test_holt_winters_beats_the_naive_baseline(known_series):
    y, _clean = known_series
    train, test = y[:-28], y[-28:]
    hw = HoltWinters(7, "multiplicative", damped=False).fit(train).forecast(28)
    naive = seasonal_naive_forecast(train, 28, 7)
    assert np.mean(np.abs(hw - test)) < np.mean(np.abs(naive - test))


def test_smoothing_parameters_stay_in_bounds(known_series):
    y, _clean = known_series
    model = HoltWinters(7, "multiplicative", damped=True).fit(y)
    p = model.params()
    assert 0 < p["alpha"] < 1
    assert 0 <= p["beta"] < 1
    assert 0 <= p["gamma"] < 1
    assert 0.5 <= p["phi"] <= 1.0


def test_additive_mode_also_works(known_series):
    y, _clean = known_series
    model = HoltWinters(7, "additive", damped=False).fit(y)
    assert model.state.season.mean() == pytest.approx(0.0, abs=y.mean() * 0.05)
    assert np.all(np.isfinite(model.forecast(14)))


def test_refuses_a_series_too_short_to_have_a_season():
    with pytest.raises(ValueError):
        HoltWinters(season_period=7).fit(np.arange(10, dtype=float))


def test_damping_pulls_a_long_horizon_back(known_series):
    """A damped trend must not run away over a long horizon."""
    y, _clean = known_series
    undamped = HoltWinters(7, "multiplicative", damped=False).fit(y).forecast(180)
    damped = HoltWinters(7, "multiplicative", damped=True).fit(y).forecast(180)
    assert damped[-1] <= undamped[-1] + 1e-6


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def test_seasonal_naive_repeats_the_last_season():
    history = np.arange(1, 15, dtype=float)  # 14 values
    out = seasonal_naive_forecast(history, 10, 7)
    assert list(out[:7]) == list(history[-7:])
    assert list(out[7:]) == list(history[-7:][:3])


def test_metric_formulas_match_hand_calculation():
    actual = np.array([100.0, 200.0, 300.0])
    predicted = np.array([110.0, 190.0, 330.0])
    insample = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])  # naive gap = 10

    # forecast_metrics rounds for reporting, so compare within that rounding.
    m = forecast_metrics(actual, predicted, insample, season_period=7)
    assert m["mae"] == pytest.approx((10 + 10 + 30) / 3, abs=0.005)
    assert m["rmse"] == pytest.approx(np.sqrt((100 + 100 + 900) / 3), abs=0.005)
    assert m["mape_pct"] == pytest.approx(np.mean([0.10, 0.05, 0.10]) * 100, abs=0.0005)
    # in-sample seasonal-naive MAE is 70 (each value differs from t-7 by 70)
    assert m["mase"] == pytest.approx(((10 + 10 + 30) / 3) / 70.0, abs=0.0005)
    assert m["bias_pct"] == pytest.approx(100 * (10 - 10 + 30) / 600, abs=0.0005)


def test_a_perfect_forecast_scores_zero():
    actual = np.array([5.0, 6.0, 7.0])
    insample = np.arange(1, 20, dtype=float)
    m = forecast_metrics(actual, actual.copy(), insample, season_period=7)
    assert m["mae"] == 0 and m["rmse"] == 0 and m["mase"] == 0
    assert m["mape_pct"] == 0 and m["bias_pct"] == 0


# --------------------------------------------------------------------------
# Feature honesty
# --------------------------------------------------------------------------
def test_future_features_use_only_the_diary():
    """Every forecast feature must be knowable before the day happens."""
    dates = pd.date_range("2024-10-20", periods=30, freq="D")
    feats = build_future_known_features(dates)
    assert len(feats) == 30
    assert feats.notna().all().all()
    # Diwali 2024 fell on 31 October; the ramp must peak there.
    peak = feats.loc[feats["festival_ramp"].idxmax(), "date"]
    assert peak == pd.Timestamp("2024-10-31")
    assert set(feats["is_payday_window"].unique()) <= {0, 1}


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------
def test_backtest_never_trains_on_its_own_test_window(star):
    daily = star["mart_daily_total"]
    folds, preds = walk_forward_backtest(daily, "revenue", horizon=21, folds=3)
    assert not folds.empty
    for _, row in folds.iterrows():
        assert pd.Timestamp(row["train_end"]) < pd.Timestamp(row["test_start"]), \
            "a fold trained on data from its own test window"
    assert folds["fold"].nunique() == 3
    assert set(folds["model"]) == {"hybrid", "holt_winters", "seasonal_naive"}


def test_hybrid_beats_the_naive_baseline_on_real_data(star):
    daily = star["mart_daily_total"]
    folds, _preds = walk_forward_backtest(daily, "revenue", horizon=21, folds=3)
    avg = folds.groupby("model")["mase"].mean()
    assert avg["hybrid"] < avg["seasonal_naive"], \
        f"hybrid MASE {avg['hybrid']:.3f} did not beat naive {avg['seasonal_naive']:.3f}"


def test_forecast_is_non_negative_and_the_right_length(star):
    daily = star["mart_daily_total"]
    model = HybridForecaster(season_period=7).fit(daily, value_col="revenue")
    out = model.predict(28)
    assert len(out) == 28
    assert (out["forecast"] >= 0).all()
    assert out["date"].min() > pd.to_datetime(daily["date"]).max()
    assert (out["date"].diff().dropna() == pd.Timedelta(days=1)).all()


def test_backtest_refuses_a_series_that_is_too_short():
    tiny = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=40),
                         "revenue": np.arange(40, dtype=float)})
    with pytest.raises(ValueError):
        walk_forward_backtest(tiny, "revenue", horizon=28, folds=4)
