"""Robust anomaly detection on store-level daily sales.

A naive "flag anything more than 3 standard deviations from the mean" detector
fires every Saturday and every Diwali, because those days *are* far from the
mean - by design. So the structure has to come out first, and only what is
left gets judged.

The model is multiplicative, and each piece answers one question:

    sales[store, day] = level[store, day] x market[day] x surprise[store, day]

* **level** - a centred rolling *median* of the store's own sales. A median,
  not a mean, so a two-day stockout cannot drag the baseline down with it.
* **market** - the median, across all stores, of each store's sales relative
  to its own level. This is the chain moving together: weekends, paydays,
  Diwali, a nationwide promo. Taking the median across stores means one
  store's disaster cannot move it.
* **surprise** - the leftover. A store that dropped 60% on a day the rest of
  the chain was flat has a small surprise factor; a store that dropped 60% on
  a day the whole chain dropped 60% has none.

The surprise is scored in logs with a *robust z-score* built on the median
absolute deviation:

    z = 0.6745 * (log_surprise - median) / MAD

The 0.6745 constant makes MAD comparable to a standard deviation for normal
data while staying immune to the outliers we are hunting. This is why the
detector ignores festivals it has never been told about, and still catches a
single-store stockout on an ordinary Tuesday.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

MAD_TO_SIGMA = 0.6745
RATIO_FLOOR = 1e-3


def _centred_rolling_median(y: pd.Series, window: int) -> pd.Series:
    if window % 2 == 0:
        window += 1
    trend = y.rolling(window, center=True, min_periods=max(3, window // 3)).median()
    return trend.bfill().ffill()


def _to_contiguous_calendar(series: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """A day with no sales at all is *absent* from the mart, not zero.

    A total stockout is exactly such a day, so the series is re-indexed onto a
    complete calendar and the gaps become explicit zeros.
    """
    out = series.sort_values("date").copy()
    out["date"] = pd.to_datetime(out["date"])
    full = pd.date_range(out["date"].min(), out["date"].max(), freq="D")
    out = out.set_index("date").reindex(full).rename_axis("date").reset_index()
    out[value_col] = out[value_col].fillna(0.0)
    return out


def decompose(series: pd.DataFrame, value_col: str = "revenue",
              period: int = 7, trend_window: int | None = None) -> pd.DataFrame:
    """Robust additive decomposition of a single daily series.

    Used for the single-series case and for the explanatory charts; the
    cross-store detector below prefers the market-factor model.
    """
    out = _to_contiguous_calendar(series, value_col)
    y = out[value_col].astype(float)
    window = trend_window or (2 * period + 1)

    out["trend"] = _centred_rolling_median(y, window)
    detrended = y - out["trend"]

    phase = np.arange(len(out)) % period
    seasonal_map = pd.Series(detrended.to_numpy()).groupby(phase).median()
    # Centre the seasonal component so it does not absorb part of the level.
    seasonal_map = seasonal_map - seasonal_map.mean()
    out["seasonal"] = seasonal_map.reindex(phase).to_numpy()
    out["residual"] = y - out["trend"] - out["seasonal"]
    return out


def robust_zscore(x: pd.Series) -> pd.Series:
    """Median/MAD based z-score, falling back to sigma when MAD degenerates."""
    med = x.median()
    mad = (x - med).abs().median()
    if mad == 0 or np.isnan(mad):
        std = x.std(ddof=0)
        if std == 0 or np.isnan(std):
            return pd.Series(np.zeros(len(x)), index=x.index)
        return (x - med) / std
    return MAD_TO_SIGMA * (x - med) / mad


def estimate_dispersion(y: np.ndarray, expected: np.ndarray, trim: float = 0.05) -> float:
    """Method-of-moments negative-binomial dispersion ``k``.

    For counts, ``Var = lambda + lambda^2 / k``: pure Poisson is ``k -> inf``,
    and small ``k`` means the day-to-day scatter is much wider than Poisson.
    The largest deviations are trimmed off first, otherwise the very incidents
    we are hunting would inflate the dispersion and hide themselves.
    """
    resid_sq = (y - expected) ** 2
    excess = resid_sq - expected
    keep = np.abs(y - expected) <= np.quantile(np.abs(y - expected), 1 - trim)
    num = float(np.mean(expected[keep] ** 2))
    den = float(np.mean(excess[keep]))
    if den <= 0 or num <= 0:
        return np.inf  # not overdispersed: plain Poisson is fine
    return max(num / den, 1e-3)


def benjamini_hochberg(pvalues: np.ndarray, q: float = 0.01) -> np.ndarray:
    """Flag p-values that survive Benjamini-Hochberg FDR control at level ``q``.

    With ~13,000 store-days on test, a raw p < 0.001 rule would still produce
    a dozen false alerts every run. BH fixes the *expected share* of wrong
    alerts instead of the per-test error, which is the number an ops team
    actually cares about.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    if m == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    passing = ranked <= q * np.arange(1, m + 1) / m
    if not passing.any():
        return np.zeros(m, dtype=bool)
    cutoff = ranked[np.max(np.flatnonzero(passing))]
    return p <= cutoff


def detect_anomalies(daily_store: pd.DataFrame, value_col: str = "revenue",
                     period: int = 7, z_threshold: float = 3.5,
                     group_col: str = "store_id",
                     use_market_factor: bool = True,
                     method: str = "robust_z", fdr_q: float = 0.01) -> pd.DataFrame:
    """Score every (store, date) and return the fully annotated series.

    ``method='robust_z'`` scores the log-surprise with a median/MAD z-score -
    the right tool for a continuous series such as revenue.

    ``method='nb_tail'`` treats the series as counts and asks the sharper
    question: *if this store were behaving normally today, how likely is a day
    at least this extreme?* The count is compared against a negative-binomial
    centred on the expected value, and the resulting p-values go through
    Benjamini-Hochberg so the alert list has a controlled false-discovery
    rate. Use it for transaction counts, where Poisson noise at ~17 baskets a
    day is far too large for a z-score to reason about correctly.
    """
    frames = []
    for key, grp in daily_store.groupby(group_col):
        d = _to_contiguous_calendar(grp[["date", value_col]].copy(), value_col)
        y = d[value_col].astype(float)
        d["trend"] = _centred_rolling_median(y, 2 * period + 1).clip(lower=1e-9)
        d["ratio"] = (y / d["trend"]).clip(lower=RATIO_FLOOR)
        d[group_col] = key
        frames.append(d)
    panel = pd.concat(frames, ignore_index=True)

    n_groups = panel[group_col].nunique()
    if use_market_factor and n_groups >= 3:
        # The chain's common movement for the day, immune to any one store.
        market = panel.groupby("date")["ratio"].median().rename("market_factor")
        panel = panel.merge(market, on="date", how="left")
    else:
        # Single series: fall back to a weekday seasonal index.
        panel["market_factor"] = 1.0
        weekday_index = panel.groupby(panel["date"].dt.dayofweek)["ratio"].median()
        panel["market_factor"] = panel["date"].dt.dayofweek.map(weekday_index).to_numpy()

    panel["market_factor"] = panel["market_factor"].clip(lower=RATIO_FLOOR)
    panel["expected"] = panel["trend"] * panel["market_factor"]
    panel["surprise"] = (panel["ratio"] / panel["market_factor"]).clip(lower=RATIO_FLOOR)
    panel["log_surprise"] = np.log(panel["surprise"])

    # Score each store against its own history of surprises.
    panel["robust_z"] = (panel.groupby(group_col)["log_surprise"]
                         .transform(lambda s: robust_zscore(s)))
    panel["residual"] = panel[value_col] - panel["expected"]

    if method == "nb_tail":
        y = panel[value_col].to_numpy(dtype=float)
        lam = np.clip(panel["expected"].to_numpy(dtype=float), 1e-6, None)
        k = estimate_dispersion(y, lam)
        if np.isinf(k):
            lower = poisson.cdf(y, lam)
            upper = poisson.sf(y - 1, lam)
        else:
            nb_p = k / (k + lam)
            lower = nbinom.cdf(y, k, nb_p)
            upper = nbinom.sf(y - 1, k, nb_p)
        panel["p_value"] = np.clip(2.0 * np.minimum(lower, upper), 0.0, 1.0)
        panel["dispersion_k"] = k
        panel["is_anomaly"] = benjamini_hochberg(panel["p_value"].to_numpy(), q=fdr_q)
        # Direction comes from the tail that fired, not from the z-score.
        panel["robust_z"] = np.where(y < lam, -panel["robust_z"].abs(), panel["robust_z"].abs())
    elif method == "robust_z":
        panel["p_value"] = np.nan
        panel["is_anomaly"] = panel["robust_z"].abs() >= z_threshold
    else:
        raise ValueError(f"unknown method {method!r}; expected 'robust_z' or 'nb_tail'")

    panel["anomaly_direction"] = np.where(
        ~panel["is_anomaly"], "normal", np.where(panel["robust_z"] < 0, "dip", "spike")
    )
    panel["deviation_pct"] = np.round(
        100 * (panel[value_col] - panel["expected"]) / panel["expected"].replace(0, np.nan), 2
    )
    return panel.sort_values([group_col, "date"]).reset_index(drop=True)


def anomaly_events(scored: pd.DataFrame, group_col: str = "store_id",
                   value_col: str = "revenue") -> pd.DataFrame:
    """Collapse consecutive flagged days into incidents an ops team can act on."""
    flagged = scored[scored["is_anomaly"]].copy()
    if flagged.empty:
        return pd.DataFrame(columns=[group_col, "start_date", "end_date", "days",
                                     "direction", "peak_z", "revenue_impact"])
    flagged = flagged.sort_values([group_col, "date"])
    gap = flagged.groupby(group_col)["date"].diff().dt.days.fillna(999)
    direction_change = flagged["anomaly_direction"] != flagged.groupby(group_col)["anomaly_direction"].shift()
    flagged["event_id"] = ((gap > 1) | direction_change).cumsum()

    events = flagged.groupby([group_col, "event_id"]).agg(
        start_date=("date", "min"),
        end_date=("date", "max"),
        days=("date", "count"),
        direction=("anomaly_direction", "first"),
        peak_z=("robust_z", lambda s: s.iloc[int(np.argmax(np.abs(s.to_numpy())))]),
        observed=(value_col, "sum"),
        expected=("expected", "sum"),
    ).reset_index()
    events["revenue_impact"] = np.round(events["observed"] - events["expected"], 2)
    events["peak_z"] = np.round(events["peak_z"], 2)
    return (events.drop(columns=["event_id"])
            .sort_values("peak_z", key=lambda s: s.abs(), ascending=False)
            .reset_index(drop=True))


def evaluate_against_truth(detected_events: pd.DataFrame, truth: pd.DataFrame,
                           group_col: str = "store_id") -> dict:
    """Score the detector against the incidents the simulator actually injected.

    A truth incident counts as *caught* if a detected event for the same store
    overlaps its window; a detected event is a *false alarm* if it overlaps no
    injected incident.
    """
    truth = truth.copy()
    truth["start_date"] = pd.to_datetime(truth["start_date"])
    truth["end_date"] = pd.to_datetime(truth["end_date"])
    det = detected_events.copy()
    if det.empty:
        return {"injected": int(len(truth)), "caught": 0, "recall": 0.0,
                "detected_events": 0, "false_alarms": 0, "precision": 0.0}
    det["start_date"] = pd.to_datetime(det["start_date"])
    det["end_date"] = pd.to_datetime(det["end_date"])

    caught = 0
    matched_det: set[int] = set()
    for t in truth.itertuples(index=False):
        overlap = det[
            (det[group_col] == getattr(t, group_col))
            & (det["start_date"] <= t.end_date)
            & (det["end_date"] >= t.start_date)
        ]
        if len(overlap):
            caught += 1
            matched_det.update(overlap.index.tolist())

    return {
        "injected": int(len(truth)),
        "caught": int(caught),
        "recall": round(caught / max(len(truth), 1), 4),
        "detected_events": int(len(det)),
        "false_alarms": int(len(det) - len(matched_det)),
        "precision": round(len(matched_det) / max(len(det), 1), 4),
    }
