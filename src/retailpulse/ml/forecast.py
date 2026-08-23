"""Demand forecasting: Holt-Winters written from scratch, plus a gradient
boosted correction, validated by walk-forward backtesting.

**Stage 1 - Holt-Winters (triple exponential smoothing).** Three numbers are
tracked and updated with every new day, each with its own learning rate:

    level_t     = alpha * (y_t / s_{t-m})   + (1 - alpha) * (level + trend)
    trend_t     = beta  * (level_t - level) + (1 - beta)  * trend
    season_t    = gamma * (y_t / level_t)   + (1 - gamma) * s_{t-m}

In words: *where are we*, *which way are we heading*, and *what does this
weekday usually do*. Seasonality is multiplicative because Saturday is "+38%",
not "+X rupees" - a fixed rupee uplift would be wrong in December and wrong
again in a growth year. The three rates are fitted by minimising one-step-ahead
squared error; nothing is hard-coded.

**Stage 2 - the residual model.** Exponential smoothing has no way to know
about Diwali, payday or a planned promotion. So a gradient boosted tree is
trained on what Holt-Winters *got wrong*, using only information a planner
genuinely has in advance: the calendar, the festival diary, the promo plan and
the state of the series at the moment the forecast is made. It is a direct
multi-horizon model: every feature is either a diary entry or a snapshot of
the series taken at the forecast origin, so nothing has to be fed back
recursively and errors cannot compound across the horizon.

**Stage 3 - honest evaluation.** Walk-forward backtesting: stand at a point in
the past, forecast 28 days with data available only up to that point, score
against what actually happened, then roll forward and repeat. Reported against
a seasonal-naive baseline via MASE, because a forecast that cannot beat
"same as last Tuesday" is not worth deploying.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor

from ..generate.calendar_effects import FIXED_HOLIDAYS, festival_dates


# --------------------------------------------------------------------------
# Holt-Winters
# --------------------------------------------------------------------------
@dataclass
class HoltWintersState:
    alpha: float
    beta: float
    gamma: float
    phi: float
    level: float
    trend: float
    season: np.ndarray
    sse: float = np.nan


class HoltWinters:
    """Triple exponential smoothing, additive or multiplicative seasonality."""

    def __init__(self, season_period: int = 7, seasonal: str = "multiplicative",
                 damped: bool = True):
        if seasonal not in ("additive", "multiplicative"):
            raise ValueError("seasonal must be 'additive' or 'multiplicative'")
        self.m = season_period
        self.seasonal = seasonal
        self.damped = damped
        self.state: HoltWintersState | None = None
        self.fitted_: np.ndarray | None = None
        self.y_: np.ndarray | None = None

    # ---- initialisation ------------------------------------------------
    def _initial_components(self, y: np.ndarray) -> tuple[float, float, np.ndarray]:
        m = self.m
        n_seasons = max(len(y) // m, 1)
        if n_seasons >= 2:
            season_means = np.array([y[i * m:(i + 1) * m].mean() for i in range(n_seasons)])
            level0 = float(season_means[0])
            trend0 = float((season_means[1] - season_means[0]) / m)
            # Average each phase across all complete seasons.
            indices = np.zeros(m)
            for phase in range(m):
                vals = [y[i * m + phase] / season_means[i]
                        if self.seasonal == "multiplicative"
                        else y[i * m + phase] - season_means[i]
                        for i in range(n_seasons) if i * m + phase < len(y)]
                indices[phase] = float(np.mean(vals)) if vals else (1.0 if self.seasonal == "multiplicative" else 0.0)
        else:
            level0, trend0 = float(y[:m].mean()), 0.0
            indices = np.ones(m) if self.seasonal == "multiplicative" else np.zeros(m)

        if self.seasonal == "multiplicative":
            indices = np.where(indices <= 0, 1.0, indices)
            indices = indices * m / indices.sum()  # normalise to average 1
        else:
            indices = indices - indices.mean()     # normalise to average 0
        return level0, trend0, indices

    # ---- recursion -----------------------------------------------------
    def _run(self, y: np.ndarray, alpha: float, beta: float, gamma: float,
             phi: float) -> tuple[np.ndarray, HoltWintersState]:
        m = self.m
        level, trend, season = self._initial_components(y)
        season = season.copy()
        fitted = np.zeros(len(y))
        mult = self.seasonal == "multiplicative"

        for t in range(len(y)):
            s_idx = t % m
            seasonal_component = season[s_idx]
            one_step = (level + phi * trend) * seasonal_component if mult else \
                       (level + phi * trend) + seasonal_component
            fitted[t] = one_step

            prev_level = level
            if mult:
                denom = seasonal_component if abs(seasonal_component) > 1e-8 else 1e-8
                level = alpha * (y[t] / denom) + (1 - alpha) * (prev_level + phi * trend)
                trend = beta * (level - prev_level) + (1 - beta) * phi * trend
                lvl = level if abs(level) > 1e-8 else 1e-8
                season[s_idx] = gamma * (y[t] / lvl) + (1 - gamma) * seasonal_component
            else:
                level = alpha * (y[t] - seasonal_component) + (1 - alpha) * (prev_level + phi * trend)
                trend = beta * (level - prev_level) + (1 - beta) * phi * trend
                season[s_idx] = gamma * (y[t] - level) + (1 - gamma) * seasonal_component

        sse = float(np.sum((y - fitted) ** 2))
        return fitted, HoltWintersState(alpha, beta, gamma, phi, float(level),
                                        float(trend), season, sse)

    # ---- fitting -------------------------------------------------------
    def fit(self, y: np.ndarray | pd.Series) -> "HoltWinters":
        y = np.asarray(y, dtype=float)
        if len(y) < 2 * self.m:
            raise ValueError(f"need at least {2 * self.m} observations to fit a "
                             f"seasonal model with period {self.m}")
        self.y_ = y

        def objective(params: np.ndarray) -> float:
            alpha, beta, gamma = params[:3]
            phi = params[3] if self.damped else 1.0
            if not (0 < alpha < 1 and 0 <= beta < 1 and 0 <= gamma < 1 and 0.5 <= phi <= 1):
                return np.inf
            try:
                fitted, state = self._run(y, alpha, beta, gamma, phi)
            except (FloatingPointError, ZeroDivisionError):
                return np.inf
            if not np.all(np.isfinite(fitted)):
                return np.inf
            return state.sse

        bounds = [(0.01, 0.99), (0.0001, 0.99), (0.0001, 0.99)]
        starts = [[0.2, 0.05, 0.2], [0.5, 0.01, 0.3], [0.05, 0.001, 0.05]]
        if self.damped:
            bounds.append((0.80, 1.0))
            starts = [s + [0.98] for s in starts]

        best, best_val = None, np.inf
        for start in starts:
            res = minimize(objective, np.array(start), method="L-BFGS-B", bounds=bounds)
            if res.fun < best_val:
                best, best_val = res, res.fun
        if best is None:
            raise RuntimeError("Holt-Winters optimisation failed")

        alpha, beta, gamma = best.x[:3]
        phi = best.x[3] if self.damped else 1.0
        self.fitted_, self.state = self._run(y, alpha, beta, gamma, phi)
        return self

    def forecast(self, horizon: int) -> np.ndarray:
        """Project the fitted state forward ``horizon`` steps."""
        if self.state is None or self.y_ is None:
            raise RuntimeError("Call fit() before forecast()")
        st, m, n = self.state, self.m, len(self.y_)
        out = np.zeros(horizon)
        for h in range(1, horizon + 1):
            # Damping: each further step adds a shrinking slice of the trend.
            damp = sum(st.phi ** i for i in range(1, h + 1)) if self.damped else h
            base = st.level + damp * st.trend
            s = st.season[(n + h - 1) % m]
            out[h - 1] = base * s if self.seasonal == "multiplicative" else base + s
        return out

    def params(self) -> dict:
        if self.state is None:
            raise RuntimeError("Call fit() first")
        st = self.state
        return {"alpha": round(st.alpha, 4), "beta": round(st.beta, 4),
                "gamma": round(st.gamma, 4), "phi": round(st.phi, 4),
                "seasonal": self.seasonal, "period": self.m,
                "final_level": round(st.level, 2), "final_trend": round(st.trend, 4)}


# --------------------------------------------------------------------------
# Calendar features the planner genuinely knows in advance
# --------------------------------------------------------------------------
def build_future_known_features(dates: pd.DatetimeIndex,
                                promo_calendar: pd.DataFrame | None = None) -> pd.DataFrame:
    """Features available *before* the day happens: diary, not observation."""
    d = pd.DatetimeIndex(dates)
    # The business calendar: movable festivals plus the fixed public holidays.
    # Both are diary entries a planner has years in advance, not observations.
    fest = festival_dates()
    fixed = pd.DatetimeIndex([
        pd.Timestamp(year=y, month=mth, day=day)
        for y in range(d.year.min() - 1, d.year.max() + 2)
        for mth, day in FIXED_HOLIDAYS.values()
    ])
    fest = pd.DatetimeIndex(sorted(set(fest).union(fixed)))
    if len(fest):
        deltas = np.array([[(day - f).days for f in fest] for day in d], dtype=float)
        days_to_festival = np.min(np.where(deltas <= 0, -deltas, np.inf), axis=1)
        days_since_festival = np.min(np.where(deltas >= 0, deltas, np.inf), axis=1)
    else:
        days_to_festival = np.full(len(d), 999.0)
        days_since_festival = np.full(len(d), 999.0)

    out = pd.DataFrame(
        {
            "date": d,
            "dow": d.dayofweek,
            "is_weekend": (d.dayofweek >= 5).astype(int),
            "month": d.month,
            "day_of_month": d.day,
            "week_of_year": d.isocalendar().week.astype(int).to_numpy(),
            "day_of_year": d.dayofyear,
            "is_payday_window": (d.day <= 5).astype(int),
            "is_month_end": (d.day >= 26).astype(int),
            "days_to_festival": np.clip(days_to_festival, 0, 60),
            "days_since_festival": np.clip(days_since_festival, 0, 60),
        }
    )
    # A smooth build-up ramp: 1.0 on the festival, decaying a fortnight out.
    out["festival_ramp"] = np.exp(-out["days_to_festival"] / 5.0)
    out["post_festival"] = np.exp(-out["days_since_festival"] / 3.0)

    if promo_calendar is not None and not promo_calendar.empty:
        promo = promo_calendar.copy()
        promo["date"] = pd.to_datetime(promo["date"])
        intensity = promo.groupby("date").agg(
            promo_products=("product_id", "nunique"),
            promo_depth=("discount_pct", "mean")).reset_index()
        out = out.merge(intensity, on="date", how="left")
    if "promo_products" not in out.columns:
        out["promo_products"] = 0.0
        out["promo_depth"] = 0.0
    out[["promo_products", "promo_depth"]] = out[["promo_products", "promo_depth"]].fillna(0.0)
    return out


def promo_calendar_from_promotions(promotions: pd.DataFrame) -> pd.DataFrame:
    """Explode promo campaigns into one row per (product, day)."""
    if promotions is None or promotions.empty:
        return pd.DataFrame(columns=["product_id", "date", "discount_pct"])
    frames = []
    for row in promotions.itertuples(index=False):
        days = pd.date_range(row.start_date, row.end_date, freq="D")
        frames.append(pd.DataFrame({"product_id": row.product_id, "date": days,
                                    "discount_pct": row.discount_pct}))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Hybrid forecaster
# --------------------------------------------------------------------------
@dataclass
class ForecastResult:
    forecast: pd.DataFrame
    hw_params: dict = field(default_factory=dict)
    residual_model_used: bool = False


class HybridForecaster:
    """Holt-Winters for structure, gradient boosting for everything it misses."""

    def __init__(self, season_period: int = 7, seasonal: str = "multiplicative",
                 damped: bool = True, use_residual_model: bool = True,
                 gbm_params: dict | None = None, random_state: int = 42):
        self.season_period = season_period
        self.seasonal = seasonal
        self.damped = damped
        self.use_residual_model = use_residual_model
        self.gbm_params = gbm_params or {"max_iter": 300, "learning_rate": 0.05,
                                         "max_depth": 3, "min_samples_leaf": 20}
        self.random_state = random_state
        self.hw: HoltWinters | None = None
        self.gbm: HistGradientBoostingRegressor | None = None
        self.feature_names_: list[str] = []
        self.history_: pd.DataFrame | None = None
        self.residuals_: np.ndarray | None = None
        self.value_col_: str = "revenue"

    # ---- residual training frame ---------------------------------------
    def _residual_frame(self, history: pd.DataFrame, residuals: np.ndarray,
                        promo_calendar: pd.DataFrame | None) -> pd.DataFrame:
        dates = pd.DatetimeIndex(history["date"])
        feats = build_future_known_features(dates, promo_calendar)
        feats["residual"] = residuals

        # State of the series at the forecast origin. During training, the
        # origin is simulated as "28 days before this row", so the model learns
        # with exactly the information it will have at prediction time.
        resid_series = pd.Series(residuals, index=dates)
        lagged = resid_series.shift(self.season_period * 4)  # 28 days back
        feats["origin_residual_mean_7"] = lagged.rolling(7, min_periods=1).mean().to_numpy()
        feats["origin_residual_mean_28"] = lagged.rolling(28, min_periods=1).mean().to_numpy()
        return feats

    def fit(self, history: pd.DataFrame, value_col: str = "revenue",
            promo_calendar: pd.DataFrame | None = None) -> "HybridForecaster":
        history = history.sort_values("date").reset_index(drop=True).copy()
        history["date"] = pd.to_datetime(history["date"])
        self.history_ = history
        y = history[value_col].to_numpy(dtype=float)

        self.value_col_ = value_col
        self.hw = HoltWinters(self.season_period, self.seasonal, self.damped).fit(y)
        residuals = y - self.hw.fitted_
        self.residuals_ = residuals

        if self.use_residual_model:
            frame = self._residual_frame(history, residuals, promo_calendar)
            # Skip the first season: the smoother is still finding its feet and
            # those residuals are initialisation noise, not signal.
            warmup = self.season_period * 4
            train = frame.iloc[warmup:].copy()
            self.feature_names_ = [c for c in train.columns if c not in ("date", "residual")]
            X = train[self.feature_names_].fillna(0.0)
            self.gbm = HistGradientBoostingRegressor(
                random_state=self.random_state, **self.gbm_params
            ).fit(X, train["residual"])
        return self

    def predict(self, horizon: int, promo_calendar: pd.DataFrame | None = None) -> pd.DataFrame:
        if self.hw is None or self.history_ is None:
            raise RuntimeError("Call fit() before predict()")
        last_date = self.history_["date"].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

        base = self.hw.forecast(horizon)
        out = pd.DataFrame({"date": future_dates, "hw_forecast": base})
        out["residual_adjustment"] = 0.0

        if self.gbm is not None:
            feats = build_future_known_features(future_dates, promo_calendar)
            # The forecast origin is "now", i.e. the tail of the fitted
            # residuals. Training used residuals lagged by 28 days, so the
            # model is never given information it would lack in production.
            tail = pd.Series(self.residuals_[-28:])
            feats["origin_residual_mean_7"] = float(tail.tail(7).mean())
            feats["origin_residual_mean_28"] = float(tail.mean())
            X = feats.reindex(columns=self.feature_names_).fillna(0.0)
            out["residual_adjustment"] = self.gbm.predict(X)

        out["forecast"] = np.clip(out["hw_forecast"] + out["residual_adjustment"], 0.0, None)
        return out

    def params(self) -> dict:
        return self.hw.params() if self.hw else {}


# --------------------------------------------------------------------------
# Accuracy metrics
# --------------------------------------------------------------------------
def forecast_metrics(actual: np.ndarray, predicted: np.ndarray,
                     insample: np.ndarray, season_period: int = 7) -> dict:
    """MAE/RMSE/MAPE/sMAPE plus MASE, scaled by the seasonal-naive error."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = predicted - actual

    naive_err = np.abs(np.asarray(insample[season_period:], dtype=float)
                       - np.asarray(insample[:-season_period], dtype=float))
    scale = float(np.mean(naive_err)) if len(naive_err) and np.mean(naive_err) > 0 else np.nan

    denom = (np.abs(actual) + np.abs(predicted)) / 2
    return {
        "mae": round(float(np.mean(np.abs(err))), 2),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 2),
        "mape_pct": round(float(np.mean(np.abs(err / np.where(actual == 0, np.nan, actual))) * 100), 3),
        "smape_pct": round(float(np.mean(np.abs(err) / np.where(denom == 0, np.nan, denom)) * 100), 3),
        "mase": round(float(np.mean(np.abs(err)) / scale), 4) if scale and not np.isnan(scale) else np.nan,
        "bias_pct": round(float(np.sum(err) / np.sum(actual) * 100), 3),
    }


def seasonal_naive_forecast(history: np.ndarray, horizon: int, season_period: int = 7) -> np.ndarray:
    """The baseline every forecast must beat: 'same as last week'."""
    history = np.asarray(history, dtype=float)
    reps = int(np.ceil(horizon / season_period))
    return np.tile(history[-season_period:], reps)[:horizon]


def walk_forward_backtest(series: pd.DataFrame, value_col: str = "revenue",
                          horizon: int = 28, folds: int = 4, season_period: int = 7,
                          promo_calendar: pd.DataFrame | None = None,
                          use_residual_model: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Roll an expanding training window forward and score each fold.

    Returns ``(fold_metrics, predictions)``. Each fold trains only on data
    strictly before its own forecast window, so no fold can see its answer.
    """
    series = series.sort_values("date").reset_index(drop=True).copy()
    series["date"] = pd.to_datetime(series["date"])
    n = len(series)
    if n < horizon * (folds + 2):
        raise ValueError("series too short for the requested backtest layout")

    rows, preds = [], []
    for fold in range(folds, 0, -1):
        cut = n - fold * horizon
        train = series.iloc[:cut]
        test = series.iloc[cut:cut + horizon]
        if len(test) < horizon:
            continue

        model = HybridForecaster(season_period=season_period,
                                 use_residual_model=use_residual_model).fit(
            train, value_col=value_col, promo_calendar=promo_calendar)
        fc = model.predict(horizon, promo_calendar=promo_calendar)

        actual = test[value_col].to_numpy(dtype=float)
        insample = train[value_col].to_numpy(dtype=float)
        naive = seasonal_naive_forecast(insample, horizon, season_period)

        hybrid_m = forecast_metrics(actual, fc["forecast"].to_numpy(), insample, season_period)
        hw_m = forecast_metrics(actual, fc["hw_forecast"].to_numpy(), insample, season_period)
        naive_m = forecast_metrics(actual, naive, insample, season_period)

        for name, m in (("hybrid", hybrid_m), ("holt_winters", hw_m), ("seasonal_naive", naive_m)):
            rows.append({"fold": folds - fold + 1, "model": name,
                         "train_end": train["date"].max().date(),
                         "test_start": test["date"].min().date(),
                         "test_end": test["date"].max().date(), **m})

        preds.append(pd.DataFrame({
            "fold": folds - fold + 1,
            "date": test["date"].to_numpy(),
            "actual": actual,
            "hybrid": fc["forecast"].to_numpy(),
            "holt_winters": fc["hw_forecast"].to_numpy(),
            "seasonal_naive": naive,
        }))

    return pd.DataFrame(rows), (pd.concat(preds, ignore_index=True) if preds else pd.DataFrame())
