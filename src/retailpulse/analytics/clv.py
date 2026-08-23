"""Probabilistic Customer Lifetime Value: BG/NBD + Gamma-Gamma, fitted from
scratch with scipy.

**The problem.** In a non-contractual business (a shop, not a subscription)
nobody tells you they have churned. A customer who has not visited for three
months is either dead or simply slow. You have to infer which.

**BG/NBD (Beta-Geometric / Negative Binomial Distribution)** answers it with
two coupled stories:

* *While alive*, a customer buys at a Poisson rate lambda. Rates differ across
  people, and are spread as ``Gamma(r, alpha)``.
* *After each purchase*, the customer flips a coin and churns for good with
  probability p. Those probabilities are spread as ``Beta(a, b)``.

Every customer is then summarised by just three numbers - ``x`` (repeat
purchases), ``t_x`` (age at last purchase) and ``T`` (age now) - and the four
population parameters are estimated by maximum likelihood. The likelihood has
a closed form (Fader, Hardie & Lee 2005):

    L = A1 * A2 * (A3 + 1{x>0} * A4)

which this module evaluates in logs for numerical safety.

**Gamma-Gamma** then models *how much* a customer spends per visit, assuming
spend is independent of frequency, and shrinks each customer's observed
average towards the population mean in proportion to how little evidence they
have given us.

CLV = expected transactions over the horizon x expected spend per transaction,
discounted month by month.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, hyp2f1, logsumexp

DAYS_PER_WEEK = 7.0
WEEKS_PER_MONTH = 365.25 / 12 / DAYS_PER_WEEK  # ~4.348


# --------------------------------------------------------------------------
# RFM summary in the shape the models need
# --------------------------------------------------------------------------
def summary_from_transactions(fact_sales: pd.DataFrame,
                              observation_end: pd.Timestamp | None = None,
                              calibration_end: pd.Timestamp | None = None,
                              freq_days: float = DAYS_PER_WEEK) -> pd.DataFrame:
    """Collapse the sales fact into the (x, t_x, T, monetary) summary.

    ``frequency`` counts *repeat* purchases only (the first ever purchase is
    the customer's birth, not an event to be predicted). ``monetary_value`` is
    the mean value of those repeat purchases, which is what Gamma-Gamma
    assumes. Multiple lines on the same day for the same customer collapse
    into one shopping trip.
    """
    df = fact_sales[fact_sales["customer_id"].notna()].copy()
    df["date"] = pd.to_datetime(df["date"])
    if calibration_end is not None:
        df = df[df["date"] <= pd.Timestamp(calibration_end)]
    end = pd.Timestamp(observation_end) if observation_end is not None else df["date"].max()

    trips = (df.groupby(["customer_id", "date"], as_index=False)["line_amount"].sum()
             .rename(columns={"line_amount": "trip_value"}))
    trips = trips.sort_values(["customer_id", "date"])

    grouped = trips.groupby("customer_id")
    first = grouped["date"].min()
    last = grouped["date"].max()
    n_trips = grouped["date"].count()

    total_value = grouped["trip_value"].sum()
    first_value = grouped["trip_value"].first()
    repeat_count = (n_trips - 1).clip(lower=0)
    repeat_value = total_value - first_value
    monetary = np.where(repeat_count > 0, repeat_value / repeat_count.replace(0, np.nan), 0.0)

    summary = pd.DataFrame(
        {
            "customer_id": first.index,
            "frequency": repeat_count.to_numpy().astype(float),
            "recency": ((last - first).dt.days / freq_days).to_numpy(),
            "T": ((end - first).dt.days / freq_days).to_numpy(),
            "monetary_value": np.nan_to_num(monetary),
            "total_value": total_value.to_numpy(),
            "first_purchase": first.to_numpy(),
            "last_purchase": last.to_numpy(),
        }
    )
    # A customer observed for zero time carries no information; keep the row
    # but give it a floor so logs stay finite.
    summary["T"] = summary["T"].clip(lower=1e-3)
    summary["recency"] = summary["recency"].clip(lower=0.0)
    return summary.reset_index(drop=True)


# --------------------------------------------------------------------------
# BG/NBD
# --------------------------------------------------------------------------
@dataclass
class BGNBDParams:
    r: float
    alpha: float
    a: float
    b: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.r, self.alpha, self.a, self.b


class BetaGeoFitter:
    """Maximum-likelihood BG/NBD, optimised in log-space."""

    def __init__(self, penalizer_coef: float = 0.0):
        self.penalizer_coef = penalizer_coef
        self.params: BGNBDParams | None = None
        self.log_likelihood_: float | None = None

    # ---- likelihood ----------------------------------------------------
    @staticmethod
    def _log_likelihood(params: np.ndarray, x: np.ndarray, t_x: np.ndarray,
                        T: np.ndarray) -> np.ndarray:
        r, alpha, a, b = params
        ln_a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
        # Beta(a, b+x)/Beta(a, b) in logs, via the log-beta function.
        ln_a2 = betaln(a, b + x) - betaln(a, b)
        ln_a3 = -(r + x) * np.log(alpha + T)
        with np.errstate(divide="ignore", invalid="ignore"):
            ln_a4 = np.where(
                x > 0,
                np.log(a) - np.log(np.maximum(b + x - 1, 1e-12)) - (r + x) * np.log(alpha + t_x),
                -np.inf,
            )
        # A3 + A4 in logs; logsumexp handles the x == 0 case where A4 vanishes.
        tail = logsumexp(np.vstack([ln_a3, ln_a4]), axis=0)
        return ln_a1 + ln_a2 + tail

    def _negative_ll(self, log_params: np.ndarray, x, t_x, T) -> float:
        params = np.exp(log_params)
        if not np.all(np.isfinite(params)) or np.any(params <= 0):
            return np.inf
        # Keep 'a' away from exactly 1, where the closed-form expectation has a
        # removable singularity.
        if abs(params[2] - 1.0) < 1e-6:
            params[2] += 1e-6
        ll = self._log_likelihood(params, x, t_x, T)
        if not np.all(np.isfinite(ll)):
            return np.inf
        penalty = self.penalizer_coef * np.sum(np.asarray(log_params) ** 2)
        return -ll.sum() + penalty

    # ---- fitting -------------------------------------------------------
    def fit(self, frequency, recency, T, initial_params: tuple | None = None,
            n_restarts: int = 5) -> "BetaGeoFitter":
        x = np.asarray(frequency, dtype=float)
        t_x = np.asarray(recency, dtype=float)
        T_ = np.asarray(T, dtype=float)

        starts = [np.array(initial_params, dtype=float)] if initial_params else []
        rng = np.random.default_rng(0)
        starts.append(np.array([1.0, 1.0, 1.0, 1.0]))
        starts.append(np.array([0.5, 5.0, 0.8, 2.5]))
        while len(starts) < n_restarts:
            starts.append(np.exp(rng.normal(0.0, 0.8, size=4)))

        best, best_val = None, np.inf
        for start in starts:
            res = minimize(
                self._negative_ll,
                np.log(np.clip(start, 1e-6, None)),
                args=(x, t_x, T_),
                method="Nelder-Mead",
                options={"maxiter": 8000, "xatol": 1e-8, "fatol": 1e-8},
            )
            if res.fun < best_val and np.all(np.isfinite(res.x)):
                best, best_val = res, res.fun

        if best is None:
            raise RuntimeError("BG/NBD optimisation failed to converge from any start")

        r, alpha, a, b = np.exp(best.x)
        if abs(a - 1.0) < 1e-6:
            a += 1e-6
        self.params = BGNBDParams(float(r), float(alpha), float(a), float(b))
        self.log_likelihood_ = float(-best_val)
        return self

    # ---- predictions ---------------------------------------------------
    def _p(self) -> tuple[float, float, float, float]:
        if self.params is None:
            raise RuntimeError("Call fit() before predicting")
        return self.params.as_tuple()

    def conditional_expected_transactions(self, t: float, frequency, recency, T):
        """E[Y(t) | x, t_x, T] - purchases expected in the next ``t`` periods."""
        r, alpha, a, b = self._p()
        x = np.asarray(frequency, dtype=float)
        t_x = np.asarray(recency, dtype=float)
        T_ = np.asarray(T, dtype=float)

        hyp = hyp2f1(r + x, b + x, a + b + x - 1.0, t / (alpha + T_ + t))
        first = (a + b + x - 1.0) / (a - 1.0)
        second = 1.0 - hyp * ((alpha + T_) / (alpha + T_ + t)) ** (r + x)
        numerator = first * second
        denominator = 1.0 + np.where(
            x > 0,
            (a / np.maximum(b + x - 1.0, 1e-12)) * ((alpha + T_) / (alpha + t_x)) ** (r + x),
            0.0,
        )
        return np.clip(numerator / denominator, 0.0, None)

    def conditional_probability_alive(self, frequency, recency, T):
        """P(still a customer | x, t_x, T)."""
        r, alpha, a, b = self._p()
        x = np.asarray(frequency, dtype=float)
        t_x = np.asarray(recency, dtype=float)
        T_ = np.asarray(T, dtype=float)
        odds_dead = np.where(
            x > 0,
            (a / np.maximum(b + x - 1.0, 1e-12)) * ((alpha + T_) / (alpha + t_x)) ** (r + x),
            0.0,
        )
        return 1.0 / (1.0 + odds_dead)

    def expected_transactions_population(self, t):
        """E[X(t)] for a *randomly chosen* newly acquired customer."""
        r, alpha, a, b = self._p()
        t = np.asarray(t, dtype=float)
        hyp = hyp2f1(r, b, a + b - 1.0, t / (alpha + t))
        return ((a + b - 1.0) / (a - 1.0)) * (1.0 - hyp * (alpha / (alpha + t)) ** r)

    def summary(self) -> dict:
        r, alpha, a, b = self._p()
        return {
            "r": round(r, 4), "alpha": round(alpha, 4), "a": round(a, 4), "b": round(b, 4),
            "log_likelihood": round(self.log_likelihood_ or float("nan"), 2),
            "mean_purchase_rate_per_period": round(r / alpha, 5),
            "mean_dropout_probability": round(a / (a + b), 4),
        }


# --------------------------------------------------------------------------
# Gamma-Gamma spend model
# --------------------------------------------------------------------------
@dataclass
class GammaGammaParams:
    p: float
    q: float
    v: float


class GammaGammaFitter:
    """Models average transaction value, shrunk towards the population mean."""

    def __init__(self, penalizer_coef: float = 0.0):
        self.penalizer_coef = penalizer_coef
        self.params: GammaGammaParams | None = None
        self.log_likelihood_: float | None = None

    @staticmethod
    def _log_likelihood(params: np.ndarray, x: np.ndarray, m: np.ndarray) -> np.ndarray:
        p, q, v = params
        return (
            gammaln(p * x + q)
            - gammaln(p * x)
            - gammaln(q)
            + q * np.log(v)
            + (p * x - 1) * np.log(m)
            + (p * x) * np.log(x)
            - (p * x + q) * np.log(v + m * x)
        )

    def _negative_ll(self, log_params: np.ndarray, x, m) -> float:
        params = np.exp(log_params)
        if not np.all(np.isfinite(params)) or np.any(params <= 0):
            return np.inf
        ll = self._log_likelihood(params, x, m)
        if not np.all(np.isfinite(ll)):
            return np.inf
        return -ll.sum() + self.penalizer_coef * np.sum(np.asarray(log_params) ** 2)

    def fit(self, frequency, monetary_value, n_restarts: int = 4) -> "GammaGammaFitter":
        x = np.asarray(frequency, dtype=float)
        m = np.asarray(monetary_value, dtype=float)
        mask = (x > 0) & (m > 0)
        if mask.sum() < 10:
            raise ValueError("Gamma-Gamma needs repeat customers with positive spend")
        x, m = x[mask], m[mask]

        mean_m = float(m.mean())
        starts = [np.array([6.0, 4.0, mean_m * 3.0 / 5.0]),
                  np.array([1.0, 1.0, mean_m]),
                  np.array([3.0, 2.0, mean_m * 0.5])]
        rng = np.random.default_rng(1)
        while len(starts) < n_restarts:
            starts.append(np.array([abs(rng.normal(4, 1)), abs(rng.normal(3, 1)), mean_m]))

        best, best_val = None, np.inf
        for start in starts:
            res = minimize(self._negative_ll, np.log(np.clip(start, 1e-6, None)),
                           args=(x, m), method="Nelder-Mead",
                           options={"maxiter": 8000, "xatol": 1e-8, "fatol": 1e-8})
            if res.fun < best_val and np.all(np.isfinite(res.x)):
                best, best_val = res, res.fun
        if best is None:
            raise RuntimeError("Gamma-Gamma optimisation failed to converge")

        p, q, v = np.exp(best.x)
        self.params = GammaGammaParams(float(p), float(q), float(v))
        self.log_likelihood_ = float(-best_val)
        return self

    def conditional_expected_average_profit(self, frequency, monetary_value):
        """Credibility-weighted blend of the customer's own mean and the market's.

        A customer with one observed basket is mostly predicted by the
        population; a customer with fifty baskets is predicted by themselves.
        """
        if self.params is None:
            raise RuntimeError("Call fit() before predicting")
        p, q, v = self.params.p, self.params.q, self.params.v
        x = np.asarray(frequency, dtype=float)
        m = np.asarray(monetary_value, dtype=float)
        population_mean = v * p / (q - 1.0)
        individual_weight = (p * x) / (p * x + q - 1.0)
        return (1.0 - individual_weight) * population_mean + individual_weight * m

    def summary(self) -> dict:
        if self.params is None:
            raise RuntimeError("Call fit() before summarising")
        p, q, v = self.params.p, self.params.q, self.params.v
        return {
            "p": round(p, 4), "q": round(q, 4), "v": round(v, 2),
            "log_likelihood": round(self.log_likelihood_ or float("nan"), 2),
            "population_mean_transaction_value": round(v * p / (q - 1.0), 2),
        }


# --------------------------------------------------------------------------
# Putting the two together
# --------------------------------------------------------------------------
def customer_lifetime_value(bgf: BetaGeoFitter, ggf: GammaGammaFitter,
                            summary: pd.DataFrame, months: int = 12,
                            discount_rate_monthly: float = 0.01,
                            margin_rate: float = 1.0) -> pd.DataFrame:
    """Discounted expected value per customer over the next ``months``.

    Month by month, expected incremental transactions are multiplied by the
    expected spend and discounted back to today, which is the standard
    DCF treatment of a customer as a small annuity.
    """
    x = summary["frequency"].to_numpy(dtype=float)
    t_x = summary["recency"].to_numpy(dtype=float)
    T = summary["T"].to_numpy(dtype=float)

    expected_value = ggf.conditional_expected_average_profit(x, summary["monetary_value"].to_numpy())
    clv = np.zeros(len(summary))
    prev_cum = np.zeros(len(summary))
    monthly_txns = {}

    for month in range(1, months + 1):
        t = month * WEEKS_PER_MONTH
        cum = bgf.conditional_expected_transactions(t, x, t_x, T)
        incremental = np.clip(cum - prev_cum, 0.0, None)
        monthly_txns[month] = incremental
        clv += (incremental * expected_value * margin_rate) / ((1 + discount_rate_monthly) ** month)
        prev_cum = cum

    out = summary[["customer_id", "frequency", "recency", "T", "monetary_value", "total_value"]].copy()
    out["prob_alive"] = np.round(bgf.conditional_probability_alive(x, t_x, T), 4)
    out["expected_transactions_next_90d"] = np.round(
        bgf.conditional_expected_transactions(90 / DAYS_PER_WEEK, x, t_x, T), 3)
    out[f"expected_transactions_{months}m"] = np.round(prev_cum, 3)
    out["expected_avg_transaction_value"] = np.round(expected_value, 2)
    out[f"clv_{months}m"] = np.round(clv, 2)
    out["clv_decile"] = pd.qcut(out[f"clv_{months}m"].rank(method="first"), 10,
                                labels=[f"D{i}" for i in range(10, 0, -1)]).astype(str)
    return out.sort_values(f"clv_{months}m", ascending=False).reset_index(drop=True)


def validate_holdout(bgf: BetaGeoFitter, fact_sales: pd.DataFrame,
                     calibration_end: pd.Timestamp, observation_end: pd.Timestamp,
                     freq_days: float = DAYS_PER_WEEK) -> dict:
    """Honest test: fit on the past, score the future nobody has seen.

    Returns aggregate and per-customer accuracy of the predicted number of
    transactions in the holdout window.
    """
    cal = summary_from_transactions(fact_sales, observation_end=calibration_end,
                                    calibration_end=calibration_end, freq_days=freq_days)
    holdout_weeks = (pd.Timestamp(observation_end) - pd.Timestamp(calibration_end)).days / freq_days

    df = fact_sales[fact_sales["customer_id"].notna()].copy()
    df["date"] = pd.to_datetime(df["date"])
    hold = df[(df["date"] > pd.Timestamp(calibration_end)) & (df["date"] <= pd.Timestamp(observation_end))]
    actual = (hold.groupby(["customer_id", "date"]).size().reset_index()
              .groupby("customer_id").size().rename("actual_holdout_txns"))

    cal = cal.merge(actual, on="customer_id", how="left")
    cal["actual_holdout_txns"] = cal["actual_holdout_txns"].fillna(0.0)
    cal["predicted_holdout_txns"] = bgf.conditional_expected_transactions(
        holdout_weeks, cal["frequency"], cal["recency"], cal["T"])

    err = cal["predicted_holdout_txns"] - cal["actual_holdout_txns"]
    total_pred = float(cal["predicted_holdout_txns"].sum())
    total_actual = float(cal["actual_holdout_txns"].sum())
    return {
        "customers_scored": int(len(cal)),
        "holdout_weeks": round(holdout_weeks, 2),
        "predicted_total_transactions": round(total_pred, 1),
        "actual_total_transactions": round(total_actual, 1),
        "aggregate_error_pct": round(100 * (total_pred - total_actual) / max(total_actual, 1), 2),
        "mae_per_customer": round(float(err.abs().mean()), 4),
        "rmse_per_customer": round(float(np.sqrt((err ** 2).mean())), 4),
        "correlation": round(float(cal["predicted_holdout_txns"].corr(cal["actual_holdout_txns"])), 4),
        "detail": cal,
    }
