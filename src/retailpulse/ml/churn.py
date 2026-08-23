"""Churn prediction with an out-of-time evaluation and a business case.

**Framing.** In a shop there is no "cancel" button, so churn has to be
defined: a customer is churned if they do not come back within 90 days of the
snapshot. The model sees only history up to the snapshot and predicts that
forward window.

**Why two snapshots.** Random train/test splits flatter a churn model badly -
the two halves share the same calendar, so the model can lean on "what was
happening in autumn 2024" instead of on customer behaviour. Here the training
set is built at one snapshot and the test set at a strictly later one, so the
score reported is the score you would have got by deploying the model and
waiting.

**What is measured.** ROC-AUC ranks, but a retention team does not act on a
ranking - it acts on a budget. So the report also carries PR-AUC (the metric
that respects class imbalance), the Brier score and a calibration curve (are
the probabilities *honest*?), lift by decile (how much better than random is
the top slice?), and a targeting simulation that converts all of it into
rupees.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import feature_columns


def _build_preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), numeric),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical),
        ],
        remainder="drop",
    )


def build_model(numeric: list[str], categorical: list[str], kind: str = "gbm",
                random_state: int = 42) -> Pipeline:
    """Gradient boosting for the real model, logistic regression as the yardstick."""
    if kind == "gbm":
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_depth=None, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=25,
            random_state=random_state,
        )
        pre = _build_preprocessor(numeric, categorical, scale=False)
    elif kind == "logistic":
        clf = LogisticRegression(max_iter=2000, C=0.5, random_state=random_state)
        pre = _build_preprocessor(numeric, categorical, scale=True)
    else:
        raise ValueError(f"unknown model kind {kind!r}")
    return Pipeline([("pre", pre), ("clf", clf)])


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Maximum separation between the churner and non-churner score curves."""
    order = np.argsort(-y_score)
    y = np.asarray(y_true)[order]
    pos = np.cumsum(y) / max(y.sum(), 1)
    neg = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(pos - neg)))


def decile_lift(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Churn rate by score decile, versus the base rate."""
    df = pd.DataFrame({"y": np.asarray(y_true), "score": np.asarray(y_score)})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["decile"] = np.minimum((np.arange(len(df)) * n_bins // len(df)) + 1, n_bins)
    base = df["y"].mean()
    out = df.groupby("decile").agg(customers=("y", "size"), churn_rate=("y", "mean"),
                                   avg_score=("score", "mean")).reset_index()
    out["lift"] = np.round(out["churn_rate"] / base, 3)
    out["cumulative_churners_captured"] = np.round(
        df.groupby("decile")["y"].sum().cumsum().to_numpy() / max(df["y"].sum(), 1), 4)
    out["churn_rate"] = np.round(out["churn_rate"], 4)
    out["avg_score"] = np.round(out["avg_score"], 4)
    return out


def calibration_table(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed churn rate per probability bucket."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_score)})
    df["bucket"] = pd.cut(df["p"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    out = df.groupby("bucket", observed=True).agg(
        customers=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")).reset_index()
    out["bucket"] = out["bucket"].astype(str)
    out[["predicted", "observed"]] = out[["predicted", "observed"]].round(4)
    out["gap"] = (out["observed"] - out["predicted"]).round(4)
    return out


def evaluate(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    return {
        "n": int(len(y_true)),
        "base_churn_rate": round(float(y_true.mean()), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        "brier": round(float(brier_score_loss(y_true, y_score)), 4),
        "log_loss": round(float(log_loss(y_true, np.clip(y_score, 1e-6, 1 - 1e-6))), 4),
        "ks": round(ks_statistic(y_true, y_score), 4),
    }


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
@dataclass
class ChurnResult:
    model: Pipeline
    metrics: dict = field(default_factory=dict)
    baseline_metrics: dict = field(default_factory=dict)
    lift: pd.DataFrame = field(default_factory=pd.DataFrame)
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    importance: pd.DataFrame = field(default_factory=pd.DataFrame)
    scored: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_names: list[str] = field(default_factory=list)


def train_churn_model(train: pd.DataFrame, test: pd.DataFrame,
                      compute_importance: bool = True,
                      random_state: int = 42) -> ChurnResult:
    """Fit on the earlier snapshot, score the later one, and explain the result."""
    numeric, categorical = feature_columns(train)
    features = numeric + categorical

    X_train, y_train = train[features], train["churned"].to_numpy()
    X_test, y_test = test[features], test["churned"].to_numpy()

    model = build_model(numeric, categorical, kind="gbm", random_state=random_state).fit(X_train, y_train)
    baseline = build_model(numeric, categorical, kind="logistic", random_state=random_state).fit(X_train, y_train)

    p_test = model.predict_proba(X_test)[:, 1]
    p_base = baseline.predict_proba(X_test)[:, 1]

    importance = pd.DataFrame()
    if compute_importance:
        # Permutation importance on the *test* set: which columns does the
        # model actually rely on when it meets data it has never seen?
        perm = permutation_importance(model, X_test, y_test, n_repeats=5,
                                      random_state=random_state, scoring="roc_auc", n_jobs=1)
        importance = (pd.DataFrame({"feature": features,
                                    "importance": perm.importances_mean,
                                    "std": perm.importances_std})
                      .sort_values("importance", ascending=False)
                      .reset_index(drop=True))

    scored = test[["customer_id", "churned", "monetary_total", "recency_days",
                   "frequency", "avg_basket_value"]].copy()
    scored["churn_probability"] = np.round(p_test, 4)
    scored["risk_band"] = pd.cut(scored["churn_probability"],
                                 bins=[-0.01, 0.25, 0.5, 0.75, 1.01],
                                 labels=["Low", "Medium", "High", "Critical"]).astype(str)
    scored = scored.sort_values("churn_probability", ascending=False).reset_index(drop=True)

    return ChurnResult(
        model=model,
        metrics=evaluate(y_test, p_test),
        baseline_metrics=evaluate(y_test, p_base),
        lift=decile_lift(y_test, p_test),
        calibration=calibration_table(y_test, p_test),
        importance=importance,
        scored=scored,
        feature_names=features,
    )


# --------------------------------------------------------------------------
# From probabilities to rupees
# --------------------------------------------------------------------------
def targeting_simulation(scored: pd.DataFrame, contact_cost: float = 40.0,
                         save_rate: float = 0.25, margin_rate: float = 0.30,
                         value_col: str = "monetary_total",
                         horizon_scale: float = 0.25) -> pd.DataFrame:
    """What does the model earn if the retention team works the top K%?

    Assumptions are explicit and easy to argue with, which is the point:

    * contacting a customer costs ``contact_cost``;
    * a genuine churner who is contacted is saved ``save_rate`` of the time;
    * a saved customer is worth ``horizon_scale`` of their historic spend over
      the next window, at ``margin_rate`` margin;
    * contacting a non-churner costs money and buys nothing.
    """
    df = scored.sort_values("churn_probability", ascending=False).reset_index(drop=True)
    rows = []
    for pct in (0.05, 0.10, 0.20, 0.30, 0.50, 1.00):
        k = max(int(round(pct * len(df))), 1)
        target = df.head(k)
        true_churners = target[target["churned"] == 1]
        saved = save_rate * len(true_churners)
        value_saved = save_rate * (true_churners[value_col] * horizon_scale * margin_rate).sum()
        cost = contact_cost * k
        rows.append(
            {
                "target_pct": pct,
                "customers_contacted": k,
                "churners_in_target": int(len(true_churners)),
                "precision": round(len(true_churners) / k, 4),
                "recall": round(len(true_churners) / max(int(df["churned"].sum()), 1), 4),
                "expected_customers_saved": round(saved, 1),
                "expected_margin_saved": round(float(value_saved), 2),
                "campaign_cost": round(cost, 2),
                "net_benefit": round(float(value_saved) - cost, 2),
                "roi": round((float(value_saved) - cost) / cost, 3) if cost else np.nan,
            }
        )
    return pd.DataFrame(rows)
