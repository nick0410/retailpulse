"""Point-in-time feature engineering for the churn model.

The single most common way to get a churn model wrong is leakage: building a
feature from data that only exists *after* the moment you claim to be
predicting from. Every function here takes a ``snapshot`` timestamp and is
forbidden from touching a single row dated after it. The label is then built
from the window that follows the snapshot, which the feature builder never
sees.

Feature families, and what each is trying to capture:

* **Recency / frequency / monetary** - the classic trio.
* **Momentum** - spend and visits in the last 90 days versus the 90 before
  that. A customer who is slowing down looks different from one who never
  bought much.
* **Rhythm** - mean and variability of the gap between visits, and how overdue
  the customer is relative to their own habit. Someone who shops every 7 days
  and has been away 30 is a much stronger signal than the same 30 days from a
  quarterly shopper.
* **Breadth** - how many distinct categories, products and stores they use.
  Breadth is stickiness.
* **Deal sensitivity** - share of spend bought on promotion. Discount-only
  shoppers churn differently.
* **Profile** - tier, age band, region, tenure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = ["loyalty_tier", "age_band", "preferred_channel", "customer_region"]


def build_customer_features(fact_sales: pd.DataFrame, dim_customer: pd.DataFrame,
                            snapshot: pd.Timestamp, lookback_days: int = 365) -> pd.DataFrame:
    """Everything we know about each customer as of ``snapshot`` - and nothing after."""
    snapshot = pd.Timestamp(snapshot)
    df = fact_sales[fact_sales["customer_id"].notna()].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= snapshot]
    if df.empty:
        return pd.DataFrame()

    # Trip grain: one shopping visit per customer per day.
    trips = (df.groupby(["customer_id", "date"], as_index=False)
             .agg(trip_value=("line_amount", "sum"),
                  trip_margin=("gross_margin", "sum"),
                  trip_lines=("line_id", "count"),
                  trip_discount=("discount_amount", "sum"),
                  store_id=("store_id", "first")))
    trips = trips.sort_values(["customer_id", "date"])

    g = trips.groupby("customer_id")
    feats = pd.DataFrame({"customer_id": g.size().index})
    feats["frequency"] = g.size().to_numpy()
    feats["first_purchase"] = g["date"].min().to_numpy()
    feats["last_purchase"] = g["date"].max().to_numpy()
    feats["monetary_total"] = g["trip_value"].sum().to_numpy()
    feats["margin_total"] = g["trip_margin"].sum().to_numpy()
    feats["avg_basket_value"] = g["trip_value"].mean().to_numpy()
    feats["std_basket_value"] = g["trip_value"].std().fillna(0).to_numpy()
    feats["max_basket_value"] = g["trip_value"].max().to_numpy()
    feats["avg_basket_lines"] = g["trip_lines"].mean().to_numpy()
    feats["discount_share"] = (g["trip_discount"].sum() /
                               g["trip_value"].sum().replace(0, np.nan)).fillna(0).to_numpy()

    feats["recency_days"] = (snapshot - pd.to_datetime(feats["last_purchase"])).dt.days
    feats["tenure_days"] = (snapshot - pd.to_datetime(feats["first_purchase"])).dt.days
    feats["purchases_per_year"] = np.where(
        feats["tenure_days"] > 0, feats["frequency"] * 365.0 / feats["tenure_days"], 0.0
    )

    # ---- rhythm: how regular is this shopper, and how overdue? -------------
    gaps = trips.groupby("customer_id")["date"].diff().dt.days
    gap_stats = gaps.groupby(trips["customer_id"]).agg(["mean", "std", "max"])
    gap_stats.columns = ["gap_mean_days", "gap_std_days", "gap_max_days"]
    feats = feats.merge(gap_stats, left_on="customer_id", right_index=True, how="left")
    feats[["gap_mean_days", "gap_std_days", "gap_max_days"]] = (
        feats[["gap_mean_days", "gap_std_days", "gap_max_days"]].fillna(0.0)
    )
    # >1 means the customer is already past their usual gap.
    feats["overdue_ratio"] = np.where(
        feats["gap_mean_days"] > 0, feats["recency_days"] / feats["gap_mean_days"], np.nan
    )
    feats["gap_cv"] = np.where(
        feats["gap_mean_days"] > 0, feats["gap_std_days"] / feats["gap_mean_days"], 0.0
    )

    # ---- momentum: last 90 days vs the 90 before ---------------------------
    recent = trips[trips["date"] > snapshot - pd.Timedelta(days=90)]
    prior = trips[(trips["date"] <= snapshot - pd.Timedelta(days=90))
                  & (trips["date"] > snapshot - pd.Timedelta(days=180))]
    recent_agg = recent.groupby("customer_id").agg(
        trips_last_90d=("date", "count"), spend_last_90d=("trip_value", "sum"))
    prior_agg = prior.groupby("customer_id").agg(
        trips_prev_90d=("date", "count"), spend_prev_90d=("trip_value", "sum"))
    feats = feats.merge(recent_agg, left_on="customer_id", right_index=True, how="left")
    feats = feats.merge(prior_agg, left_on="customer_id", right_index=True, how="left")
    for col in ("trips_last_90d", "spend_last_90d", "trips_prev_90d", "spend_prev_90d"):
        feats[col] = feats[col].fillna(0.0)
    feats["trip_momentum"] = (feats["trips_last_90d"] - feats["trips_prev_90d"]) / (
        feats["trips_last_90d"] + feats["trips_prev_90d"] + 1.0)
    feats["spend_momentum"] = (feats["spend_last_90d"] - feats["spend_prev_90d"]) / (
        feats["spend_last_90d"] + feats["spend_prev_90d"] + 1.0)

    # ---- breadth: variety is stickiness ------------------------------------
    window = df[df["date"] > snapshot - pd.Timedelta(days=lookback_days)]
    breadth = window.groupby("customer_id").agg(
        distinct_categories=("category", "nunique"),
        distinct_products=("product_id", "nunique"),
        distinct_stores=("store_id", "nunique"),
        distinct_channels=("channel", "nunique"),
    )
    feats = feats.merge(breadth, left_on="customer_id", right_index=True, how="left")
    for col in ("distinct_categories", "distinct_products", "distinct_stores", "distinct_channels"):
        feats[col] = feats[col].fillna(0.0)

    # Store loyalty: share of trips at the customer's most-used store.
    store_share = (trips.groupby(["customer_id", "store_id"]).size()
                   .groupby(level=0).apply(lambda s: s.max() / s.sum()))
    feats = feats.merge(store_share.rename("home_store_share"),
                        left_on="customer_id", right_index=True, how="left")
    feats["home_store_share"] = feats["home_store_share"].fillna(1.0)

    feats["weekend_trip_share"] = (
        trips.assign(is_weekend=trips["date"].dt.dayofweek >= 5)
        .groupby("customer_id")["is_weekend"].mean().to_numpy()
    )

    # ---- profile -----------------------------------------------------------
    profile_cols = ["customer_id", "loyalty_tier", "age_band", "preferred_channel",
                    "customer_region", "signup_date"]
    profile = dim_customer[[c for c in profile_cols if c in dim_customer.columns]].copy()
    feats = feats.merge(profile, on="customer_id", how="left")
    if "signup_date" in feats.columns:
        feats["days_since_signup"] = (snapshot - pd.to_datetime(feats["signup_date"])).dt.days
        feats = feats.drop(columns=["signup_date"])

    feats["overdue_ratio"] = feats["overdue_ratio"].fillna(feats["recency_days"] / 30.0)
    feats["snapshot"] = snapshot
    return feats.reset_index(drop=True)


def build_churn_label(fact_sales: pd.DataFrame, snapshot: pd.Timestamp,
                      horizon_days: int = 90) -> pd.DataFrame:
    """1 = the customer did not come back within ``horizon_days`` of the snapshot.

    This is the only place allowed to look past the snapshot.
    """
    snapshot = pd.Timestamp(snapshot)
    df = fact_sales[fact_sales["customer_id"].notna()].copy()
    df["date"] = pd.to_datetime(df["date"])
    future = df[(df["date"] > snapshot) & (df["date"] <= snapshot + pd.Timedelta(days=horizon_days))]
    returned = future.groupby("customer_id").agg(
        future_trips=("date", "nunique"), future_spend=("line_amount", "sum")).reset_index()
    returned["churned"] = 0
    return returned


def build_training_frame(fact_sales: pd.DataFrame, dim_customer: pd.DataFrame,
                         snapshot: pd.Timestamp, horizon_days: int = 90,
                         lookback_days: int = 365,
                         min_history_days: int = 30) -> pd.DataFrame:
    """Features as of the snapshot, joined to the forward-looking label.

    Customers who signed up in the last ``min_history_days`` are excluded:
    there is not enough history to say anything about them, and including them
    would let the model learn "new = unknown" instead of a churn signal.
    """
    feats = build_customer_features(fact_sales, dim_customer, snapshot, lookback_days)
    if feats.empty:
        return feats
    feats = feats[feats["tenure_days"] >= min_history_days].copy()

    label = build_churn_label(fact_sales, snapshot, horizon_days)
    out = feats.merge(label[["customer_id", "churned", "future_trips", "future_spend"]],
                      on="customer_id", how="left")
    out["churned"] = out["churned"].fillna(1).astype(int)
    out["future_trips"] = out["future_trips"].fillna(0)
    out["future_spend"] = out["future_spend"].fillna(0.0)
    return out.reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Split the frame into (numeric features, categorical features)."""
    drop = {"customer_id", "churned", "future_trips", "future_spend", "snapshot",
            "first_purchase", "last_purchase"}
    numeric = [c for c in df.columns
               if c not in drop and c not in CATEGORICAL_FEATURES
               and pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return numeric, categorical
