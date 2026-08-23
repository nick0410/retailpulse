"""RFM segmentation.

Recency / Frequency / Monetary is the oldest trick in retail analytics and it
is still the one a business person understands in ten seconds: *how recently*,
*how often*, and *how much*. Each customer gets a 1-5 score on each axis
(quintiles), and the 125 possible score triples are collapsed into nine named
segments that map to an action.

The scoring uses rank-based quintiles rather than raw ``qcut`` so that heavily
tied distributions (thousands of customers with exactly 1 purchase) still split
into non-degenerate buckets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# (recency_score_range, frequency_monetary_score_range) -> segment
SEGMENT_RULES: list[tuple[str, set[int], set[int]]] = [
    ("Champions",            {5, 4},       {5, 4}),
    ("Loyal Customers",      {5, 4, 3},    {5, 4, 3}),
    ("Potential Loyalist",   {5, 4},       {2, 1}),
    ("Recent Customers",     {5},          {1}),
    ("Promising",            {4, 3},       {1}),
    ("Needs Attention",      {3},          {3, 2}),
    ("At Risk",              {2},          {5, 4, 3}),
    ("Cannot Lose Them",     {1},          {5, 4}),
    ("Hibernating",          {2, 1},       {2, 1}),
]

SEGMENT_ACTION = {
    "Champions": "Reward and upsell - they set the ceiling for basket size",
    "Loyal Customers": "Cross-sell adjacent categories; ask for referrals",
    "Potential Loyalist": "Nudge to a second/third purchase with a bundle",
    "Recent Customers": "Onboarding journey; make the second visit easy",
    "Promising": "Low-cost incentive to build the habit",
    "Needs Attention": "Time-boxed offer before they slip further",
    "At Risk": "Win-back campaign with their favourite category",
    "Cannot Lose Them": "High-value lapsed - personal outreach",
    "Hibernating": "Cheapest reactivation channel only; low expected return",
}


def _quintile_score(series: pd.Series, ascending: bool = True, q: int = 5) -> pd.Series:
    """Rank-based quintile score in 1..q, robust to heavy ties.

    ``ascending=True`` means a higher raw value earns a higher score.
    """
    ranks = series.rank(method="average", pct=True, ascending=ascending)
    scores = np.ceil(ranks * q).clip(1, q)
    return scores.astype(int)


def build_rfm(fact_sales: pd.DataFrame, snapshot_date: pd.Timestamp | None = None,
              quantiles: int = 5) -> pd.DataFrame:
    """Compute the per-customer RFM table from the sales fact.

    Only identified (loyalty) customers can be segmented; anonymous walk-in
    lines are excluded, which is exactly what happens in a real programme.
    """
    df = fact_sales[fact_sales["customer_id"].notna()].copy()
    df["date"] = pd.to_datetime(df["date"])
    snapshot = pd.Timestamp(snapshot_date) if snapshot_date is not None else df["date"].max()

    txn = (df.groupby(["customer_id", "transaction_id"], as_index=False)
           .agg(date=("date", "first"), amount=("line_amount", "sum"),
                margin=("gross_margin", "sum"), lines=("line_id", "count")))

    rfm = txn.groupby("customer_id").agg(
        last_purchase=("date", "max"),
        first_purchase=("date", "min"),
        frequency=("transaction_id", "nunique"),
        monetary=("amount", "sum"),
        margin=("margin", "sum"),
        avg_basket_value=("amount", "mean"),
        avg_basket_lines=("lines", "mean"),
    ).reset_index()

    rfm["recency_days"] = (snapshot - rfm["last_purchase"]).dt.days
    rfm["tenure_days"] = (snapshot - rfm["first_purchase"]).dt.days
    rfm["purchase_frequency_per_year"] = np.where(
        rfm["tenure_days"] > 0, rfm["frequency"] * 365.0 / rfm["tenure_days"], np.nan
    )

    # Lower recency is better, hence ascending=False on the raw day count.
    rfm["R"] = _quintile_score(rfm["recency_days"], ascending=False, q=quantiles)
    rfm["F"] = _quintile_score(rfm["frequency"], ascending=True, q=quantiles)
    rfm["M"] = _quintile_score(rfm["monetary"], ascending=True, q=quantiles)
    rfm["rfm_cell"] = rfm["R"].astype(str) + rfm["F"].astype(str) + rfm["M"].astype(str)
    rfm["rfm_score"] = rfm[["R", "F", "M"]].sum(axis=1)

    rfm["segment"] = _assign_segments(rfm)
    rfm["recommended_action"] = rfm["segment"].map(SEGMENT_ACTION)
    rfm["snapshot_date"] = snapshot
    return rfm.sort_values("monetary", ascending=False).reset_index(drop=True)


def _assign_segments(rfm: pd.DataFrame) -> pd.Series:
    """Collapse the R x FM grid into named segments (first matching rule wins)."""
    fm = np.ceil((rfm["F"] + rfm["M"]) / 2).astype(int)
    segment = pd.Series("Hibernating", index=rfm.index, dtype=object)
    assigned = pd.Series(False, index=rfm.index)
    for name, r_set, fm_set in SEGMENT_RULES:
        match = (~assigned) & rfm["R"].isin(r_set) & fm.isin(fm_set)
        segment[match] = name
        assigned |= match
    return segment


def segment_summary(rfm: pd.DataFrame) -> pd.DataFrame:
    """Board-level view: size, value and behaviour of each segment."""
    total_rev = rfm["monetary"].sum()
    out = rfm.groupby("segment").agg(
        customers=("customer_id", "count"),
        revenue=("monetary", "sum"),
        margin=("margin", "sum"),
        avg_recency_days=("recency_days", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_basket_value=("avg_basket_value", "mean"),
    ).reset_index()
    out["revenue_share"] = np.round(out["revenue"] / total_rev, 4)
    out["customer_share"] = np.round(out["customers"] / len(rfm), 4)
    out["revenue_per_customer"] = np.round(out["revenue"] / out["customers"], 2)
    out["action"] = out["segment"].map(SEGMENT_ACTION)
    return out.sort_values("revenue", ascending=False).reset_index(drop=True)


def pareto_concentration(rfm: pd.DataFrame) -> pd.DataFrame:
    """How much revenue the top X% of customers actually produce."""
    ranked = rfm.sort_values("monetary", ascending=False).reset_index(drop=True)
    cum_share = ranked["monetary"].cumsum() / ranked["monetary"].sum()
    rows = []
    for pct in (0.01, 0.05, 0.10, 0.20, 0.50, 0.80, 1.00):
        idx = max(int(np.ceil(pct * len(ranked))) - 1, 0)
        rows.append({"top_customer_pct": pct,
                     "customers": idx + 1,
                     "revenue_share": round(float(cum_share.iat[idx]), 4)})
    return pd.DataFrame(rows)
