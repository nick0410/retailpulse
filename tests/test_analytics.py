"""RFM, cohorts and market-basket mining.

The Apriori tests are the interesting ones: the algorithm is checked twice -
once against a five-basket example whose supports can be counted by hand, and
once against the real dataset, where it has to rediscover the product pairs
the simulator planted without being told they exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from retailpulse.analytics.basket import (BasketEncoder, apriori, association_rules,
                                          cross_sell_recommendations, mine_rules)
from retailpulse.analytics.cohort import build_cohort_table, retention_curve, retention_matrix
from retailpulse.analytics.rfm import build_rfm, pareto_concentration, segment_summary


# --------------------------------------------------------------------------
# A hand-countable basket example
# --------------------------------------------------------------------------
@pytest.fixture
def toy_baskets() -> pd.DataFrame:
    """Five baskets, chosen so every support can be verified by eye.

        T1: bread, butter, milk
        T2: bread, butter
        T3: bread, milk
        T4: butter, milk
        T5: bread, butter, milk
    """
    rows = [
        ("T1", "bread"), ("T1", "butter"), ("T1", "milk"),
        ("T2", "bread"), ("T2", "butter"),
        ("T3", "bread"), ("T3", "milk"),
        ("T4", "butter"), ("T4", "milk"),
        ("T5", "bread"), ("T5", "butter"), ("T5", "milk"),
    ]
    return pd.DataFrame(rows, columns=["transaction_id", "product_id"])


def test_encoder_builds_the_right_matrix(toy_baskets):
    enc = BasketEncoder(toy_baskets)
    assert enc.n_transactions == 5
    assert enc.items == ["bread", "butter", "milk"]
    # bread appears in T1, T2, T3, T5 -> 4 of 5
    assert enc.matrix[:, enc.item_index["bread"]].sum() == 4


def test_apriori_supports_match_hand_counts(toy_baskets):
    enc = BasketEncoder(toy_baskets)
    freq = apriori(enc, min_support=0.01, max_len=3)
    idx = enc.item_index
    support = {tuple(sorted(k)): v for k, v in zip(freq["itemset"], freq["support"])}

    assert support[(idx["bread"],)] == pytest.approx(4 / 5)
    assert support[(idx["butter"],)] == pytest.approx(4 / 5)
    assert support[(idx["milk"],)] == pytest.approx(4 / 5)
    # bread+butter in T1, T2, T5
    assert support[tuple(sorted((idx["bread"], idx["butter"])))] == pytest.approx(3 / 5)
    # all three together in T1, T5
    assert support[tuple(sorted((idx["bread"], idx["butter"], idx["milk"])))] == pytest.approx(2 / 5)


def test_apriori_prunes_below_threshold(toy_baskets):
    enc = BasketEncoder(toy_baskets)
    freq = apriori(enc, min_support=0.5, max_len=3)
    # The 3-itemset has support 0.4 and must be pruned away.
    assert freq["length"].max() == 2
    assert (freq["support"] >= 0.5).all()


def test_rule_metrics_follow_their_definitions(toy_baskets):
    enc = BasketEncoder(toy_baskets)
    freq = apriori(enc, min_support=0.01, max_len=3)
    rules = association_rules(freq, enc, min_confidence=0.0, min_lift=0.0)

    idx = enc.item_index
    row = rules[(rules["antecedent"] == (idx["bread"],))
                & (rules["consequent"] == (idx["butter"],))].iloc[0]
    # support(bread & butter) = 3/5, support(bread) = 4/5, support(butter) = 4/5
    assert row["confidence"] == pytest.approx((3 / 5) / (4 / 5))
    assert row["lift"] == pytest.approx(((3 / 5) / (4 / 5)) / (4 / 5))
    assert row["leverage"] == pytest.approx(3 / 5 - (4 / 5) * (4 / 5))


def test_independent_items_have_lift_of_one():
    """Two products bought independently must score lift ~= 1, not more."""
    rng = np.random.default_rng(0)
    n = 4000
    a = rng.random(n) < 0.4
    b = rng.random(n) < 0.3          # independent of a by construction
    rows = []
    for i in range(n):
        if a[i]:
            rows.append((f"T{i}", "A"))
        if b[i]:
            rows.append((f"T{i}", "B"))
        rows.append((f"T{i}", "filler"))
    enc = BasketEncoder(pd.DataFrame(rows, columns=["transaction_id", "product_id"]))
    freq = apriori(enc, min_support=0.01, max_len=2)
    rules = association_rules(freq, enc, min_confidence=0.0, min_lift=0.0)
    ab = rules[(rules["antecedent"] == (enc.item_index["A"],))
               & (rules["consequent"] == (enc.item_index["B"],))].iloc[0]
    assert ab["lift"] == pytest.approx(1.0, abs=0.1)


def test_apriori_rediscovers_the_planted_pairs(fact, star, ground_truth):
    """The real test: find the affinities without being told they exist."""
    _freq, rules = mine_rules(fact, star["dim_product"], min_support=0.0015,
                              min_confidence=0.05, min_lift=1.1, max_len=2)
    assert not rules.empty

    found = {(a, c) for a, c in zip(rules["antecedent_names"], rules["consequent_names"])}
    planted = [(r["antecedent_name"], r["consequent_name"]) for r in ground_truth["affinity_rules"]]
    hits = [p for p in planted if p in found or (p[1], p[0]) in found]
    # The small test dataset is thin, but the strong pairs must still surface.
    assert len(hits) >= len(planted) // 2, f"only recovered {hits}"

    # And the confidence must be close to the probability actually used.
    for rule in ground_truth["affinity_rules"]:
        match = rules[(rules["antecedent_names"] == rule["antecedent_name"])
                      & (rules["consequent_names"] == rule["consequent_name"])]
        if match.empty:
            continue
        assert match["confidence"].iat[0] == pytest.approx(rule["planted_probability"], abs=0.18)


def test_cross_sell_only_offers_single_products(fact, star):
    _freq, rules = mine_rules(fact, star["dim_product"], min_support=0.0015,
                              min_confidence=0.05, min_lift=1.1, max_len=3)
    cross = cross_sell_recommendations(rules, top_n=3)
    if cross.empty:
        pytest.skip("no 1->1 rules in the small dataset")
    assert (~cross["recommend"].str.contains(r" \+ ")).all()
    assert cross.groupby("anchor").size().max() <= 3


# --------------------------------------------------------------------------
# RFM
# --------------------------------------------------------------------------
def test_rfm_scores_are_in_range_and_directional(fact):
    rfm = build_rfm(fact)
    for col in ("R", "F", "M"):
        assert rfm[col].between(1, 5).all()
    # Recency is inverted: a *lower* day count must earn a *higher* R score.
    assert rfm[["recency_days", "R"]].corr().iloc[0, 1] < 0
    assert rfm[["frequency", "F"]].corr().iloc[0, 1] > 0
    assert rfm[["monetary", "M"]].corr().iloc[0, 1] > 0


def test_every_customer_lands_in_exactly_one_segment(fact):
    rfm = build_rfm(fact)
    assert rfm["segment"].notna().all()
    assert rfm["recommended_action"].notna().all()
    summary = segment_summary(rfm)
    assert summary["customers"].sum() == len(rfm)
    assert summary["revenue_share"].sum() == pytest.approx(1.0, abs=1e-3)


def test_rfm_excludes_anonymous_walk_ins(fact):
    rfm = build_rfm(fact)
    assert rfm["customer_id"].notna().all()
    assert len(rfm) <= fact["customer_id"].nunique()


def test_concentration_curve_is_monotone(fact):
    pareto = pareto_concentration(build_rfm(fact))
    shares = pareto["revenue_share"].to_numpy()
    assert (np.diff(shares) >= -1e-9).all()
    assert shares[-1] == pytest.approx(1.0, abs=1e-6)
    # Revenue is never spread perfectly evenly in retail.
    top20 = pareto.loc[pareto["top_customer_pct"] == 0.20, "revenue_share"].iat[0]
    assert top20 > 0.20


# --------------------------------------------------------------------------
# Cohorts
# --------------------------------------------------------------------------
def test_cohort_retention_is_a_valid_probability(fact, star):
    table = build_cohort_table(fact, star["dim_customer"])
    assert (table["retention_rate"] >= 0).all()
    assert (table["retention_rate"] <= 1.0 + 1e-9).all()
    assert (table["months_since_signup"] >= 0).all()
    assert (table["active_customers"] <= table["cohort_size"]).all()


def test_retention_decays_after_the_first_month(fact, star):
    curve = retention_curve(build_cohort_table(fact, star["dim_customer"]), min_cohort_size=20)
    assert len(curve) > 2
    m0 = curve.loc[curve["months_since_signup"] == 0, "retention_rate"].iat[0]
    m1 = curve.loc[curve["months_since_signup"] == 1, "retention_rate"].iat[0]
    # Month 0 contains the sign-up purchase itself, so it is near-total.
    assert m0 > 0.8
    assert m1 < m0


def test_retention_matrix_is_triangular(fact, star):
    table = build_cohort_table(fact, star["dim_customer"])
    matrix = retention_matrix(table, max_months=12)
    # The newest cohort cannot have been observed for as long as the oldest.
    oldest_observed = matrix.iloc[0].notna().sum()
    newest_observed = matrix.iloc[-1].notna().sum()
    assert oldest_observed >= newest_observed
