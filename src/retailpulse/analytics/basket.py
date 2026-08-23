"""Market-basket analysis: Apriori frequent itemsets and association rules,
implemented from scratch.

Nothing here comes from mlxtend or a rules library - the whole point is that
the algorithm is visible:

1. **Encode.** Every basket becomes a row in a boolean matrix (transactions x
   products). Each product also gets a *tidset*: the boolean column marking
   which baskets contain it.
2. **Level-wise search (Apriori).** Start with the single items that clear
   `min_support`. To build candidates of size k, join two frequent (k-1)-sets
   that share their first k-2 items, then apply the Apriori pruning rule: a
   candidate can only be frequent if *every* one of its (k-1)-subsets is
   frequent. That prune is what stops the search exploding combinatorially.
3. **Count.** Support of a candidate is the AND of its parents' tidsets, so
   each level reuses the level below instead of rescanning the data.
4. **Rules.** Split each frequent itemset into antecedent -> consequent every
   possible way, and score with support, confidence, lift, leverage,
   conviction and Zhang's metric.

Read in business terms: *lift* is "how many times more likely these products
are bought together than if shoppers picked them independently". Lift 3.0 on
coffee -> milk means put them on the same aisle end.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


class BasketEncoder:
    """Transactions -> boolean matrix + per-item tidsets."""

    def __init__(self, fact_sales: pd.DataFrame, item_col: str = "product_id",
                 basket_col: str = "transaction_id"):
        df = fact_sales[[basket_col, item_col]].drop_duplicates()
        self.items: list[str] = sorted(df[item_col].unique().tolist())
        self.item_index = {item: i for i, item in enumerate(self.items)}

        baskets = df[basket_col].astype("category")
        self.n_transactions = len(baskets.cat.categories)

        matrix = np.zeros((self.n_transactions, len(self.items)), dtype=bool)
        rows = baskets.cat.codes.to_numpy()
        cols = df[item_col].map(self.item_index).to_numpy()
        matrix[rows, cols] = True
        self.matrix = matrix

    def tidset(self, item_idx: int) -> np.ndarray:
        return self.matrix[:, item_idx]

    def item_supports(self) -> np.ndarray:
        return self.matrix.mean(axis=0)


def apriori(encoder: BasketEncoder, min_support: float = 0.005,
            max_len: int = 3) -> pd.DataFrame:
    """Find every itemset whose support clears ``min_support``.

    Returns a frame with columns ``itemset`` (tuple of item ids), ``support``,
    ``count`` and ``length``.
    """
    n = encoder.n_transactions
    min_count = int(np.ceil(min_support * n))

    # ---- Level 1 -----------------------------------------------------------
    counts = encoder.matrix.sum(axis=0)
    frequent: dict[tuple[int, ...], np.ndarray] = {}
    for i, c in enumerate(counts):
        if c >= min_count:
            frequent[(i,)] = encoder.matrix[:, i]

    all_levels: list[dict[tuple[int, ...], np.ndarray]] = [frequent]
    results: list[dict] = [
        {"itemset": k, "count": int(v.sum()), "support": float(v.sum() / n)}
        for k, v in frequent.items()
    ]

    # ---- Levels 2..max_len -------------------------------------------------
    k = 2
    while k <= max_len and all_levels[-1]:
        prev = all_levels[-1]
        prev_keys = sorted(prev.keys())
        prev_set = set(prev_keys)
        candidates: dict[tuple[int, ...], np.ndarray] = {}

        # F(k-1) x F(k-1) join: merge pairs sharing their first k-2 items.
        for a_i in range(len(prev_keys)):
            a = prev_keys[a_i]
            for b_i in range(a_i + 1, len(prev_keys)):
                b = prev_keys[b_i]
                if a[:-1] != b[:-1]:
                    break  # sorted order: no further b can share the prefix
                candidate = a + (b[-1],)

                # Apriori pruning: every (k-1)-subset must itself be frequent.
                if any(sub not in prev_set for sub in combinations(candidate, k - 1)):
                    continue

                tids = prev[a] & encoder.matrix[:, b[-1]]
                count = int(tids.sum())
                if count >= min_count:
                    candidates[candidate] = tids
                    results.append({"itemset": candidate, "count": count,
                                    "support": float(count / n)})
        all_levels.append(candidates)
        k += 1

    out = pd.DataFrame(results)
    if out.empty:
        return pd.DataFrame(columns=["itemset", "count", "support", "length"])
    out["length"] = out["itemset"].map(len)
    return out.sort_values(["length", "support"], ascending=[True, False]).reset_index(drop=True)


def association_rules(frequent: pd.DataFrame, encoder: BasketEncoder,
                      min_confidence: float = 0.1, min_lift: float = 1.0,
                      product_names: pd.DataFrame | None = None) -> pd.DataFrame:
    """Turn frequent itemsets into scored if-then rules."""
    if frequent.empty:
        return pd.DataFrame()

    support_of: dict[tuple[int, ...], float] = dict(
        zip(frequent["itemset"], frequent["support"])
    )

    rows = []
    for itemset, support in zip(frequent["itemset"], frequent["support"]):
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for antecedent in combinations(itemset, r):
                consequent = tuple(sorted(set(itemset) - set(antecedent)))
                sup_a = support_of.get(tuple(sorted(antecedent)))
                sup_c = support_of.get(consequent)
                if not sup_a or not sup_c:
                    continue
                confidence = support / sup_a
                lift = confidence / sup_c
                if confidence < min_confidence or lift < min_lift:
                    continue
                leverage = support - sup_a * sup_c
                conviction = np.inf if confidence >= 1.0 else (1 - sup_c) / (1 - confidence)
                denom = max(confidence * (1 - sup_c), sup_c * (1 - confidence))
                zhang = (confidence - sup_c) / denom if denom > 0 else 0.0
                rows.append(
                    {
                        "antecedent": tuple(sorted(antecedent)),
                        "consequent": consequent,
                        "support": support,
                        "antecedent_support": sup_a,
                        "consequent_support": sup_c,
                        "confidence": confidence,
                        "lift": lift,
                        "leverage": leverage,
                        "conviction": conviction,
                        "zhang": zhang,
                        "basket_count": int(round(support * encoder.n_transactions)),
                    }
                )

    rules = pd.DataFrame(rows)
    if rules.empty:
        return rules

    idx_to_item = {i: item for item, i in encoder.item_index.items()}
    rules["antecedent_ids"] = rules["antecedent"].map(lambda t: tuple(idx_to_item[i] for i in t))
    rules["consequent_ids"] = rules["consequent"].map(lambda t: tuple(idx_to_item[i] for i in t))

    if product_names is not None:
        name_map = dict(zip(product_names["product_id"], product_names["product_name"]))
        rules["antecedent_names"] = rules["antecedent_ids"].map(
            lambda t: " + ".join(name_map.get(p, p) for p in t))
        rules["consequent_names"] = rules["consequent_ids"].map(
            lambda t: " + ".join(name_map.get(p, p) for p in t))

    numeric = ["support", "antecedent_support", "consequent_support",
               "confidence", "lift", "leverage", "zhang"]
    rules[numeric] = rules[numeric].round(6)
    return rules.sort_values("lift", ascending=False).reset_index(drop=True)


def mine_rules(fact_sales: pd.DataFrame, products: pd.DataFrame,
               min_support: float = 0.005, min_confidence: float = 0.1,
               min_lift: float = 1.0, max_len: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: encode -> apriori -> rules."""
    encoder = BasketEncoder(fact_sales)
    frequent = apriori(encoder, min_support=min_support, max_len=max_len)
    rules = association_rules(frequent, encoder, min_confidence=min_confidence,
                              min_lift=min_lift, product_names=products)
    idx_to_item = {i: item for item, i in encoder.item_index.items()}
    name_map = dict(zip(products["product_id"], products["product_name"]))
    frequent = frequent.copy()
    frequent["items"] = frequent["itemset"].map(
        lambda t: " + ".join(name_map.get(idx_to_item[i], idx_to_item[i]) for i in t))
    return frequent, rules


def cross_sell_recommendations(rules: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """Best single follow-on product for each anchor product.

    Only 1 -> 1 rules are used, because that is what a "customers also bought"
    slot on a product page can actually display.
    """
    if rules.empty:
        return pd.DataFrame()
    simple = rules[(rules["antecedent"].map(len) == 1) & (rules["consequent"].map(len) == 1)].copy()
    if simple.empty:
        return pd.DataFrame()
    simple["anchor"] = simple["antecedent_names"] if "antecedent_names" in simple else simple["antecedent_ids"]
    simple["recommend"] = simple["consequent_names"] if "consequent_names" in simple else simple["consequent_ids"]
    ranked = (simple.sort_values(["anchor", "lift"], ascending=[True, False])
              .groupby("anchor")
              .head(top_n))
    return ranked[["anchor", "recommend", "support", "confidence", "lift",
                   "basket_count"]].reset_index(drop=True)
