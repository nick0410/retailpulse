"""The generator must be deterministic, and the quality engine must find
exactly the defects the generator buried.

This pairing is the backbone of the whole project: because the data is
simulated, "did the validation work?" has a real answer rather than a vibe.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from retailpulse.etl.quality import quality_score, run_quality_suite
from retailpulse.generate import generate_dataset


# --------------------------------------------------------------------------
# Generator
# --------------------------------------------------------------------------
def test_same_seed_gives_identical_data(sim_config):
    a = generate_dataset(sim_config, write=False)
    b = generate_dataset(sim_config, write=False)
    pd.testing.assert_frame_equal(a["transactions"], b["transactions"])
    pd.testing.assert_frame_equal(a["transaction_items"], b["transaction_items"])


def test_different_seed_gives_different_data(sim_config):
    other = generate_dataset(replace(sim_config, seed=sim_config.seed + 1), write=False)
    base = generate_dataset(sim_config, write=False)
    assert len(other["transactions"]) != len(base["transactions"])


def test_tables_are_internally_consistent(raw, sim_config):
    txns, items = raw["transactions"], raw["transaction_items"]
    products, stores = raw["products"], raw["stores"]

    assert len(txns) > 0 and len(items) > 0
    assert set(txns["store_id"]) <= set(stores["store_id"])
    assert set(items["product_id"]) <= set(products["product_id"])

    dates = pd.to_datetime(txns["date"])
    assert dates.min() >= pd.Timestamp(sim_config.start_date)
    assert dates.max() <= pd.Timestamp(sim_config.end_date)

    # Walk-in traffic is anonymous by construction; loyalty traffic is not.
    assert txns["customer_id"].isna().any(), "expected some anonymous walk-ins"
    assert txns["customer_id"].notna().any(), "expected some identified customers"


def test_line_amount_equals_quantity_times_price(raw):
    """Every row except the ones deliberately corrupted must reconcile."""
    items = raw["transaction_items"]
    expected = items["quantity"] * items["unit_price"]
    mismatch = (items["line_amount"] - expected).abs() > 0.02
    # Only the injected zero-price rows may break this identity.
    assert mismatch.sum() <= raw["_ground_truth"]["data_quality_defects"]["zero_price_rows"]


def test_products_never_priced_below_cost(raw):
    p = raw["products"]
    assert (p["unit_cost"] < p["base_price"]).all()


# --------------------------------------------------------------------------
# Quality engine
# --------------------------------------------------------------------------
def test_quality_suite_finds_every_injected_defect(raw, sim_config, ground_truth):
    tables = {k: v for k, v in raw.items() if not k.startswith("_")}
    report, quarantine, _results = run_quality_suite(tables, sim_config.start_date, sim_config.end_date)
    defects = ground_truth["data_quality_defects"]

    def rows_failed(table: str, check: str) -> int:
        row = report[(report["table"] == table) & (report["check"] == check)]
        assert len(row) == 1, f"missing check {table}.{check}"
        return int(row["rows_failed"].iat[0])

    # Each of these numbers was chosen by the generator, not by the checker.
    assert rows_failed("customers", "city_populated") == defects["customers_missing_city"]
    assert rows_failed("customers", "email_well_formed") == defects["customers_invalid_email"]
    assert rows_failed("transactions", "transaction_id_unique") == defects["duplicate_transaction_rows"]
    assert rows_failed("transaction_items", "quantity_positive") == defects["negative_quantity_rows"]
    assert rows_failed("transaction_items", "unit_price_positive") == defects["zero_price_rows"]
    assert rows_failed("transaction_items", "transaction_fk_resolves") == defects["orphan_line_items"]


def test_clean_data_passes_every_check(raw, sim_config):
    """With the defects repaired, nothing should fire.

    Note the customer rows are *repaired*, not dropped: deleting them would
    orphan their transactions and the referential check would rightly fire,
    which is the failure mode this test would otherwise hide.
    """
    tables = {k: v.copy() for k, v in raw.items() if not k.startswith("_")}
    tables["customers"]["city"] = tables["customers"]["city"].fillna("Mumbai")
    bad_email = ~tables["customers"]["email"].str.contains("@", na=False)
    tables["customers"].loc[bad_email, "email"] = "repaired@example.com"
    tables["transactions"] = tables["transactions"].drop_duplicates("transaction_id")
    items = tables["transaction_items"]
    items = items[(items["quantity"] > 0) & (items["unit_price"] > 0)]
    items = items[items["transaction_id"].isin(set(tables["transactions"]["transaction_id"]))]
    tables["transaction_items"] = items

    report, _quarantine, _results = run_quality_suite(tables, sim_config.start_date, sim_config.end_date)
    failed = report[~report["passed"]]
    assert failed.empty, f"unexpected failures:\n{failed[['table', 'check', 'rows_failed']]}"


def test_quality_score_is_bounded_and_sensitive(cleaned):
    _silver, report, _quarantine = cleaned
    score = quality_score(report)
    assert 0.0 <= score <= 100.0
    # The dataset is dirty on purpose, so a perfect score would be a bug.
    assert score < 100.0


def test_quarantine_keeps_the_first_copy_of_a_duplicate(cleaned, raw, ground_truth):
    silver, _report, quarantine = cleaned
    n_dupe = ground_truth["data_quality_defects"]["duplicate_transaction_rows"]
    raw_ids = raw["transactions"]["transaction_id"]
    # Every transaction id still present exactly once, none lost entirely.
    assert silver["transactions"]["transaction_id"].is_unique
    assert set(silver["transactions"]["transaction_id"]) == set(raw_ids)
    assert (quarantine["reason"] == "transaction_id_unique").sum() == n_dupe


def test_every_quarantined_row_carries_a_reason(cleaned):
    _silver, _report, quarantine = cleaned
    assert not quarantine.empty
    assert quarantine["reason"].notna().all()
    assert (quarantine["reason"].str.len() > 0).all()
    assert "unspecified" not in set(quarantine["reason"])
