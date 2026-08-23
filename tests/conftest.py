"""Shared fixtures.

Tests run against a *small* simulated universe built once per session and held
in memory, so the suite stays fast and never depends on whatever happens to be
sitting in ``data/``. Tests that genuinely need the full-size warehouse skip
themselves when it has not been built.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from retailpulse.config import SimulationConfig, WAREHOUSE_DB  # noqa: E402
from retailpulse.etl import build_star_schema, clean_layer, load_warehouse  # noqa: E402
from retailpulse.generate import generate_dataset  # noqa: E402

SMALL = SimulationConfig(
    start_date="2023-01-01",
    end_date="2024-06-30",
    n_customers=900,
    n_stores=6,
    n_products=40,
    n_anomalies=10,
    seed=7,
)


@pytest.fixture(scope="session")
def sim_config() -> SimulationConfig:
    return SMALL


@pytest.fixture(scope="session")
def raw(sim_config) -> dict:
    """A complete small dataset, generated in memory (nothing written to disk)."""
    return generate_dataset(sim_config, write=False)


@pytest.fixture(scope="session")
def ground_truth(raw) -> dict:
    return raw["_ground_truth"]


@pytest.fixture(scope="session")
def cleaned(raw, sim_config):
    """(silver tables, quality report, quarantine) for the small dataset."""
    tables = {k: v for k, v in raw.items() if not k.startswith("_")}
    return clean_layer(tables, sim_config.start_date, sim_config.end_date)


@pytest.fixture(scope="session")
def star(cleaned) -> dict:
    silver, _report, _quarantine = cleaned
    return build_star_schema(silver)


@pytest.fixture(scope="session")
def fact(star):
    return star["fact_sales"]


@pytest.fixture(scope="session")
def warehouse(star, tmp_path_factory) -> Path:
    """A throwaway SQLite warehouse built from the small dataset."""
    db = tmp_path_factory.mktemp("warehouse") / "test.db"
    load_warehouse(star, db_path=db)
    return db


@pytest.fixture(scope="session")
def full_warehouse() -> Path:
    """The real warehouse, if the pipeline has been run."""
    if not WAREHOUSE_DB.exists():
        pytest.skip("full warehouse not built - run `python -m retailpulse all`")
    return WAREHOUSE_DB
