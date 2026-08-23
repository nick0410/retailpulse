"""A tiny data-quality engine in the spirit of Great Expectations.

Every check is a small declarative object. Running the suite produces a tidy
report (one row per check) plus the offending row keys, so bad records can be
*quarantined* instead of silently dropped - which is the part most home-grown
pipelines get wrong.

Six quality dimensions are covered, which is also how the results are grouped
in the dashboard:

completeness   - are required fields populated?
uniqueness     - are primary keys actually unique?
validity       - do values obey their domain (ranges, regex, allowed sets)?
consistency    - do derived fields agree with their inputs?
referential    - do foreign keys resolve?
timeliness     - do event dates fall inside the expected window?
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

Severity = str  # "critical" | "warning"


@dataclass
class CheckResult:
    table: str
    check: str
    dimension: str
    severity: Severity
    rows_scanned: int
    rows_failed: int
    threshold: float
    passed: bool
    detail: str = ""
    failed_index: pd.Index = field(default_factory=lambda: pd.Index([]), repr=False)

    @property
    def failure_rate(self) -> float:
        return self.rows_failed / self.rows_scanned if self.rows_scanned else 0.0


@dataclass
class Check:
    """One declarative expectation.

    ``predicate`` returns a boolean Series that is True for *valid* rows.
    ``threshold`` is the fraction of failing rows tolerated before the check
    is marked as failed (0.0 = zero tolerance).
    """

    name: str
    dimension: str
    predicate: Callable[[pd.DataFrame], pd.Series]
    severity: Severity = "critical"
    threshold: float = 0.0
    detail: str = ""

    def run(self, table: str, df: pd.DataFrame) -> CheckResult:
        if df.empty:
            return CheckResult(table, self.name, self.dimension, self.severity,
                               0, 0, self.threshold, True, "empty table")
        ok = self.predicate(df)
        ok = ok.fillna(False).astype(bool)
        failed = df.index[~ok]
        rate = len(failed) / len(df)
        return CheckResult(
            table=table,
            check=self.name,
            dimension=self.dimension,
            severity=self.severity,
            rows_scanned=len(df),
            rows_failed=int(len(failed)),
            threshold=self.threshold,
            passed=bool(rate <= self.threshold),
            detail=self.detail,
            failed_index=failed,
        )


# --------------------------------------------------------------------------
# Reusable predicate builders
# --------------------------------------------------------------------------
def not_null(col: str) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda df: df[col].notna()


def unique(col: str) -> Callable[[pd.DataFrame], pd.Series]:
    """First occurrence of a key is valid; every later copy is a defect.

    Flagging *both* copies would throw away a genuine record along with the
    accidental one, so the surviving row is kept and only the surplus is
    quarantined.
    """
    return lambda df: ~df.duplicated(subset=[col], keep="first")


def in_range(col: str, low: float | None = None, high: float | None = None,
             allow_null: bool = False) -> Callable[[pd.DataFrame], pd.Series]:
    def _pred(df: pd.DataFrame) -> pd.Series:
        s = pd.to_numeric(df[col], errors="coerce")
        ok = pd.Series(True, index=df.index)
        if low is not None:
            ok &= s >= low
        if high is not None:
            ok &= s <= high
        return ok | (s.isna() if allow_null else False)
    return _pred


def matches(col: str, pattern: str) -> Callable[[pd.DataFrame], pd.Series]:
    rx = re.compile(pattern)
    return lambda df: df[col].astype(str).map(lambda v: bool(rx.fullmatch(v)))


def in_set(col: str, allowed: set) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda df: df[col].isin(allowed)


def foreign_key(col: str, valid_keys: set) -> Callable[[pd.DataFrame], pd.Series]:
    return lambda df: df[col].isin(valid_keys)


def between_dates(col: str, start, end) -> Callable[[pd.DataFrame], pd.Series]:
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    def _pred(df: pd.DataFrame) -> pd.Series:
        s = pd.to_datetime(df[col], errors="coerce")
        return (s >= lo) & (s <= hi)
    return _pred


def derived_equals(target: str, parts: tuple[str, str], tol: float = 0.02):
    """Check that ``target ~= parts[0] * parts[1]`` (e.g. amount = qty x price)."""
    def _pred(df: pd.DataFrame) -> pd.Series:
        expected = pd.to_numeric(df[parts[0]], errors="coerce") * pd.to_numeric(df[parts[1]], errors="coerce")
        actual = pd.to_numeric(df[target], errors="coerce")
        return (actual - expected).abs() <= tol + 1e-9
    return _pred


# --------------------------------------------------------------------------
# Suite
# --------------------------------------------------------------------------
class QualitySuite:
    """A named bundle of checks bound to one table."""

    def __init__(self, table: str, checks: list[Check]):
        self.table = table
        self.checks = checks

    def run(self, df: pd.DataFrame) -> list[CheckResult]:
        return [c.run(self.table, df) for c in self.checks]


def build_suites(raw: dict[str, pd.DataFrame], start_date: str, end_date: str) -> dict[str, QualitySuite]:
    """Wire up the full expectation suite for the raw layer."""
    customer_keys = set(raw["customers"]["customer_id"])
    product_keys = set(raw["products"]["product_id"])
    store_keys = set(raw["stores"]["store_id"])
    txn_keys = set(raw["transactions"]["transaction_id"])

    email_rx = r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}"

    customers = QualitySuite("customers", [
        Check("customer_id_not_null", "completeness", not_null("customer_id")),
        Check("customer_id_unique", "uniqueness", unique("customer_id")),
        Check("city_populated", "completeness", not_null("city"),
              severity="warning", threshold=0.01,
              detail="Missing city blocks region-level roll-ups"),
        Check("email_well_formed", "validity", matches("email", email_rx),
              severity="warning", threshold=0.005),
        Check("signup_date_in_window", "timeliness", between_dates("signup_date", start_date, end_date)),
        Check("loyalty_tier_known", "validity",
              in_set("loyalty_tier", {"Bronze", "Silver", "Gold", "Platinum"})),
    ])

    products = QualitySuite("products", [
        Check("product_id_unique", "uniqueness", unique("product_id")),
        Check("base_price_positive", "validity", in_range("base_price", low=1.0)),
        Check("cost_below_price", "consistency",
              lambda df: df["unit_cost"] < df["base_price"],
              detail="Selling below cost would break margin reporting"),
        Check("category_populated", "completeness", not_null("category")),
    ])

    stores = QualitySuite("stores", [
        Check("store_id_unique", "uniqueness", unique("store_id")),
        Check("floor_area_sane", "validity", in_range("floor_area_sqft", low=200, high=200_000)),
    ])

    transactions = QualitySuite("transactions", [
        Check("transaction_id_unique", "uniqueness", unique("transaction_id"),
              detail="Duplicate transaction rows double-count revenue"),
        Check("store_fk_resolves", "referential", foreign_key("store_id", store_keys)),
        Check("customer_fk_resolves", "referential",
              lambda df: df["customer_id"].isna() | df["customer_id"].isin(customer_keys),
              detail="NULL customer_id is legitimate: anonymous walk-in"),
        Check("date_in_window", "timeliness", between_dates("date", start_date, end_date)),
        Check("channel_known", "validity", in_set("channel", {"in_store", "app", "web"})),
    ])

    items = QualitySuite("transaction_items", [
        Check("line_id_unique", "uniqueness", unique("line_id")),
        Check("transaction_fk_resolves", "referential", foreign_key("transaction_id", txn_keys),
              detail="Orphan lines cannot be attributed to a basket"),
        Check("product_fk_resolves", "referential", foreign_key("product_id", product_keys)),
        Check("quantity_positive", "validity", in_range("quantity", low=1),
              detail="Negative quantity means a return booked into the sales table"),
        Check("unit_price_positive", "validity", in_range("unit_price", low=0.01),
              detail="Zero price indicates a broken feed"),
        Check("line_amount_consistent", "consistency",
              derived_equals("line_amount", ("quantity", "unit_price")),
              threshold=0.0),
        Check("discount_in_range", "validity", in_range("discount_pct", low=0.0, high=0.9)),
    ])

    return {
        "customers": customers,
        "products": products,
        "stores": stores,
        "transactions": transactions,
        "transaction_items": items,
    }


def run_quality_suite(raw: dict[str, pd.DataFrame], start_date: str,
                      end_date: str) -> tuple[pd.DataFrame, dict[str, pd.Index], list[CheckResult]]:
    """Execute every suite.

    Returns ``(report, per-table quarantine index, raw results)``. The raw
    results are handed back because a failing row's *reason* can only be read
    off the original evaluation: set-level checks such as uniqueness give a
    different answer when re-run on a subset of the rows, so re-deriving
    reasons later would silently mislabel them.
    """
    suites = build_suites(raw, start_date, end_date)
    results: list[CheckResult] = []
    quarantine: dict[str, pd.Index] = {}

    for table, suite in suites.items():
        table_results = suite.run(raw[table])
        results.extend(table_results)
        bad = pd.Index([])
        for res in table_results:
            # Only critical failures get quarantined; warnings are reported
            # but the rows still flow through to the warehouse.
            if res.severity == "critical" and res.rows_failed:
                bad = bad.union(res.failed_index)
        quarantine[table] = bad

    report = pd.DataFrame(
        [
            {
                "table": r.table,
                "check": r.check,
                "dimension": r.dimension,
                "severity": r.severity,
                "rows_scanned": r.rows_scanned,
                "rows_failed": r.rows_failed,
                "failure_rate": round(r.failure_rate, 6),
                "threshold": r.threshold,
                "passed": r.passed,
                "detail": r.detail,
            }
            for r in results
        ]
    )
    return report, quarantine, results


def quality_score(report: pd.DataFrame) -> float:
    """A single 0-100 headline number for the dashboard.

    Critical checks carry three times the weight of warnings, and a check is
    scored by how clean it is, not merely pass/fail.
    """
    if report.empty:
        return 100.0
    weights = np.where(report["severity"] == "critical", 3.0, 1.0)
    cleanliness = 1.0 - report["failure_rate"].to_numpy()
    return float(np.round(100.0 * np.average(cleanliness, weights=weights), 2))
