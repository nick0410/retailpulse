"""Central configuration for the RetailPulse platform.

Everything that a reviewer might want to tweak (date ranges, sizes, model
hyper-parameters, file locations) lives here so that no module has to hardcode
magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
WAREHOUSE_DIR = DATA_DIR / "warehouse"
OUTPUT_DIR = DATA_DIR / "outputs"
REPORT_DIR = ROOT / "reports"

WAREHOUSE_DB = WAREHOUSE_DIR / "retailpulse.db"

for _d in (RAW_DIR, WAREHOUSE_DIR, OUTPUT_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Simulation settings
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SimulationConfig:
    """Controls the synthetic retail data generator."""

    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"
    n_customers: int = 12_000
    n_stores: int = 12
    n_products: int = 80
    seed: int = 42

    # BG/NBD ground-truth parameters. The generator literally samples customer
    # behaviour from this process, so the fitted model can be checked against it.
    bgnbd_r: float = 0.70       # gamma shape for purchase rate
    bgnbd_alpha: float = 6.0    # gamma scale for purchase rate (in weeks)
    bgnbd_a: float = 0.80       # beta alpha for dropout
    bgnbd_b: float = 3.20       # beta beta for dropout

    # Gamma-Gamma ground truth for spend per transaction.
    gg_p: float = 5.0
    gg_q: float = 3.5
    gg_v: float = 900.0

    # Share of days that carry a promotion, and how strongly price moves demand.
    promo_rate: float = 0.12
    price_elasticity: float = -1.8

    # Number of injected operational anomalies (stockouts / viral spikes).
    n_anomalies: int = 28


# --------------------------------------------------------------------------
# Analytics settings
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AnalyticsConfig:
    """Thresholds used by the analytics layer."""

    rfm_quantiles: int = 5
    churn_inactivity_days: int = 90       # label definition for the churn model
    churn_feature_window_days: int = 365  # history used to build features
    basket_min_support: float = 0.0012
    basket_min_confidence: float = 0.10
    basket_min_lift: float = 1.15
    basket_max_len: int = 3
    anomaly_z_threshold: float = 3.5
    anomaly_season_period: int = 7
    anomaly_method: str = "nb_tail"      # "nb_tail" (counts) or "robust_z"
    anomaly_value_col: str = "transactions"
    anomaly_fdr_q: float = 0.01          # Benjamini-Hochberg false-discovery rate

    # Out-of-time design for the churn model: features are built as of the
    # snapshot, the label looks forward `churn_inactivity_days` from there.
    churn_train_snapshot: str = "2024-06-30"
    churn_test_snapshot: str = "2024-09-30"


# --------------------------------------------------------------------------
# Forecasting settings
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ForecastConfig:
    """Configuration for the hybrid demand forecaster."""

    horizon_days: int = 28
    season_period: int = 7
    backtest_folds: int = 4
    residual_lags: tuple[int, ...] = (1, 2, 3, 7, 14, 28)
    residual_rolling: tuple[int, ...] = (7, 28)
    gbm_estimators: int = 300
    gbm_learning_rate: float = 0.05
    gbm_max_depth: int = 3


@dataclass(frozen=True)
class Config:
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)


CONFIG = Config()
