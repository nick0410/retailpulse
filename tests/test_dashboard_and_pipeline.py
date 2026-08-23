"""The dashboard must render, and the whole pipeline must run end to end.

The dashboard test uses Streamlit's own AppTest harness, which executes the
real app script and surfaces any exception from any tab - far more reliable
than clicking around a browser, and it runs in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_renders_without_errors(full_warehouse):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=300)
    app.run()

    assert not app.exception, [str(e.value) for e in app.exception]
    assert [t.value for t in app.title] == ["RetailPulse"]
    assert len(app.tabs) == 8, "a tab went missing"
    # Every tab should have produced at least one stat tile or table.
    assert len(app.metric) > 15
    assert len(app.dataframe) > 5


RUNNER_FOR_COMMAND = {
    "generate": "run_generate",
    "etl": "run_etl",
    "analytics": "run_customer_analytics",
    "clv": "run_clv",
    "anomaly": "run_anomaly",
    "churn": "run_churn",
    "forecast": "run_forecast",
}


def test_every_cli_command_has_a_runner():
    """`python -m retailpulse <cmd>` must not advertise a stage that cannot run."""
    from retailpulse import pipeline
    from retailpulse.cli import COMMANDS

    documented = set(COMMANDS) - {"all", "dashboard"}
    assert documented == set(RUNNER_FOR_COMMAND), "CLI commands and runners drifted apart"
    for command, runner in RUNNER_FOR_COMMAND.items():
        assert callable(getattr(pipeline, runner, None)), f"no runner for `{command}`"


@pytest.mark.slow
def test_full_pipeline_runs_and_reports(tmp_path, monkeypatch):
    """The headline claim: one command, and everything downstream exists."""
    from retailpulse import pipeline
    from retailpulse.config import CONFIG, OUTPUT_DIR, REPORT_DIR

    summary = pipeline.run_all(CONFIG, skip_generate=True)

    assert summary.stages["TOTAL"] > 0
    for stage in ("etl", "customer_analytics", "clv", "anomaly", "churn", "forecast"):
        assert stage in summary.metrics, f"{stage} produced no metrics"

    assert summary.metrics["etl"]["quality_score"] > 90
    assert summary.metrics["churn"]["gbm"]["roc_auc"] > 0.7
    assert summary.metrics["forecast"]["improvement_vs_seasonal_naive_pct"] > 0
    assert summary.metrics["clv"]["total_clv_12m"] > 0

    assert (REPORT_DIR / "EXECUTIVE_SUMMARY.md").exists()
    assert (REPORT_DIR / "run_summary.json").exists()
    for artefact in ("rfm_customers", "clv_customers", "churn_scores",
                     "forecast_future", "anomaly_events", "data_quality_report"):
        assert (OUTPUT_DIR / f"{artefact}.csv").exists(), f"{artefact}.csv was not written"
