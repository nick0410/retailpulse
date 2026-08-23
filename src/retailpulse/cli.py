"""Command line entry point: ``python -m retailpulse <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

from .config import CONFIG, REPORT_DIR
from . import pipeline

COMMANDS = {
    "generate": "Simulate the raw retail dataset into data/raw",
    "etl": "Validate, clean and load the warehouse (data/warehouse/retailpulse.db)",
    "analytics": "RFM, cohorts and market-basket mining",
    "clv": "Fit BG/NBD + Gamma-Gamma and score customer lifetime value",
    "anomaly": "Detect store-level sales incidents",
    "churn": "Train and evaluate the churn model out-of-time",
    "forecast": "Backtest and run the demand forecast",
    "all": "Run the entire pipeline end to end",
    "dashboard": "Launch the Streamlit dashboard",
}


def _print(payload: dict) -> None:
    print(json.dumps(payload, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="retailpulse",
        description="RetailPulse - retail intelligence and demand forecasting platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="commands:\n" + "\n".join(f"  {k:<12} {v}" for k, v in COMMANDS.items()),
    )
    parser.add_argument("command", choices=list(COMMANDS), help=argparse.SUPPRESS)
    parser.add_argument("--skip-generate", action="store_true",
                        help="reuse the existing data/raw instead of simulating again")
    parser.add_argument("--quiet", action="store_true", help="only log warnings")
    args = parser.parse_args(argv)

    pipeline.configure_logging(logging.WARNING if args.quiet else logging.INFO)
    cmd = args.command

    if cmd == "dashboard":
        app = Path(__file__).resolve().parents[2] / "dashboard" / "app.py"
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])

    if cmd == "all":
        summary = pipeline.run_all(CONFIG, skip_generate=args.skip_generate)
        print(summary.to_json())
        print(f"\nExecutive summary: {REPORT_DIR / 'EXECUTIVE_SUMMARY.md'}")
        return 0

    runners = {
        "generate": pipeline.run_generate,
        "etl": pipeline.run_etl,
        "analytics": pipeline.run_customer_analytics,
        "clv": pipeline.run_clv,
        "anomaly": pipeline.run_anomaly,
        "churn": pipeline.run_churn,
        "forecast": pipeline.run_forecast,
    }
    _print(runners[cmd](CONFIG))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
