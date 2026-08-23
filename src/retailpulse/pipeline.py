"""End-to-end orchestration.

The whole platform runs as one command. Every stage is timed, logged, and
writes its artefacts to ``data/outputs`` so the dashboard, the tests and a
human reviewer all read the same numbers.

    generate -> ingest+validate -> warehouse -> analytics -> ml -> report

Stages are deliberately re-runnable in isolation: each reads what it needs
from the warehouse rather than from the stage before it in memory.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import pandas as pd

from .analytics import (anomaly_events, build_cohort_table, build_rfm,
                        cohort_quality_trend, cross_sell_recommendations,
                        detect_anomalies, evaluate_against_truth, mine_rules,
                        pareto_concentration, retention_curve, retention_matrix,
                        segment_summary)
from .analytics.clv import (BetaGeoFitter, GammaGammaFitter, customer_lifetime_value,
                            summary_from_transactions, validate_holdout)
from .config import CONFIG, OUTPUT_DIR, RAW_DIR, REPORT_DIR, WAREHOUSE_DB, Config
from .etl import (build_star_schema, clean_layer, load_raw, load_warehouse,
                  quality_score, read_table)
from .generate import generate_dataset
from .ml import (build_training_frame, promo_calendar_from_promotions,
                 targeting_simulation, train_churn_model, walk_forward_backtest)
from .ml.forecast import HybridForecaster

log = logging.getLogger("retailpulse")


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


@dataclass
class RunSummary:
    stages: dict[str, float] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"stage_seconds": {k: round(v, 2) for k, v in self.stages.items()},
                           "metrics": self.metrics}, indent=2, default=str)


@contextmanager
def stage(name: str, summary: RunSummary):
    log.info("-> %s", name)
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    summary.stages[name] = elapsed
    log.info("   %s done in %.1fs", name, elapsed)


def _write(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------
def run_generate(cfg: Config = CONFIG) -> dict:
    tables = generate_dataset(cfg.simulation)
    return {"raw_transactions": len(tables["transactions"]),
            "raw_line_items": len(tables["transaction_items"]),
            "customers": len(tables["customers"])}


def run_etl(cfg: Config = CONFIG) -> dict:
    raw = load_raw()
    silver, report, quarantine = clean_layer(
        raw, cfg.simulation.start_date, cfg.simulation.end_date)
    star = build_star_schema(silver)
    written = load_warehouse(star)

    _write(report, "data_quality_report")
    _write(quarantine, "data_quality_quarantine")

    failed = report[~report["passed"]]
    return {
        "quality_score": quality_score(report),
        "checks_run": int(len(report)),
        "checks_failed": int(len(failed)),
        "rows_quarantined": int(len(quarantine)),
        "warehouse_rows": {k: int(v) for k, v in written.items()},
        "failed_checks": failed[["table", "check", "rows_failed"]].to_dict("records"),
    }


def run_customer_analytics(cfg: Config = CONFIG) -> dict:
    fact = read_table("fact_sales", parse_dates=["date"])
    dim_customer = read_table("dim_customer", parse_dates=["signup_date"])
    dim_product = read_table("dim_product")

    # ---- RFM ---------------------------------------------------------------
    rfm = build_rfm(fact, quantiles=cfg.analytics.rfm_quantiles)
    summary = segment_summary(rfm)
    pareto = pareto_concentration(rfm)
    _write(rfm, "rfm_customers")
    _write(summary, "rfm_segment_summary")
    _write(pareto, "revenue_concentration")

    # ---- cohorts -----------------------------------------------------------
    cohorts = build_cohort_table(fact, dim_customer)
    _write(cohorts, "cohort_table")
    _write(retention_matrix(cohorts).reset_index(), "cohort_retention_matrix")
    curve = retention_curve(cohorts)
    _write(curve, "retention_curve")
    _write(cohort_quality_trend(cohorts), "cohort_quality_trend")

    # ---- market basket -----------------------------------------------------
    frequent, rules = mine_rules(
        fact, dim_product,
        min_support=cfg.analytics.basket_min_support,
        min_confidence=cfg.analytics.basket_min_confidence,
        min_lift=cfg.analytics.basket_min_lift,
        max_len=cfg.analytics.basket_max_len,
    )
    _write(frequent.assign(itemset=frequent["itemset"].astype(str)), "basket_frequent_itemsets")
    exportable = rules.copy()
    for col in ("antecedent", "consequent", "antecedent_ids", "consequent_ids"):
        if col in exportable:
            exportable[col] = exportable[col].astype(str)
    _write(exportable, "basket_association_rules")
    _write(cross_sell_recommendations(rules), "cross_sell_recommendations")

    top_rule = rules.iloc[0] if len(rules) else None
    return {
        "customers_segmented": int(len(rfm)),
        "segments": summary[["segment", "customers", "revenue_share"]].to_dict("records"),
        "top_1pct_revenue_share": float(pareto.loc[pareto["top_customer_pct"] == 0.01, "revenue_share"].iat[0]),
        "top_20pct_revenue_share": float(pareto.loc[pareto["top_customer_pct"] == 0.20, "revenue_share"].iat[0]),
        "month_1_retention": float(curve.loc[curve["months_since_signup"] == 1, "retention_rate"].iat[0])
        if len(curve) > 1 else None,
        "frequent_itemsets": int(len(frequent)),
        "association_rules": int(len(rules)),
        "top_rule": None if top_rule is None else
        f"{top_rule['antecedent_names']} -> {top_rule['consequent_names']} (lift {top_rule['lift']:.1f})",
    }


def run_clv(cfg: Config = CONFIG) -> dict:
    fact = read_table("fact_sales", parse_dates=["date"])
    summary = summary_from_transactions(fact)

    bgf = BetaGeoFitter().fit(summary["frequency"], summary["recency"], summary["T"])
    ggf = GammaGammaFitter().fit(summary["frequency"], summary["monetary_value"])
    clv = customer_lifetime_value(bgf, ggf, summary, months=12)
    _write(clv, "clv_customers")

    end = pd.to_datetime(fact["date"]).max()
    calibration_end = end - pd.Timedelta(days=180)
    holdout = validate_holdout(bgf, fact, calibration_end, end)
    detail = holdout.pop("detail")
    _write(detail, "clv_holdout_detail")

    params = {"bgnbd": bgf.summary(), "gamma_gamma": ggf.summary(), "holdout": holdout}
    (OUTPUT_DIR / "clv_model_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    return {
        **params,
        "total_clv_12m": round(float(clv["clv_12m"].sum()), 2),
        "median_clv_12m": round(float(clv["clv_12m"].median()), 2),
        "customers_at_risk": int((clv["prob_alive"] < 0.3).sum()),
    }


def run_anomaly(cfg: Config = CONFIG) -> dict:
    mart = read_table("mart_daily_store", parse_dates=["date"])
    a = cfg.analytics
    scored = detect_anomalies(mart, value_col=a.anomaly_value_col,
                              period=a.anomaly_season_period,
                              z_threshold=a.anomaly_z_threshold,
                              method=a.anomaly_method, fdr_q=a.anomaly_fdr_q)
    events = anomaly_events(scored, value_col=a.anomaly_value_col)
    _write(scored, "anomaly_scored_series")
    _write(events, "anomaly_events")

    result = {"events_detected": int(len(events)),
              "dips": int((events["direction"] == "dip").sum()) if len(events) else 0,
              "spikes": int((events["direction"] == "spike").sum()) if len(events) else 0}

    truth_path = RAW_DIR / "anomaly_ground_truth.csv"
    if truth_path.exists():
        result["vs_injected_truth"] = evaluate_against_truth(events, pd.read_csv(truth_path))
    return result


def run_churn(cfg: Config = CONFIG) -> dict:
    fact = read_table("fact_sales", parse_dates=["date"])
    dim_customer = read_table("dim_customer", parse_dates=["signup_date"])
    a = cfg.analytics

    train = build_training_frame(fact, dim_customer, pd.Timestamp(a.churn_train_snapshot),
                                 horizon_days=a.churn_inactivity_days,
                                 lookback_days=a.churn_feature_window_days)
    test = build_training_frame(fact, dim_customer, pd.Timestamp(a.churn_test_snapshot),
                                horizon_days=a.churn_inactivity_days,
                                lookback_days=a.churn_feature_window_days)
    result = train_churn_model(train, test)
    targeting = targeting_simulation(result.scored)

    _write(result.scored, "churn_scores")
    _write(result.lift, "churn_decile_lift")
    _write(result.calibration, "churn_calibration")
    _write(result.importance, "churn_feature_importance")
    _write(targeting, "churn_targeting_simulation")

    # Two different questions, two different answers. With a base churn rate
    # this high and a cheap contact channel, blanket targeting wins on total
    # rupees - but it is the model that makes the *efficient* slice possible,
    # which is what ROI shows. Both are reported rather than cherry-picked.
    best_value = targeting.loc[targeting["net_benefit"].idxmax()]
    best_roi = targeting.loc[targeting["roi"].idxmax()]
    base_rate = result.metrics["base_churn_rate"]
    return {
        "train_snapshot": a.churn_train_snapshot,
        "test_snapshot": a.churn_test_snapshot,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "gbm": result.metrics,
        "logistic_baseline": result.baseline_metrics,
        "top_decile_lift": float(result.lift["lift"].iat[0]),
        # A lift of 1/base_rate means the decile is *entirely* churners: that
        # is the ceiling, not a disappointing number.
        "max_possible_lift": round(1.0 / base_rate, 3),
        "top_features": result.importance.head(5)["feature"].tolist() if len(result.importance) else [],
        "best_campaign_by_value": {"target_pct": float(best_value["target_pct"]),
                                   "net_benefit": float(best_value["net_benefit"]),
                                   "roi": float(best_value["roi"])},
        "best_campaign_by_roi": {"target_pct": float(best_roi["target_pct"]),
                                 "net_benefit": float(best_roi["net_benefit"]),
                                 "roi": float(best_roi["roi"]),
                                 "precision": float(best_roi["precision"])},
    }


def run_forecast(cfg: Config = CONFIG) -> dict:
    daily = read_table("mart_daily_total", parse_dates=["date"])
    promos = read_table("dim_promotion")
    promo_calendar = promo_calendar_from_promotions(promos)
    f = cfg.forecast

    folds, preds = walk_forward_backtest(
        daily, value_col="revenue", horizon=f.horizon_days, folds=f.backtest_folds,
        season_period=f.season_period, promo_calendar=promo_calendar)
    _write(folds, "forecast_backtest_folds")
    _write(preds, "forecast_backtest_predictions")

    # Final model on the full history, projecting past the end of the data.
    model = HybridForecaster(season_period=f.season_period).fit(
        daily, value_col="revenue", promo_calendar=promo_calendar)
    future = model.predict(f.horizon_days, promo_calendar=promo_calendar)
    _write(future, "forecast_future")

    avg = folds.groupby("model")[["mae", "mape_pct", "smape_pct", "mase"]].mean().round(4)
    hybrid_mase = float(avg.loc["hybrid", "mase"])
    naive_mase = float(avg.loc["seasonal_naive", "mase"])
    return {
        "horizon_days": f.horizon_days,
        "folds": f.backtest_folds,
        "holt_winters_params": model.params(),
        "accuracy_by_model": avg.to_dict("index"),
        "improvement_vs_seasonal_naive_pct": round(100 * (1 - hybrid_mase / naive_mase), 2),
        "next_period_revenue_forecast": round(float(future["forecast"].sum()), 2),
    }


# --------------------------------------------------------------------------
# Executive report
# --------------------------------------------------------------------------
def write_executive_summary(metrics: dict) -> None:
    """A one-page markdown brief a non-technical reader can act on."""
    etl = metrics.get("etl", {})
    cust = metrics.get("customer_analytics", {})
    clv = metrics.get("clv", {})
    churn = metrics.get("churn", {})
    fc = metrics.get("forecast", {})
    anom = metrics.get("anomaly", {})

    def pct(x) -> str:
        return "n/a" if x is None else f"{100 * float(x):.1f}%"

    lines = [
        "# RetailPulse - Executive Summary",
        "",
        f"_Generated automatically by the pipeline. Warehouse: `{WAREHOUSE_DB.name}`._",
        "",
        "## 1. Can we trust the data?",
        f"- Data quality score: **{etl.get('quality_score', 'n/a')}/100** "
        f"across {etl.get('checks_run', 0)} automated checks.",
        f"- {etl.get('checks_failed', 0)} checks failed; "
        f"{etl.get('rows_quarantined', 0):,} rows were quarantined rather than silently dropped.",
        "",
        "## 2. Who are the customers?",
        f"- Segmented **{cust.get('customers_segmented', 0):,}** identified customers into RFM segments.",
        f"- The top 1% of customers produce **{pct(cust.get('top_1pct_revenue_share'))}** of revenue; "
        f"the top 20% produce **{pct(cust.get('top_20pct_revenue_share'))}**.",
        f"- Month-1 repeat rate is **{pct(cust.get('month_1_retention'))}**.",
        "",
        "## 3. What are they worth?",
        f"- Modelled 12-month customer value: **Rs {clv.get('total_clv_12m', 0):,.0f}** across the book.",
        f"- {clv.get('customers_at_risk', 0):,} customers are more likely dead than alive "
        "(P(alive) < 0.30).",
        f"- Holdout check: predicted {clv.get('holdout', {}).get('predicted_total_transactions', 'n/a')} "
        f"transactions vs {clv.get('holdout', {}).get('actual_total_transactions', 'n/a')} actual "
        f"({clv.get('holdout', {}).get('aggregate_error_pct', 'n/a')}% error).",
        "",
        "## 4. Who is about to leave?",
        f"- Out-of-time ROC-AUC **{churn.get('gbm', {}).get('roc_auc', 'n/a')}**, "
        f"PR-AUC {churn.get('gbm', {}).get('pr_auc', 'n/a')} "
        f"(trained {churn.get('train_snapshot')}, tested {churn.get('test_snapshot')}).",
        f"- The riskiest decile churns at **{churn.get('top_decile_lift', 'n/a')}x** the base rate "
        f"(the ceiling is {churn.get('max_possible_lift', 'n/a')}x, i.e. a decile of pure churners - "
        "with a base rate this high, ranking has little room to run).",
        f"- Most efficient campaign: contact the top "
        f"{100 * churn.get('best_campaign_by_roi', {}).get('target_pct', 0):.0f}% "
        f"at **{churn.get('best_campaign_by_roi', {}).get('roi', 0)}x ROI** "
        f"({pct(churn.get('best_campaign_by_roi', {}).get('precision'))} of them really do churn).",
        f"- Largest total return: contact the top "
        f"{100 * churn.get('best_campaign_by_value', {}).get('target_pct', 0):.0f}% "
        f"for Rs {churn.get('best_campaign_by_value', {}).get('net_benefit', 0):,.0f} net.",
        "",
        "## 5. What sells together?",
        f"- {cust.get('association_rules', 0)} association rules mined from "
        f"{cust.get('frequent_itemsets', 0)} frequent itemsets.",
        f"- Strongest: {cust.get('top_rule', 'n/a')}.",
        "",
        "## 6. What happens next?",
        f"- {fc.get('horizon_days', 0)}-day revenue forecast: **Rs {fc.get('next_period_revenue_forecast', 0):,.0f}**.",
        f"- Backtested across {fc.get('folds', 0)} walk-forward folds, the hybrid model is "
        f"**{fc.get('improvement_vs_seasonal_naive_pct', 0)}% more accurate** (MASE) than a "
        "seasonal-naive baseline.",
        "",
        "## 7. What went wrong in the stores?",
        f"- {anom.get('events_detected', 0)} incidents flagged "
        f"({anom.get('dips', 0)} dips, {anom.get('spikes', 0)} spikes).",
    ]
    truth = anom.get("vs_injected_truth")
    if truth:
        lines.append(
            f"- Against the {truth['injected']} incidents the simulator injected: "
            f"recall {truth['recall']}, precision {truth['precision']}."
        )
    lines += ["", "---", "", "Run `python -m retailpulse dashboard` for the interactive version."]
    (REPORT_DIR / "EXECUTIVE_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# Full run
# --------------------------------------------------------------------------
def run_all(cfg: Config = CONFIG, skip_generate: bool = False) -> RunSummary:
    summary = RunSummary()
    total_start = time.perf_counter()

    if not skip_generate:
        with stage("generate", summary):
            summary.metrics["generate"] = run_generate(cfg)
    with stage("etl", summary):
        summary.metrics["etl"] = run_etl(cfg)
    with stage("customer_analytics", summary):
        summary.metrics["customer_analytics"] = run_customer_analytics(cfg)
    with stage("clv", summary):
        summary.metrics["clv"] = run_clv(cfg)
    with stage("anomaly", summary):
        summary.metrics["anomaly"] = run_anomaly(cfg)
    with stage("churn", summary):
        summary.metrics["churn"] = run_churn(cfg)
    with stage("forecast", summary):
        summary.metrics["forecast"] = run_forecast(cfg)

    summary.stages["TOTAL"] = time.perf_counter() - total_start
    (REPORT_DIR / "run_summary.json").write_text(summary.to_json(), encoding="utf-8")
    write_executive_summary(summary.metrics)
    log.info("Pipeline finished in %.1fs", summary.stages["TOTAL"])
    log.info("Report: %s", REPORT_DIR / "EXECUTIVE_SUMMARY.md")
    return summary
