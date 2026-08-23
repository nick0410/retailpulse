"""RetailPulse dashboard.

Launch with::

    python -m retailpulse dashboard          # or: streamlit run dashboard/app.py

Everything on screen is read from the warehouse and from ``data/outputs``,
which means the dashboard can never disagree with the pipeline - if a number
here looks wrong, the pipeline produced it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from retailpulse.config import OUTPUT_DIR, REPORT_DIR, WAREHOUSE_DB  # noqa: E402
from retailpulse.etl import query, read_table  # noqa: E402
from retailpulse.viz import charts  # noqa: E402
from retailpulse.viz.theme import Theme, format_inr  # noqa: E402

st.set_page_config(page_title="RetailPulse", page_icon="📊", layout="wide")


# --------------------------------------------------------------------------
# Data access (cached - the warehouse does not change while the app is open)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_output(name: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    path = OUTPUT_DIR / f"{name}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=parse_dates or [])


@st.cache_data(show_spinner=False)
def load_table(name: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    try:
        return read_table(name, parse_dates=parse_dates)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict:
    for folder in (OUTPUT_DIR, REPORT_DIR):
        path = folder / f"{name}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return {}


@st.cache_data(show_spinner=False)
def headline_kpis() -> dict:
    row = query(
        """
        SELECT COUNT(DISTINCT transaction_id) AS transactions,
               COUNT(DISTINCT customer_id)    AS customers,
               SUM(line_amount)               AS revenue,
               SUM(gross_margin)              AS margin,
               SUM(quantity)                  AS units
        FROM fact_sales
        """
    ).iloc[0]
    return row.to_dict()


def guard_warehouse() -> bool:
    if WAREHOUSE_DB.exists():
        return True
    st.error("No warehouse found. Build it first:")
    st.code("python -m retailpulse all", language="bash")
    return False


def stat_tile(col, label: str, value: str, caption: str = "") -> None:
    with col:
        st.metric(label, value, help=caption or None)


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
def main() -> None:
    st.title("RetailPulse")
    st.caption("Retail intelligence and demand forecasting - every number on this "
               "page is produced by the pipeline, not typed in.")

    if not guard_warehouse():
        return

    # Match Streamlit's own base theme by default so the chart surface and the
    # page surface agree; the radio only exists to preview the other mode.
    base = (st.get_option("theme.base") or "light").lower()
    mode = st.sidebar.radio("Chart palette", ["light", "dark"],
                            index=1 if base == "dark" else 0,
                            horizontal=True, key="chart_mode")
    theme = Theme(mode)
    theme.register()

    st.sidebar.divider()
    st.sidebar.caption(f"Warehouse: `{WAREHOUSE_DB.name}`")
    summary = load_json("run_summary")
    if summary.get("stage_seconds", {}).get("TOTAL"):
        st.sidebar.caption(f"Last full run: {summary['stage_seconds']['TOTAL']:.0f}s")

    tabs = st.tabs(["Overview", "Data quality", "Customers", "Lifetime value",
                    "Churn", "Basket", "Forecast", "Incidents"])

    # ---------------- Overview ---------------------------------------------
    with tabs[0]:
        kpi = headline_kpis()
        cols = st.columns(5)
        stat_tile(cols[0], "Revenue", format_inr(kpi["revenue"]))
        stat_tile(cols[1], "Gross margin", format_inr(kpi["margin"]),
                  f"{100 * kpi['margin'] / kpi['revenue']:.1f}% of revenue")
        stat_tile(cols[2], "Transactions", f"{int(kpi['transactions']):,}")
        stat_tile(cols[3], "Identified customers", f"{int(kpi['customers']):,}")
        stat_tile(cols[4], "Units sold", f"{int(kpi['units']):,}")

        daily = load_table("mart_daily_total", parse_dates=["date"])
        if not daily.empty:
            st.plotly_chart(charts.revenue_trend(daily, theme), width="stretch", theme=None)

        left, right = st.columns(2)
        by_category = query(
            "SELECT category, SUM(line_amount) AS revenue, SUM(quantity) AS units "
            "FROM fact_sales GROUP BY category ORDER BY revenue DESC")
        with left:
            st.plotly_chart(charts.category_revenue(by_category, theme), width="stretch", theme=None)
        with right:
            by_store = query(
                "SELECT s.store_city AS city, s.store_format, SUM(f.line_amount) AS revenue "
                "FROM fact_sales f JOIN dim_store s ON f.store_id = s.store_id "
                "GROUP BY s.store_city, s.store_format ORDER BY revenue DESC")
            st.markdown("**Revenue by store**")
            st.dataframe(by_store.style.format({"revenue": "{:,.0f}"}),
                         width="stretch", hide_index=True, height=380)

    # ---------------- Data quality -----------------------------------------
    with tabs[1]:
        report = load_output("data_quality_report")
        quarantine = load_output("data_quality_quarantine")
        if report.empty:
            st.info("Run the pipeline to produce a quality report.")
        else:
            etl = summary.get("metrics", {}).get("etl", {})
            cols = st.columns(4)
            stat_tile(cols[0], "Quality score", f"{etl.get('quality_score', 0):.2f}/100")
            stat_tile(cols[1], "Checks run", f"{len(report)}")
            stat_tile(cols[2], "Checks failed", f"{int((~report['passed']).sum())}")
            stat_tile(cols[3], "Rows quarantined", f"{len(quarantine):,}")
            st.caption("Rows failing a *critical* check are moved to quarantine with the "
                       "reason attached - never silently dropped. Warnings are reported "
                       "but still flow through to the warehouse.")
            st.plotly_chart(charts.quality_chart(report, theme), width="stretch", theme=None)
            st.dataframe(report, width="stretch", hide_index=True)
            if not quarantine.empty:
                st.markdown("**Quarantine reasons**")
                st.dataframe(quarantine["reason"].value_counts().rename_axis("reason")
                             .reset_index(name="rows"),
                             width="stretch", hide_index=True)

    # ---------------- Customers --------------------------------------------
    with tabs[2]:
        seg = load_output("rfm_segment_summary")
        pareto = load_output("revenue_concentration")
        if seg.empty:
            st.info("Run `python -m retailpulse analytics`.")
        else:
            left, right = st.columns(2)
            with left:
                st.plotly_chart(charts.segment_revenue(seg, theme), width="stretch", theme=None)
            with right:
                st.plotly_chart(charts.concentration_curve(pareto, theme),
                                width="stretch", theme=None)
            st.markdown("**What to do with each segment**")
            st.dataframe(
                seg[["segment", "customers", "revenue", "revenue_share",
                     "avg_frequency", "avg_recency_days", "action"]]
                .style.format({"revenue": "{:,.0f}", "revenue_share": "{:.1%}",
                               "avg_frequency": "{:.1f}", "avg_recency_days": "{:.0f}"}),
                width="stretch", hide_index=True)

            matrix = load_output("cohort_retention_matrix")
            if not matrix.empty:
                st.plotly_chart(charts.retention_heatmap(matrix, theme),
                                width="stretch", theme=None)
                st.caption("Each row is a sign-up month followed forward. A column that "
                           "holds up as you read down means newer customers are sticking "
                           "better than older ones.")

    # ---------------- CLV ---------------------------------------------------
    with tabs[3]:
        clv = load_output("clv_customers")
        params = load_json("clv_model_params")
        if clv.empty:
            st.info("Run `python -m retailpulse clv`.")
        else:
            cols = st.columns(4)
            stat_tile(cols[0], "Book value (12m)", format_inr(clv["clv_12m"].sum()))
            stat_tile(cols[1], "Median customer", format_inr(clv["clv_12m"].median(), 0))
            stat_tile(cols[2], "Likely churned",
                      f"{int((clv['prob_alive'] < 0.3).sum()):,}",
                      "P(alive) below 30%")
            hold = params.get("holdout", {})
            stat_tile(cols[3], "Holdout error",
                      f"{hold.get('aggregate_error_pct', 0):+.2f}%",
                      "Predicted vs actual transactions in a window the model never saw")

            left, right = st.columns([2, 1])
            with left:
                top = clv.head(25)[["customer_id", "frequency", "prob_alive",
                                    "expected_transactions_12m",
                                    "expected_avg_transaction_value", "clv_12m"]]
                st.markdown("**Most valuable customers over the next 12 months**")
                st.dataframe(top.style.format({
                    "prob_alive": "{:.2f}", "expected_transactions_12m": "{:.1f}",
                    "expected_avg_transaction_value": "{:,.0f}", "clv_12m": "{:,.0f}"}),
                    width="stretch", hide_index=True, height=460)
            with right:
                st.markdown("**Fitted model**")
                st.json({"BG/NBD": params.get("bgnbd", {}),
                         "Gamma-Gamma": params.get("gamma_gamma", {})}, expanded=True)
                st.caption("r and alpha describe how often people buy; a and b describe "
                           "how likely they are to stop. Both are estimated by maximum "
                           "likelihood from three numbers per customer.")

    # ---------------- Churn -------------------------------------------------
    with tabs[4]:
        lift = load_output("churn_decile_lift")
        calib = load_output("churn_calibration")
        imp = load_output("churn_feature_importance")
        targeting = load_output("churn_targeting_simulation")
        churn_metrics = summary.get("metrics", {}).get("churn", {})
        if lift.empty:
            st.info("Run `python -m retailpulse churn`.")
        else:
            g = churn_metrics.get("gbm", {})
            b = churn_metrics.get("logistic_baseline", {})
            cols = st.columns(4)
            stat_tile(cols[0], "ROC-AUC", f"{g.get('roc_auc', 0):.3f}",
                      f"Logistic baseline: {b.get('roc_auc', 0):.3f}")
            stat_tile(cols[1], "PR-AUC", f"{g.get('pr_auc', 0):.3f}")
            stat_tile(cols[2], "Brier score", f"{g.get('brier', 0):.3f}",
                      "Lower is better; measures probability accuracy")
            stat_tile(cols[3], "Base churn rate", f"{g.get('base_churn_rate', 0):.1%}")
            st.caption(f"Trained on the {churn_metrics.get('train_snapshot')} snapshot and "
                       f"tested on {churn_metrics.get('test_snapshot')} - a genuine "
                       "out-of-time split, not a random one.")

            left, right = st.columns(2)
            with left:
                st.plotly_chart(charts.decile_lift_chart(lift, theme), width="stretch", theme=None)
            with right:
                st.plotly_chart(charts.calibration_chart(calib, theme), width="stretch", theme=None)
            if not imp.empty:
                st.plotly_chart(charts.feature_importance(imp, theme), width="stretch", theme=None)
            if not targeting.empty:
                st.markdown("**If the retention team worked the top K%**")
                st.dataframe(targeting.style.format({
                    "target_pct": "{:.0%}", "precision": "{:.1%}", "recall": "{:.1%}",
                    "expected_margin_saved": "{:,.0f}", "campaign_cost": "{:,.0f}",
                    "net_benefit": "{:,.0f}", "roi": "{:.2f}x"}),
                    width="stretch", hide_index=True)

    # ---------------- Basket ------------------------------------------------
    with tabs[5]:
        rules = load_output("basket_association_rules")
        cross = load_output("cross_sell_recommendations")
        if rules.empty:
            st.info("Run `python -m retailpulse analytics`.")
        else:
            cols = st.columns(3)
            stat_tile(cols[0], "Rules found", f"{len(rules):,}")
            stat_tile(cols[1], "Strongest lift", f"{rules['lift'].max():.1f}x")
            stat_tile(cols[2], "Best confidence", f"{rules['confidence'].max():.0%}")
            st.plotly_chart(charts.basket_rules_scatter(rules, theme), width="stretch", theme=None)
            st.caption("Lift is how many times more often two products are bought together "
                       "than chance would predict. Lift above ~3 is worth acting on.")
            show = ["antecedent_names", "consequent_names", "support", "confidence",
                    "lift", "basket_count"]
            st.dataframe(rules[[c for c in show if c in rules]].head(50).style.format(
                {"support": "{:.4f}", "confidence": "{:.2f}", "lift": "{:.1f}"}),
                width="stretch", hide_index=True)
            if not cross.empty:
                st.markdown("**Cross-sell slot: what to show next to each product**")
                st.dataframe(cross.head(30), width="stretch", hide_index=True)

    # ---------------- Forecast ----------------------------------------------
    with tabs[6]:
        folds = load_output("forecast_backtest_folds")
        preds = load_output("forecast_backtest_predictions", parse_dates=["date"])
        future = load_output("forecast_future", parse_dates=["date"])
        daily = load_table("mart_daily_total", parse_dates=["date"])
        if folds.empty:
            st.info("Run `python -m retailpulse forecast`.")
        else:
            avg = folds.groupby("model")[["mape_pct", "smape_pct", "mase"]].mean()
            fmetrics = summary.get("metrics", {}).get("forecast", {})
            cols = st.columns(4)
            stat_tile(cols[0], "Next 28 days",
                      format_inr(future["forecast"].sum()) if not future.empty else "-")
            stat_tile(cols[1], "Hybrid MAPE", f"{avg.loc['hybrid', 'mape_pct']:.1f}%")
            stat_tile(cols[2], "MASE", f"{avg.loc['hybrid', 'mase']:.3f}",
                      "Below 1.0 means better than a seasonal-naive forecast")
            stat_tile(cols[3], "vs naive baseline",
                      f"{fmetrics.get('improvement_vs_seasonal_naive_pct', 0):.0f}% better")

            if not daily.empty and not future.empty:
                st.plotly_chart(charts.forecast_chart(daily, future, theme),
                                width="stretch", theme=None)
            if not preds.empty:
                st.plotly_chart(charts.backtest_chart(preds, theme), width="stretch", theme=None)
            st.markdown("**Accuracy per fold**")
            st.dataframe(folds[["fold", "model", "test_start", "test_end", "mae",
                                "mape_pct", "smape_pct", "mase", "bias_pct"]]
                         .style.format({"mae": "{:,.0f}", "mape_pct": "{:.2f}",
                                        "smape_pct": "{:.2f}", "mase": "{:.3f}",
                                        "bias_pct": "{:+.2f}"}),
                         width="stretch", hide_index=True)
            st.caption("Each fold trains only on data before its own test window, so no "
                       "fold can see its own answer.")

    # ---------------- Incidents ---------------------------------------------
    with tabs[7]:
        events = load_output("anomaly_events", parse_dates=["start_date", "end_date"])
        scored = load_output("anomaly_scored_series", parse_dates=["date"])
        anom = summary.get("metrics", {}).get("anomaly", {})
        if events.empty:
            st.info("Run `python -m retailpulse anomaly`.")
        else:
            truth = anom.get("vs_injected_truth", {})
            cols = st.columns(4)
            stat_tile(cols[0], "Incidents flagged", f"{len(events)}")
            stat_tile(cols[1], "Dips", f"{int((events['direction'] == 'dip').sum())}")
            stat_tile(cols[2], "Spikes", f"{int((events['direction'] == 'spike').sum())}")
            if truth:
                stat_tile(cols[3], "Precision", f"{truth.get('precision', 0):.0%}",
                          f"Recall {truth.get('recall', 0):.0%} against the incidents "
                          "the simulator injected")
            if not scored.empty:
                store = st.selectbox("Store", sorted(scored["store_id"].unique()))
                st.plotly_chart(
                    charts.anomaly_chart(scored, theme, store,
                                         value_col="transactions" if "transactions" in scored
                                         else "revenue"),
                    width="stretch", theme=None)
            st.markdown("**Incident log**")
            st.dataframe(events.style.format({"peak_z": "{:+.2f}",
                                              "revenue_impact": "{:,.0f}"}),
                         width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
