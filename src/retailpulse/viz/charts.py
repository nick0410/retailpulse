"""Reusable plotly figure builders.

Each function takes a tidy frame and returns a figure. Keeping them here (and
out of the Streamlit app) means the charts can be built and eyeballed from a
script or a notebook, and the dashboard file stays readable.

House rules applied throughout: one y-axis per chart, a legend whenever two or
more series share a plot, thin marks, recessive grid lines, and no number
printed on every point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .theme import Theme


# --------------------------------------------------------------------------
# Time series
# --------------------------------------------------------------------------
def revenue_trend(daily: pd.DataFrame, theme: Theme, value_col: str = "revenue",
                  smooth_window: int = 28) -> go.Figure:
    """Daily revenue with a trailing average so the weekly saw-tooth reads."""
    d = daily.sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["date"], y=d[value_col], name="Daily",
        mode="lines", line=dict(width=1, color=theme.series(0)), opacity=0.35,
        hovertemplate="%{x|%d %b %Y}<br>Rs %{y:,.0f}<extra>Daily</extra>"))
    fig.add_trace(go.Scatter(
        x=d["date"], y=d[value_col].rolling(smooth_window, min_periods=1).mean(),
        name=f"{smooth_window}-day average", mode="lines",
        line=dict(width=2, color=theme.series(0)),
        hovertemplate="%{x|%d %b %Y}<br>Rs %{y:,.0f}<extra>Average</extra>"))
    fig.update_layout(title="Revenue over time", yaxis_title="Revenue (Rs)",
                      xaxis_title=None, height=380)
    return fig


def forecast_chart(history: pd.DataFrame, future: pd.DataFrame, theme: Theme,
                   value_col: str = "revenue", tail_days: int = 120) -> go.Figure:
    """Recent actuals, then the forward projection, on one continuous axis."""
    hist = history.sort_values("date").tail(tail_days)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist[value_col], name="Actual", mode="lines",
        line=dict(width=2, color=theme.series(0)),
        hovertemplate="%{x|%d %b}<br>Rs %{y:,.0f}<extra>Actual</extra>"))
    # Join the two lines so the forecast does not float away from history.
    bridge_x = [hist["date"].iloc[-1]] + list(future["date"])
    bridge_y = [hist[value_col].iloc[-1]] + list(future["forecast"])
    fig.add_trace(go.Scatter(
        x=bridge_x, y=bridge_y, name="Forecast", mode="lines",
        line=dict(width=2, color=theme.series(1), dash="dot"),
        hovertemplate="%{x|%d %b}<br>Rs %{y:,.0f}<extra>Forecast</extra>"))
    fig.add_vline(x=hist["date"].iloc[-1], line_width=1, line_dash="dash",
                  line_color=theme.ink["grid"])
    fig.update_layout(title="Demand forecast", yaxis_title="Revenue (Rs)", height=380)
    return fig


def backtest_chart(preds: pd.DataFrame, theme: Theme) -> go.Figure:
    """Actual vs each model across the walk-forward folds."""
    fig = go.Figure()
    series = [("actual", "Actual", 0, "solid"),
              ("hybrid", "Hybrid (HW + GBM)", 1, "solid"),
              ("seasonal_naive", "Seasonal naive", 2, "dot")]
    for col, label, slot, dash in series:
        if col not in preds:
            continue
        fig.add_trace(go.Scatter(
            x=preds["date"], y=preds[col], name=label, mode="lines",
            line=dict(width=2 if col == "actual" else 1.6,
                      color=theme.series(slot), dash=dash),
            hovertemplate="%{x|%d %b}<br>Rs %{y:,.0f}<extra>" + label + "</extra>"))
    fig.update_layout(title="Walk-forward backtest", yaxis_title="Revenue (Rs)", height=380)
    return fig


# --------------------------------------------------------------------------
# Magnitude
# --------------------------------------------------------------------------
def _bar(y_labels, values, theme: Theme, color: str | None = None,
         text: list[str] | None = None, hover: str = "%{x:,.0f}") -> go.Figure:
    fig = go.Figure(go.Bar(
        x=values, y=y_labels, orientation="h",
        marker=dict(color=color or theme.series(0),
                    line=dict(width=2, color=theme.surface)),
        text=text, textposition="auto",
        hovertemplate=hover + "<extra></extra>"))
    # Category names on a horizontal bar chart are long and live in a narrow
    # column; automargin lets the axis claim the width it needs instead of
    # truncating the labels.
    fig.update_layout(bargap=0.28, hovermode="closest")
    fig.update_yaxes(automargin=True)
    return fig


def segment_revenue(summary: pd.DataFrame, theme: Theme) -> go.Figure:
    """Revenue by RFM segment - the ranking is the message, so bars sorted."""
    d = summary.sort_values("revenue")
    fig = _bar(d["segment"], d["revenue"], theme,
               text=[f"{s:.0%}" for s in d["revenue_share"]],
               hover="Rs %{x:,.0f}")
    fig.update_layout(title="Revenue by customer segment",
                      xaxis_title="Revenue (Rs)", height=380)
    return fig


def category_revenue(fact_by_category: pd.DataFrame, theme: Theme) -> go.Figure:
    d = fact_by_category.sort_values("revenue")
    fig = _bar(d["category"], d["revenue"], theme, hover="Rs %{x:,.0f}")
    fig.update_layout(title="Revenue by category", xaxis_title="Revenue (Rs)", height=380)
    return fig


def feature_importance(importance: pd.DataFrame, theme: Theme, top_n: int = 12) -> go.Figure:
    d = importance.head(top_n).sort_values("importance")
    fig = _bar(d["feature"], d["importance"], theme, hover="%{x:.4f}")
    fig.update_layout(title="What the churn model actually uses",
                      xaxis_title="Drop in ROC-AUC when the column is shuffled",
                      height=420)
    return fig


def decile_lift_chart(lift: pd.DataFrame, theme: Theme) -> go.Figure:
    """Lift per risk decile against the 1.0 'no better than random' line."""
    fig = go.Figure(go.Bar(
        x=lift["decile"], y=lift["lift"],
        marker=dict(color=theme.series(0), line=dict(width=2, color=theme.surface)),
        hovertemplate="Decile %{x}<br>Lift %{y:.2f}x<extra></extra>"))
    fig.add_hline(y=1.0, line_width=1, line_dash="dash", line_color=theme.ink["secondary"],
                  annotation_text="random", annotation_position="right")
    fig.update_layout(title="Churn risk by decile", xaxis_title="Risk decile (1 = riskiest)",
                      yaxis_title="Lift vs base rate", height=360, hovermode="closest")
    return fig


def calibration_chart(calibration: pd.DataFrame, theme: Theme) -> go.Figure:
    """Are the probabilities honest? Predicted vs observed, against y = x."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name="Perfect calibration", mode="lines",
        line=dict(width=1, dash="dash", color=theme.ink["secondary"]),
        hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=calibration["predicted"], y=calibration["observed"], name="Model",
        mode="lines+markers", line=dict(width=2, color=theme.series(0)),
        marker=dict(size=9, line=dict(width=2, color=theme.surface)),
        hovertemplate="Predicted %{x:.2f}<br>Observed %{y:.2f}<extra></extra>"))
    fig.update_layout(title="Calibration", xaxis_title="Predicted churn probability",
                      yaxis_title="Observed churn rate", height=360, hovermode="closest")
    return fig


def concentration_curve(pareto: pd.DataFrame, theme: Theme) -> go.Figure:
    """How concentrated is revenue in the top customers?"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0] + list(pareto["top_customer_pct"]), y=[0] + list(pareto["revenue_share"]),
        name="Customers", mode="lines+markers",
        line=dict(width=2, color=theme.series(0)),
        marker=dict(size=8, line=dict(width=2, color=theme.surface)),
        hovertemplate="Top %{x:.0%} of customers<br>%{y:.1%} of revenue<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name="If everyone spent the same", mode="lines",
        line=dict(width=1, dash="dash", color=theme.ink["secondary"]), hoverinfo="skip"))
    fig.update_layout(title="Revenue concentration", xaxis_title="Share of customers",
                      yaxis_title="Share of revenue", height=360, hovermode="closest")
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    return fig


# --------------------------------------------------------------------------
# Matrix
# --------------------------------------------------------------------------
def retention_heatmap(matrix: pd.DataFrame, theme: Theme, max_months: int = 12) -> go.Figure:
    """Cohort retention triangle - magnitude, so one hue light to dark."""
    m = matrix.copy()
    if "cohort" in m.columns:
        m = m.set_index("cohort")
    cols = [c for c in m.columns if str(c).isdigit() and int(c) <= max_months]
    m = m[cols]
    fig = go.Figure(go.Heatmap(
        z=m.to_numpy() * 100, x=[f"M{c}" for c in cols], y=m.index.astype(str),
        colorscale=[[i / (len(theme.sequential) - 1), c]
                    for i, c in enumerate(theme.sequential)],
        hovertemplate="Cohort %{y}<br>%{x}: %{z:.1f}% retained<extra></extra>",
        colorbar=dict(title="% retained", thickness=12, outlinewidth=0),
        xgap=2, ygap=2))
    fig.update_layout(title="Cohort retention", height=520,
                      xaxis_title="Months since sign-up", yaxis_title="Sign-up cohort",
                      hovermode="closest")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


# --------------------------------------------------------------------------
# Anomalies
# --------------------------------------------------------------------------
def anomaly_chart(scored: pd.DataFrame, theme: Theme, store_id: str,
                  value_col: str = "transactions") -> go.Figure:
    """One store's series, its expected level, and the days that broke it."""
    d = scored[scored["store_id"] == store_id].sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d["date"], y=d[value_col], name="Actual", mode="lines",
        line=dict(width=1.4, color=theme.series(0)),
        hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<extra>Actual</extra>"))
    fig.add_trace(go.Scatter(
        x=d["date"], y=d["expected"], name="Expected", mode="lines",
        line=dict(width=1.6, color=theme.series(1), dash="dot"),
        hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<extra>Expected</extra>"))

    for direction, label, colour in (("dip", "Dip (incident)", theme.status["critical"]),
                                     ("spike", "Spike (incident)", theme.status["warning"])):
        hits = d[d["anomaly_direction"] == direction]
        if hits.empty:
            continue
        fig.add_trace(go.Scatter(
            x=hits["date"], y=hits[value_col], name=label, mode="markers",
            marker=dict(size=11, color=colour, symbol="circle",
                        line=dict(width=2, color=theme.surface)),
            hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<br>" + label + "<extra></extra>"))

    fig.update_layout(title=f"Store {store_id}: detected incidents",
                      yaxis_title=value_col.replace("_", " ").title(), height=400)
    return fig


def quality_chart(report: pd.DataFrame, theme: Theme) -> go.Figure:
    """Failing checks by how many rows they caught."""
    failed = report[~report["passed"]].copy()
    if failed.empty:
        fig = go.Figure()
        fig.update_layout(title="Every quality check passed", height=300)
        return fig
    failed["label"] = failed["table"] + " - " + failed["check"]
    failed = failed.sort_values("rows_failed")
    colours = [theme.status["critical"] if s == "critical" else theme.status["warning"]
               for s in failed["severity"]]
    fig = go.Figure(go.Bar(
        x=failed["rows_failed"], y=failed["label"], orientation="h",
        marker=dict(color=colours, line=dict(width=2, color=theme.surface)),
        customdata=failed["severity"],
        hovertemplate="%{y}<br>%{x:,} rows (%{customdata})<extra></extra>"))
    fig.update_layout(title="Failing data-quality checks", xaxis_title="Rows caught",
                      height=360, bargap=0.28, hovermode="closest")
    return fig


def basket_rules_scatter(rules: pd.DataFrame, theme: Theme, top_n: int = 60) -> go.Figure:
    """Support vs confidence, with lift as size - one hue, magnitude by area."""
    d = rules.head(top_n).copy()
    label = d["antecedent_names"] + " -> " + d["consequent_names"] \
        if "antecedent_names" in d else d.index.astype(str)
    fig = go.Figure(go.Scatter(
        x=d["support"], y=d["confidence"], mode="markers", text=label,
        marker=dict(size=np.clip(d["lift"], 1, 40), sizemode="area",
                    sizeref=2.0 * max(np.clip(d["lift"], 1, 40)) / (32 ** 2), sizemin=6,
                    color=theme.series(0), opacity=0.75,
                    line=dict(width=2, color=theme.surface)),
        hovertemplate="%{text}<br>support %{x:.4f}<br>confidence %{y:.2f}"
                      "<br>lift %{marker.size:.1f}x<extra></extra>"))
    fig.update_layout(title="Association rules (bubble size = lift)",
                      xaxis_title="Support (share of baskets)",
                      yaxis_title="Confidence", height=420, hovermode="closest")
    return fig
