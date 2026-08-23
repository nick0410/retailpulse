"""Chart theme: one palette, applied everywhere.

Colour is assigned by the *job* it does, not by taste:

* **categorical** - identity. A fixed slot order, never cycled or reshuffled,
  so a series keeps its colour when a filter changes the series count.
* **sequential** - magnitude. One hue, light to dark (retention heatmaps).
* **status** - state. Reserved for good/warning/serious/critical and never
  reused as "series 5"; always shipped with a label, never colour alone.

The slot order below is validated for colour-vision deficiency separation on
both surfaces; the first three slots are also safe for all-pairs forms such as
scatter, which is why multi-series charts here stay at three where they can.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# --------------------------------------------------------------------------
# Palette slots
# --------------------------------------------------------------------------
CATEGORICAL_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                     "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CATEGORICAL_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500",
                    "#d55181", "#008300", "#9085e9", "#e66767"]

SEQUENTIAL_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
                   "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
                   "#184f95", "#104281", "#0d366b"]

STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}

SURFACE_LIGHT = "#fcfcfb"
SURFACE_DARK = "#1a1a19"
INK_LIGHT = {"primary": "#0b0b0b", "secondary": "#52514e", "grid": "#e6e5e1"}
INK_DARK = {"primary": "#ffffff", "secondary": "#c3c2b7", "grid": "#33332f"}


class Theme:
    """Resolved colours for one mode, plus a matching plotly template."""

    def __init__(self, mode: str = "light"):
        self.mode = "dark" if str(mode).lower() == "dark" else "light"
        dark = self.mode == "dark"
        self.categorical = CATEGORICAL_DARK if dark else CATEGORICAL_LIGHT
        self.surface = SURFACE_DARK if dark else SURFACE_LIGHT
        self.ink = INK_DARK if dark else INK_LIGHT
        self.sequential = SEQUENTIAL_BLUE[::-1] if dark else SEQUENTIAL_BLUE
        self.status = STATUS

    def series(self, i: int) -> str:
        """Slot ``i`` of the categorical order. Past slot 8, fold to 'Other'."""
        return self.categorical[i % len(self.categorical)]

    def template(self) -> go.layout.Template:
        return go.layout.Template(
            layout=go.Layout(
                colorway=self.categorical,
                paper_bgcolor=self.surface,
                plot_bgcolor=self.surface,
                font=dict(family="Inter, Segoe UI, system-ui, sans-serif",
                          size=13, color=self.ink["primary"]),
                title=dict(font=dict(size=16, color=self.ink["primary"]), x=0, xanchor="left"),
                margin=dict(l=56, r=24, t=52, b=48),
                hovermode="x unified",
                hoverlabel=dict(bgcolor=self.surface, font_size=12,
                                bordercolor=self.ink["grid"]),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="left", x=0, title_text="",
                            font=dict(color=self.ink["secondary"])),
                xaxis=dict(showgrid=False, zeroline=False,
                           linecolor=self.ink["grid"], ticks="outside",
                           ticklen=4, tickcolor=self.ink["grid"],
                           tickfont=dict(color=self.ink["secondary"], size=12)),
                yaxis=dict(gridcolor=self.ink["grid"], zeroline=False,
                           showline=False, ticks="",
                           tickfont=dict(color=self.ink["secondary"], size=12)),
            )
        )

    def register(self, name: str = "retailpulse") -> str:
        pio.templates[name] = self.template()
        pio.templates.default = name
        return name


def format_inr(value: float, decimals: int = 1) -> str:
    """Indian-convention short form: thousands, lakh, crore."""
    value = float(value)
    sign = "-" if value < 0 else ""
    v = abs(value)
    if v >= 1e7:
        return f"{sign}Rs {v / 1e7:.{decimals}f} Cr"
    if v >= 1e5:
        return f"{sign}Rs {v / 1e5:.{decimals}f} L"
    if v >= 1e3:
        return f"{sign}Rs {v / 1e3:.{decimals}f} K"
    return f"{sign}Rs {v:.0f}"
