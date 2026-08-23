"""Deterministic calendar effects for the retail demand simulator.

The point of this module is that every seasonal pattern the forecaster later
"discovers" is planted here on purpose, in a form a human can read.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Major Indian retail festivals (dates that actually moved in the real calendar).
FESTIVALS: dict[str, list[str]] = {
    "diwali": ["2022-10-24", "2023-11-12", "2024-10-31"],
    "holi": ["2022-03-18", "2023-03-08", "2024-03-25"],
    "dussehra": ["2022-10-05", "2023-10-24", "2024-10-12"],
    "eid": ["2022-05-03", "2023-04-22", "2024-04-11"],
}

# Fixed-date holidays (month, day).
FIXED_HOLIDAYS: dict[str, tuple[int, int]] = {
    "new_year": (1, 1),
    "republic_day": (1, 26),
    "independence_day": (8, 15),
    "christmas": (12, 25),
}

DAY_OF_WEEK_FACTOR = np.array([0.88, 0.84, 0.90, 0.96, 1.12, 1.38, 1.32])
"""Monday..Sunday footfall multipliers - weekends carry the week."""


def festival_dates() -> pd.DatetimeIndex:
    flat = [d for dates in FESTIVALS.values() for d in dates]
    return pd.DatetimeIndex(sorted(pd.to_datetime(flat)))


def build_calendar(start: str, end: str) -> pd.DataFrame:
    """Return one row per date with every demand multiplier broken out.

    Columns are additive-in-logs factors so the total effect is a simple
    product, which keeps the story explainable: trend x weekday x season x
    festival x payday.
    """
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    cal = pd.DataFrame({"date": dates})

    # 1. Long-run growth: ~22% over the full window, compounding smoothly.
    t = np.arange(n) / max(n - 1, 1)
    cal["trend_factor"] = 1.0 + 0.22 * t

    # 2. Day-of-week rhythm.
    cal["dow"] = cal["date"].dt.dayofweek
    cal["dow_factor"] = DAY_OF_WEEK_FACTOR[cal["dow"].to_numpy()]

    # 3. Smooth annual season (summer dip, winter/festive lift).
    doy = cal["date"].dt.dayofyear.to_numpy()
    cal["season_factor"] = 1.0 + 0.13 * np.sin(2 * np.pi * (doy - 80) / 365.25)

    # 4. Festival build-up: demand ramps for ~12 days then spikes on the day.
    festival = np.ones(n)
    date_pos = {d: i for i, d in enumerate(dates)}
    for fest_date in festival_dates():
        if fest_date not in date_pos:
            continue
        idx = date_pos[fest_date]
        for lead in range(0, 13):
            j = idx - lead
            if j < 0:
                continue
            festival[j] = max(festival[j], 1.0 + 0.95 * np.exp(-lead / 5.0))
        for lag in range(1, 4):  # post-festival hangover
            j = idx + lag
            if j < n:
                festival[j] = min(festival[j], 0.80 + 0.05 * lag)
    for month, day in FIXED_HOLIDAYS.values():
        mask = (cal["date"].dt.month == month) & (cal["date"].dt.day == day)
        festival[mask.to_numpy()] *= 1.35
    cal["festival_factor"] = festival

    # 5. Payday effect: the first week of the month is richer than the last.
    dom = cal["date"].dt.day.to_numpy()
    cal["payday_factor"] = np.where(dom <= 5, 1.18, np.where(dom >= 26, 0.92, 1.0))

    cal["demand_factor"] = (
        cal["trend_factor"]
        * cal["dow_factor"]
        * cal["season_factor"]
        * cal["festival_factor"]
        * cal["payday_factor"]
    )
    cal["is_festival_window"] = cal["festival_factor"] > 1.05
    return cal
