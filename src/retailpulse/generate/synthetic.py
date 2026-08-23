"""Synthetic retail data generator with *planted* ground truth.

Why not just download a public CSV?  Because every model in this project is
validated against structure we deliberately buried in the data:

* customer purchase timing is drawn from the exact BG/NBD process the CLV
  model later tries to recover;
* transaction values come from the Gamma-Gamma process the spend model fits;
* a fixed list of product pairs is co-purchased far more often than chance,
  which the market-basket miner must rediscover;
* operational anomalies (stockouts, viral spikes) are injected at known
  store/date coordinates, which the anomaly detector must flag;
* data-quality defects (duplicates, nulls, impossible values) are injected at
  known rates, which the validation engine must count.

The generator writes plain CSVs to data/raw plus a ground_truth.json that the
test-suite asserts against.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from ..config import CONFIG, RAW_DIR, SimulationConfig
from .calendar_effects import DAY_OF_WEEK_FACTOR, build_calendar

# --------------------------------------------------------------------------
# Reference/master data
# --------------------------------------------------------------------------
CATEGORIES: dict[str, tuple[float, float]] = {
    # category -> (base price mean, price spread)
    "Groceries": (240.0, 90.0),
    "Beverages": (180.0, 70.0),
    "Personal Care": (320.0, 140.0),
    "Home Care": (280.0, 110.0),
    "Snacks": (95.0, 45.0),
    "Dairy": (120.0, 50.0),
    "Electronics Accessories": (1450.0, 800.0),
    "Apparel": (1150.0, 620.0),
}

CITIES = [
    ("Mumbai", "West"), ("Pune", "West"), ("Delhi", "North"), ("Gurugram", "North"),
    ("Bengaluru", "South"), ("Chennai", "South"), ("Hyderabad", "South"),
    ("Kolkata", "East"), ("Bhubaneswar", "East"), ("Ahmedabad", "West"),
    ("Jaipur", "North"), ("Lucknow", "North"),
]

STORE_FORMATS = ["Hypermarket", "Supermarket", "Express"]
CHANNELS = ["in_store", "app", "web"]
AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55+"]

# Product pairs that are co-purchased on purpose. The basket miner has to find
# these without being told they exist.
AFFINITY_RULES: list[tuple[str, str, float]] = [
    ("Filter Coffee Powder", "Full Cream Milk", 0.62),
    ("Instant Noodles", "Tomato Ketchup", 0.55),
    ("Baby Diapers", "Baby Wipes", 0.68),
    ("Shampoo", "Hair Conditioner", 0.58),
    ("Bread Loaf", "Butter 500g", 0.60),
    ("Dish Wash Gel", "Scrub Pad", 0.57),
    ("Phone Charger", "USB Cable", 0.52),
    ("Green Tea", "Honey", 0.49),
]

# Named products that must exist so the affinity rules have something to bind
# to. The rest of the catalogue is generated procedurally.
SEED_PRODUCTS: list[tuple[str, str]] = [
    ("Filter Coffee Powder", "Beverages"), ("Full Cream Milk", "Dairy"),
    ("Instant Noodles", "Snacks"), ("Tomato Ketchup", "Groceries"),
    ("Baby Diapers", "Personal Care"), ("Baby Wipes", "Personal Care"),
    ("Shampoo", "Personal Care"), ("Hair Conditioner", "Personal Care"),
    ("Bread Loaf", "Groceries"), ("Butter 500g", "Dairy"),
    ("Dish Wash Gel", "Home Care"), ("Scrub Pad", "Home Care"),
    ("Phone Charger", "Electronics Accessories"), ("USB Cable", "Electronics Accessories"),
    ("Green Tea", "Beverages"), ("Honey", "Groceries"),
]

GENERIC_NOUNS = [
    "Atta", "Basmati Rice", "Toor Dal", "Sunflower Oil", "Sugar", "Salt",
    "Masala Mix", "Biscuits", "Chips", "Cola", "Orange Juice", "Curd",
    "Paneer", "Cheese Slices", "Face Wash", "Toothpaste", "Soap Bar",
    "Hand Wash", "Floor Cleaner", "Detergent Powder", "Toilet Cleaner",
    "Air Freshener", "Earphones", "Power Bank", "Memory Card", "Mouse",
    "T-Shirt", "Denim Jeans", "Kurta", "Socks Pack", "Bath Towel", "Bedsheet",
]

BRANDS = ["Everyday", "GoldLeaf", "Nimbus", "Saffron", "UrbanBasket", "Kwality", "Zenith"]


# --------------------------------------------------------------------------
# Master data builders
# --------------------------------------------------------------------------
def _build_stores(rng: np.random.Generator, cfg: SimulationConfig) -> pd.DataFrame:
    n = cfg.n_stores
    cities = [CITIES[i % len(CITIES)] for i in range(n)]
    rows = []
    for i, (city, region) in enumerate(cities):
        fmt = STORE_FORMATS[i % len(STORE_FORMATS)]
        format_pull = {"Hypermarket": 1.6, "Supermarket": 1.0, "Express": 0.6}[fmt]
        opened = pd.Timestamp(cfg.start_date) - pd.Timedelta(days=int(rng.integers(400, 2600)))
        rows.append(
            {
                "store_id": f"ST{i + 1:03d}",
                "store_name": f"RetailPulse {city} {fmt}",
                "city": city,
                "region": region,
                "store_format": fmt,
                "opened_date": opened.date(),
                "floor_area_sqft": int(rng.integers(1800, 42000)),
                # Base daily walk-in traffic multiplier for this store.
                "traffic_index": float(np.round(rng.uniform(0.75, 1.45) * format_pull, 3)),
            }
        )
    return pd.DataFrame(rows)


def _build_products(rng: np.random.Generator, cfg: SimulationConfig) -> pd.DataFrame:
    names: list[tuple[str, str]] = list(SEED_PRODUCTS)
    cat_names = list(CATEGORIES)
    i = 0
    while len(names) < cfg.n_products:
        noun = GENERIC_NOUNS[i % len(GENERIC_NOUNS)]
        variant = i // len(GENERIC_NOUNS)
        suffix = ["", "Pro", "Value Pack", "Mini"][variant % 4]
        label = f"{noun} {suffix}".strip()
        cat = cat_names[i % len(cat_names)]
        if (label, cat) not in names:
            names.append((label, cat))
        i += 1

    rows = []
    for i, (name, cat) in enumerate(names[: cfg.n_products]):
        mean, spread = CATEGORIES[cat]
        base_price = float(np.round(max(19.0, rng.normal(mean, spread)), 2))
        margin = float(rng.uniform(0.18, 0.42))
        rows.append(
            {
                "product_id": f"P{i + 1:04d}",
                "sku": f"SKU-{cat[:3].upper()}-{i + 1:04d}",
                "product_name": name,
                "category": cat,
                "brand": BRANDS[i % len(BRANDS)],
                "base_price": base_price,
                "unit_cost": float(np.round(base_price * (1 - margin), 2)),
                # Pareto popularity: a few heroes carry most of the volume.
                "popularity": float(np.round(rng.pareto(1.6) + 0.25, 4)),
            }
        )
    df = pd.DataFrame(rows)
    df["popularity"] = df["popularity"] / df["popularity"].sum()
    return df


def _build_customers(rng: np.random.Generator, cfg: SimulationConfig) -> pd.DataFrame:
    start, end = pd.Timestamp(cfg.start_date), pd.Timestamp(cfg.end_date)
    n = cfg.n_customers

    # Loyalty sign-ups grow month over month, which gives cohort analysis
    # something real to show.
    months = pd.date_range(start, end - pd.Timedelta(days=45), freq="MS")
    weights = np.linspace(1.0, 2.1, len(months))
    weights = weights / weights.sum()
    month_choice = rng.choice(len(months), size=n, p=weights)
    day_offset = rng.integers(0, 28, size=n)
    signup = pd.DatetimeIndex(months[month_choice]) + pd.to_timedelta(day_offset, unit="D")
    cutoff = end - pd.Timedelta(days=15)
    signup = pd.DatetimeIndex(np.minimum(signup.values, cutoff.to_datetime64()))

    city_idx = rng.integers(0, len(CITIES), size=n)
    cities = [CITIES[i][0] for i in city_idx]
    regions = [CITIES[i][1] for i in city_idx]

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i + 1:06d}" for i in range(n)],
            "signup_date": signup.date,
            "city": cities,
            "region": regions,
            "age_band": rng.choice(AGE_BANDS, size=n, p=[0.18, 0.31, 0.24, 0.16, 0.11]),
            "gender": rng.choice(["F", "M", "Other"], size=n, p=[0.48, 0.49, 0.03]),
            "preferred_channel": rng.choice(CHANNELS, size=n, p=[0.55, 0.30, 0.15]),
            "loyalty_tier": rng.choice(["Bronze", "Silver", "Gold", "Platinum"],
                                       size=n, p=[0.50, 0.28, 0.16, 0.06]),
            "email": [f"customer{i + 1:06d}@example.com" for i in range(n)],
        }
    )
    return df


def _build_promotions(rng: np.random.Generator, cfg: SimulationConfig,
                      products: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Multi-day promo campaigns on random products, denser near festivals."""
    dates = calendar["date"].to_numpy()
    festive = calendar["is_festival_window"].to_numpy()
    n_campaigns = max(1, int(cfg.promo_rate * len(dates) * cfg.n_products / 55))
    rows = []
    for c in range(n_campaigns):
        pid = products["product_id"].iloc[int(rng.integers(0, len(products)))]
        # Most campaigns are deliberately aimed at festival windows.
        if rng.random() < 0.6 and festive.any():
            candidates = np.flatnonzero(festive)
        else:
            candidates = np.arange(len(dates))
        start_i = int(rng.choice(candidates))
        length = int(rng.integers(3, 15))
        end_i = min(start_i + length, len(dates) - 1)
        rows.append(
            {
                "promo_id": f"PR{c + 1:05d}",
                "product_id": pid,
                "start_date": pd.Timestamp(dates[start_i]).date(),
                "end_date": pd.Timestamp(dates[end_i]).date(),
                "discount_pct": float(rng.choice([0.05, 0.10, 0.15, 0.20, 0.25, 0.33, 0.40])),
                "promo_type": str(rng.choice(["Price Off", "BOGO", "Bundle", "Festive Deal"])),
            }
        )
    return pd.DataFrame(rows)


def _promo_lookup(promotions: pd.DataFrame) -> pd.DataFrame:
    """Explode campaigns into a (product_id, date) -> discount map."""
    if promotions.empty:
        return pd.DataFrame(columns=["product_id", "date", "discount_pct"])
    frames = []
    for row in promotions.itertuples(index=False):
        days = pd.date_range(row.start_date, row.end_date, freq="D")
        frames.append(pd.DataFrame({"product_id": row.product_id, "date": days,
                                    "discount_pct": row.discount_pct}))
    out = pd.concat(frames, ignore_index=True)
    # Overlapping campaigns: the shopper always gets the best price.
    return out.groupby(["product_id", "date"], as_index=False)["discount_pct"].max()


# --------------------------------------------------------------------------
# Customer transaction timing: the BG/NBD generative process
# --------------------------------------------------------------------------
def _simulate_customer_transactions(rng: np.random.Generator, cfg: SimulationConfig,
                                    customers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw repeat-purchase timing straight from BG/NBD.

    lambda_i ~ Gamma(r, 1/alpha)  transactions per week while the customer is alive
    p_i      ~ Beta(a, b)         probability of churning right after a purchase

    Every customer makes a trial purchase on sign-up day; after each repeat
    purchase they flip a p_i coin and may leave forever.
    """
    n = len(customers)
    lam = np.clip(rng.gamma(shape=cfg.bgnbd_r, scale=1.0 / cfg.bgnbd_alpha, size=n), 1e-4, None)
    p_drop = rng.beta(cfg.bgnbd_a, cfg.bgnbd_b, size=n)

    end = pd.Timestamp(cfg.end_date)
    signup = pd.to_datetime(customers["signup_date"])
    horizon_weeks = ((end - signup).dt.days / 7.0).to_numpy()

    cust_ids: list[str] = []
    offsets_weeks: list[float] = []
    alive_flags = np.zeros(n, dtype=bool)
    customer_ids = customers["customer_id"].to_numpy()

    for i in range(n):
        cid = customer_ids[i]
        horizon = horizon_weeks[i]
        cust_ids.append(cid)
        offsets_weeks.append(0.0)  # trial purchase on sign-up day
        t = 0.0
        alive = True
        while True:
            t += rng.exponential(1.0 / lam[i])
            if t > horizon:
                break
            cust_ids.append(cid)
            offsets_weeks.append(t)
            if rng.random() < p_drop[i]:
                alive = False
                break
        alive_flags[i] = alive

    signup_map = dict(zip(customer_ids, signup.to_numpy()))
    base = np.array([signup_map[c] for c in cust_ids], dtype="datetime64[ns]")
    dates = pd.DatetimeIndex(base) + pd.to_timedelta(np.array(offsets_weeks) * 7.0, unit="D")
    txns = pd.DataFrame({"customer_id": cust_ids, "date": dates.normalize()})
    txns = txns[txns["date"] <= end].reset_index(drop=True)

    truth = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "true_lambda_weekly": lam,
            "true_dropout_p": p_drop,
            "true_alive_at_end": alive_flags,
        }
    )
    return txns, truth


def _apply_weekday_shift(rng: np.random.Generator, dates: pd.Series) -> pd.Series:
    """Nudge loyalty purchases towards weekends without changing their count.

    Each transaction hops up to +/- 3 days, sampled in proportion to that
    weekday's footfall. Per-customer counts are preserved, so BG/NBD parameter
    recovery stays valid while the daily series gains a weekly rhythm.
    """
    offsets = np.arange(-3, 4)
    d = pd.to_datetime(dates)
    dow = d.dt.dayofweek.to_numpy()[:, None]
    cand_dow = (dow + offsets[None, :]) % 7
    weights = DAY_OF_WEEK_FACTOR[cand_dow]
    weights = weights / weights.sum(axis=1, keepdims=True)
    cum = weights.cumsum(axis=1)
    draw = rng.random(len(d))[:, None]
    pick = (draw > cum).sum(axis=1).clip(0, len(offsets) - 1)
    return d + pd.to_timedelta(offsets[pick], unit="D")


# --------------------------------------------------------------------------
# Walk-in traffic: the seasonal, forecastable stream
# --------------------------------------------------------------------------
def _build_anomalies(rng: np.random.Generator, cfg: SimulationConfig,
                     stores: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Inject known operational incidents to be re-discovered later."""
    dates = calendar["date"].to_numpy()
    store_ids = stores["store_id"].to_numpy()
    rows = []
    for i in range(cfg.n_anomalies):
        store_id = store_ids[int(rng.integers(0, len(store_ids)))]
        start_i = int(rng.integers(30, len(dates) - 10))
        if rng.random() < 0.55:
            kind, mult, length = "stockout", float(rng.uniform(0.18, 0.42)), int(rng.integers(2, 6))
        else:
            kind, mult, length = "demand_spike", float(rng.uniform(2.4, 4.2)), int(rng.integers(1, 4))
        end_i = min(start_i + length - 1, len(dates) - 1)
        rows.append(
            {
                "anomaly_id": f"AN{i + 1:03d}",
                "store_id": store_id,
                "start_date": pd.Timestamp(dates[start_i]).date(),
                "end_date": pd.Timestamp(dates[end_i]).date(),
                "anomaly_type": kind,
                "impact_multiplier": round(mult, 3),
            }
        )
    return pd.DataFrame(rows)


def _apply_anomalies_to_loyalty(rng: np.random.Generator, loyalty: pd.DataFrame,
                                anomalies: pd.DataFrame) -> pd.DataFrame:
    """Drop loyalty trips that fall inside a store's stockout/closure window."""
    if loyalty.empty or anomalies.empty:
        return loyalty
    keep_prob = np.ones(len(loyalty))
    dates = pd.to_datetime(loyalty["date"]).to_numpy()
    stores = loyalty["store_id"].to_numpy()
    for row in anomalies.itertuples(index=False):
        if row.impact_multiplier >= 1.0:
            continue  # spikes do not suppress existing demand
        window = ((stores == row.store_id)
                  & (dates >= np.datetime64(pd.Timestamp(row.start_date), "ns"))
                  & (dates <= np.datetime64(pd.Timestamp(row.end_date), "ns")))
        keep_prob[window] = np.minimum(keep_prob[window], row.impact_multiplier)
    return loyalty.loc[rng.random(len(loyalty)) < keep_prob].copy()


def _simulate_walkin_transactions(rng: np.random.Generator, stores: pd.DataFrame,
                                  calendar: pd.DataFrame, anomalies: pd.DataFrame) -> pd.DataFrame:
    """Anonymous footfall per store-day: Poisson around the planted demand curve."""
    dates = calendar["date"].to_numpy()
    factor = calendar["demand_factor"].to_numpy()

    anomaly_map: dict[tuple[str, np.datetime64], float] = {}
    for row in anomalies.itertuples(index=False):
        for d in pd.date_range(row.start_date, row.end_date, freq="D"):
            anomaly_map[(row.store_id, np.datetime64(d, "ns"))] = row.impact_multiplier

    frames = []
    for store in stores.itertuples(index=False):
        base = 8.0 * store.traffic_index
        mult = np.array([anomaly_map.get((store.store_id, d), 1.0) for d in dates])
        counts = rng.poisson(base * factor * mult)
        frames.append(pd.DataFrame({"store_id": store.store_id, "date": np.repeat(dates, counts)}))
    walkins = pd.concat(frames, ignore_index=True)
    walkins["customer_id"] = pd.NA
    return walkins


# --------------------------------------------------------------------------
# Basket construction
# --------------------------------------------------------------------------
def _build_baskets(rng: np.random.Generator, cfg: SimulationConfig,
                   headers: pd.DataFrame, products: pd.DataFrame,
                   promo_map: pd.DataFrame, cust_value: dict[str, float]) -> pd.DataFrame:
    """Turn transaction headers into line items, then plant co-purchase pairs."""
    n_txn = len(headers)
    basket_size = 1 + rng.poisson(1.7, size=n_txn)
    total_lines = int(basket_size.sum())

    prod_idx = rng.choice(len(products), size=total_lines, p=products["popularity"].to_numpy())
    items = pd.DataFrame(
        {
            "transaction_id": np.repeat(headers["transaction_id"].to_numpy(), basket_size),
            "date": np.repeat(headers["date"].to_numpy(), basket_size),
            "product_id": products["product_id"].to_numpy()[prod_idx],
        }
    )

    # --- planted affinities -------------------------------------------------
    name_to_id = dict(zip(products["product_name"], products["product_id"]))
    extra_frames = []
    for lhs, rhs, prob in AFFINITY_RULES:
        if lhs not in name_to_id or rhs not in name_to_id:
            continue
        lhs_id, rhs_id = name_to_id[lhs], name_to_id[rhs]
        carriers = (items.loc[items["product_id"] == lhs_id, ["transaction_id", "date"]]
                    .drop_duplicates("transaction_id"))
        if carriers.empty:
            continue
        add = carriers.loc[rng.random(len(carriers)) < prob].copy()
        add["product_id"] = rhs_id
        extra_frames.append(add)
    if extra_frames:
        items = pd.concat([items] + extra_frames, ignore_index=True)

    # One line per (transaction, product): repeats collapse into quantity.
    items = (items.groupby(["transaction_id", "date", "product_id"], as_index=False)
             .size().rename(columns={"size": "base_units"}))

    # --- pricing and promo elasticity ---------------------------------------
    items = items.merge(products[["product_id", "base_price", "unit_cost", "category"]],
                        on="product_id", how="left")
    items["date"] = pd.to_datetime(items["date"])
    if promo_map.empty:
        items["discount_pct"] = 0.0
    else:
        items = items.merge(promo_map, on=["product_id", "date"], how="left")
        items["discount_pct"] = items["discount_pct"].fillna(0.0)

    items["unit_price"] = np.round(items["base_price"] * (1 - items["discount_pct"]), 2)

    # Demand responds to price: q ~ (p/p0)^elasticity with elasticity < 0.
    price_ratio = (1 - items["discount_pct"]).clip(lower=0.05)
    lift = price_ratio.to_numpy() ** cfg.price_elasticity
    lam_qty = np.clip((items["base_units"].to_numpy() - 1) * 0.7 + 0.45 * lift, 0.01, 12.0)
    items["quantity"] = (1 + rng.poisson(lam_qty)).astype(int)

    # --- scale the basket towards the customer's Gamma-Gamma spend level -----
    items["line_amount"] = items["quantity"] * items["unit_price"]
    target = headers[["transaction_id", "customer_id"]].copy()
    target["target_value"] = [
        cust_value.get(c, np.nan) if isinstance(c, str) else np.nan
        for c in target["customer_id"]
    ]
    items = items.merge(target[["transaction_id", "target_value"]], on="transaction_id", how="left")
    basket_total = items.groupby("transaction_id")["line_amount"].transform("sum").replace(0, np.nan)
    scale = (items["target_value"] / basket_total).fillna(1.0).clip(0.35, 3.0)
    items["quantity"] = np.maximum(1, np.round(items["quantity"] * scale)).astype(int)
    items["line_amount"] = np.round(items["quantity"] * items["unit_price"], 2)
    items["line_cost"] = np.round(items["quantity"] * items["unit_cost"], 2)
    items["gross_margin"] = np.round(items["line_amount"] - items["line_cost"], 2)

    items = items.drop(columns=["base_units", "target_value", "base_price", "unit_cost"])
    items.insert(0, "line_id", [f"L{i + 1:08d}" for i in range(len(items))])
    return items


# --------------------------------------------------------------------------
# Deliberate data-quality defects
# --------------------------------------------------------------------------
def _inject_quality_issues(rng: np.random.Generator, customers: pd.DataFrame,
                           headers: pd.DataFrame, items: pd.DataFrame):
    """Dirty the data on purpose so the validation engine has real work to do."""
    truth: dict[str, int] = {}

    # 1. Missing customer city (completeness check).
    n_null_city = int(0.021 * len(customers))
    idx = rng.choice(customers.index.to_numpy(), size=n_null_city, replace=False)
    customers.loc[idx, "city"] = None
    truth["customers_missing_city"] = n_null_city

    # 2. Malformed emails (validity check).
    n_bad_email = int(0.013 * len(customers))
    idx = rng.choice(customers.index.to_numpy(), size=n_bad_email, replace=False)
    customers.loc[idx, "email"] = customers.loc[idx, "email"].str.replace("@example.com", "", regex=False)
    truth["customers_invalid_email"] = n_bad_email

    # 3. Duplicated transaction rows (classic double-ingest bug).
    n_dupe = int(0.004 * len(headers))
    dupe_idx = rng.choice(headers.index.to_numpy(), size=n_dupe, replace=False)
    headers = pd.concat([headers, headers.loc[dupe_idx]], ignore_index=True)
    truth["duplicate_transaction_rows"] = n_dupe

    # 4. Negative quantities (returns booked against the sales table).
    n_neg = int(0.0016 * len(items))
    idx = rng.choice(items.index.to_numpy(), size=n_neg, replace=False)
    items.loc[idx, "quantity"] = -items.loc[idx, "quantity"].abs()
    items.loc[idx, "line_amount"] = -items.loc[idx, "line_amount"].abs()
    truth["negative_quantity_rows"] = n_neg

    # 5. Impossible unit prices (feed/parsing failure).
    n_price = int(0.0009 * len(items))
    idx = rng.choice(items.index.to_numpy(), size=n_price, replace=False)
    items.loc[idx, "unit_price"] = 0.0
    truth["zero_price_rows"] = n_price

    # 6. Orphan line items pointing at transactions that do not exist.
    n_orphan = 45
    orphan = items.sample(n=n_orphan, random_state=7).copy()
    orphan["line_id"] = [f"LX{i:07d}" for i in range(n_orphan)]
    orphan["transaction_id"] = [f"T99{i:07d}" for i in range(n_orphan)]
    items = pd.concat([items, orphan], ignore_index=True)
    truth["orphan_line_items"] = n_orphan

    return customers, headers, items, truth


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def generate_dataset(cfg: SimulationConfig | None = None, write: bool = True) -> dict:
    """Build the whole synthetic universe and (optionally) write it to data/raw."""
    cfg = cfg or CONFIG.simulation
    rng = np.random.default_rng(cfg.seed)

    calendar = build_calendar(cfg.start_date, cfg.end_date)
    stores = _build_stores(rng, cfg)
    products = _build_products(rng, cfg)
    customers = _build_customers(rng, cfg)
    anomalies = _build_anomalies(rng, cfg, stores, calendar)
    promotions = _build_promotions(rng, cfg, products, calendar)
    promo_map = _promo_lookup(promotions)

    # --- transaction headers ------------------------------------------------
    loyalty_txns, cust_truth = _simulate_customer_transactions(rng, cfg, customers)
    loyalty_txns["date"] = _apply_weekday_shift(rng, loyalty_txns["date"])
    in_window = (
        (loyalty_txns["date"] >= pd.Timestamp(cfg.start_date))
        & (loyalty_txns["date"] <= pd.Timestamp(cfg.end_date))
    )
    loyalty_txns = loyalty_txns.loc[in_window].copy()

    # Loyalty shoppers stick to a home store most of the time.
    store_ids = stores["store_id"].to_numpy()
    home_store = dict(zip(customers["customer_id"],
                          store_ids[rng.integers(0, len(store_ids), size=len(customers))]))
    roam = rng.random(len(loyalty_txns)) < 0.15
    random_store = store_ids[rng.integers(0, len(store_ids), size=len(loyalty_txns))]
    loyalty_txns["store_id"] = np.where(
        roam, random_store, [home_store[c] for c in loyalty_txns["customer_id"]]
    )

    # A stockout or closure turns away loyalty shoppers exactly as it turns
    # away walk-ins, so the same incident is applied to both streams. Demand
    # *spikes* are footfall-driven and left to the walk-in stream alone.
    loyalty_txns = _apply_anomalies_to_loyalty(rng, loyalty_txns, anomalies)

    walkins = _simulate_walkin_transactions(rng, stores, calendar, anomalies)

    cols = ["customer_id", "store_id", "date"]
    headers = (pd.concat([loyalty_txns[cols], walkins[cols]], ignore_index=True)
               .sort_values("date", kind="mergesort")
               .reset_index(drop=True))
    headers.insert(0, "transaction_id", [f"T{i + 1:08d}" for i in range(len(headers))])
    headers["channel"] = np.where(
        headers["customer_id"].notna(),
        rng.choice(CHANNELS, size=len(headers), p=[0.55, 0.30, 0.15]),
        "in_store",
    )
    headers["payment_method"] = rng.choice(
        ["UPI", "Card", "Cash", "Wallet"], size=len(headers), p=[0.46, 0.27, 0.19, 0.08]
    )

    # --- Gamma-Gamma spend levels ------------------------------------------
    # nu_i ~ Gamma(q, scale=1/v) so that E[spend_i] = p / nu_i.
    nu = rng.gamma(shape=cfg.gg_q, scale=1.0 / cfg.gg_v, size=len(customers))
    cust_mean_value = np.clip(cfg.gg_p / np.clip(nu, 1e-9, None), 180.0, 25_000.0)
    cust_value = dict(zip(customers["customer_id"], cust_mean_value))
    cust_truth["true_mean_transaction_value"] = cust_mean_value

    items = _build_baskets(rng, cfg, headers, products, promo_map, cust_value)

    # --- dirty it up --------------------------------------------------------
    customers, headers, items, dq_truth = _inject_quality_issues(rng, customers, headers, items)

    headers["date"] = pd.to_datetime(headers["date"]).dt.date
    items["date"] = pd.to_datetime(items["date"]).dt.date

    ground_truth = {
        "simulation_config": asdict(cfg),
        "affinity_rules": [
            {"antecedent_name": a, "consequent_name": b, "planted_probability": p}
            for a, b, p in AFFINITY_RULES
        ],
        "data_quality_defects": dq_truth,
        "n_transactions": int(len(headers)),
        "n_line_items": int(len(items)),
        "n_anomaly_events": int(len(anomalies)),
    }

    tables = {
        "customers": customers,
        "products": products,
        "stores": stores,
        "promotions": promotions,
        "transactions": headers,
        "transaction_items": items,
        "anomaly_ground_truth": anomalies,
        "customer_ground_truth": cust_truth,
        "calendar": calendar,
    }

    if write:
        for name, df in tables.items():
            df.to_csv(RAW_DIR / f"{name}.csv", index=False)
        (RAW_DIR / "ground_truth.json").write_text(
            json.dumps(ground_truth, indent=2, default=str), encoding="utf-8"
        )

    tables["_ground_truth"] = ground_truth
    return tables
