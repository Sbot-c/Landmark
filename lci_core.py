"""
lci_core.py — shared, deterministic helpers for the Landmark Customer Intelligence Engine.

Used by:
  * notebooks/Landmark_Customer_Intelligence.ipynb  (all analysis, model training, artifact export)
  * app.py                                           (dashboard: reads artifacts; light, non-model calculations only)

Keeping data loading, feature engineering, financial aggregation and the epsilon-greedy simulator in ONE module
means the notebook and the dashboard can never disagree about a definition (e.g. what "recency" or
"recorded revenue" means). No model is trained in this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------------------------------
SEED = 42
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ART = ROOT / "artifacts"
MODELS = ART / "models"
METRICS = ART / "metrics"
VIS = ART / "visualisations"
for _d in (MODELS, METRICS, VIS):
    _d.mkdir(parents=True, exist_ok=True)

CORE_BANNERS = ["Home Centre", "Homebox", "Splash", "Max", "Babyshop", "Centrepoint", "Funcity"]
ALL_BANNERS = CORE_BANNERS + ["Viva"]
BANNER_CATEGORY = {
    "Home Centre": "Home", "Homebox": "Home", "Splash": "Fashion", "Max": "Fashion",
    "Babyshop": "Baby", "Centrepoint": "Department", "Funcity": "Entertainment", "Viva": "Grocery (Viva)",
}
BANNER_POSITION = {
    "Home Centre": "Premium home", "Homebox": "Value home", "Splash": "Premium / mid fashion",
    "Max": "Value fashion", "Babyshop": "Mid baby & kids", "Centrepoint": "Mixed department store",
    "Funcity": "Family entertainment", "Viva": "Value grocery (not on Shukran)",
}
CATEGORIES = ["Home", "Fashion", "Baby", "Department", "Entertainment"]

# Validated categorical palette (dataviz validator: adjacent-pair CVD dE >= 10.7, normal-vision >= 16.7,
# all >= 3:1 contrast on white).  Colour follows the ENTITY, never its rank.
BANNER_COLORS = {
    "Home Centre": "#2A6FB5", "Homebox": "#C0603F", "Splash": "#2A9D8F", "Max": "#B8860B",
    "Babyshop": "#8E5A9B", "Centrepoint": "#4F8A3A", "Funcity": "#D17A9A", "Viva": "#A0672E",
}
SERIES = ["#2A6FB5", "#C0603F", "#2A9D8F", "#B8860B", "#8E5A9B", "#4F8A3A", "#D17A9A", "#A0672E"]
UI = {
    "navy": "#14213D", "navy2": "#1F3A5F", "gold": "#B08D57", "ivory": "#F7F5F0", "card": "#FFFFFF",
    "slate": "#5C6670", "ink": "#0B1324", "muted": "#898781", "grid": "#E6E2D8", "teal": "#2A9D8F",
    "terra": "#C0603F",
}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

REQUIRED_SCHEMA = {
    "customers": ["customer_id", "age", "gender", "emirate", "family_size", "has_kids", "tenure_months",
                  "signup_date", "churned"],
    "transactions": ["transaction_id", "customer_id", "date", "banner", "category", "amount", "discount_used"],
    "viva_transactions": ["transaction_id", "viva_member_id", "date", "category", "amount"],
}


# --------------------------------------------------------------------------------------------------
# Loading & validation
# --------------------------------------------------------------------------------------------------
def find_csv(stem: str, folder: Path = DATA_DIR) -> Path:
    """Return the CSV for `stem`, accepting the original name or an uploaded copy such as `stem(1).csv`."""
    folder = Path(folder)
    patterns = [f"{stem}.csv", f"{stem}(*).csv", f"{stem} (*).csv"]
    hits = sorted({p for pat in patterns for p in folder.glob(pat)})
    if not hits:
        raise FileNotFoundError(
            f"No file for '{stem}' in {folder}. Expected '{stem}.csv' or '{stem}(1).csv'. "
            "Place the three supplied CSVs in the data/ folder.")
    return hits[0]


def md5(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def load_raw(folder: Path = DATA_DIR):
    """Load the three raw files exactly as supplied (no cleaning) and validate the schema."""
    paths = {k: find_csv(k, folder) for k in REQUIRED_SCHEMA}
    cust = pd.read_csv(paths["customers"], parse_dates=["signup_date"])
    tx = pd.read_csv(paths["transactions"], parse_dates=["date"])
    viva = pd.read_csv(paths["viva_transactions"], parse_dates=["date"])
    for name, df in zip(REQUIRED_SCHEMA, (cust, tx, viva)):
        missing = set(REQUIRED_SCHEMA[name]) - set(df.columns)
        if missing:
            raise ValueError(f"Schema error in {name}: missing columns {sorted(missing)}")
    # explicit, documented dtypes (pandas 3 infers the 'str' dtype for text columns)
    for df, cols in ((cust, ["customer_id", "gender", "emirate"]),
                     (tx, ["transaction_id", "customer_id", "banner", "category"]),
                     (viva, ["transaction_id", "category"])):
        for c in cols:
            df[c] = df[c].astype(str)
    viva["viva_member_id"] = viva["viva_member_id"].astype("string")
    return cust, tx, viva, paths


def audit_frame(df: pd.DataFrame, name: str, key: str) -> dict:
    """Per-file audit used in Stage 0."""
    num = df.select_dtypes("number")
    dates = df.select_dtypes("datetime")
    out = {
        "file": name, "rows": int(len(df)), "columns": int(df.shape[1]),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": {k: int(v) for k, v in df.isna().sum().items()},
        "missing_pct": {k: round(float(v) * 100, 2) for k, v in df.isna().mean().items()},
        "full_row_duplicates": int(df.duplicated().sum()),
        "key": key, "key_duplicates": int(df[key].duplicated(keep=False).sum()),
        "key_unique": bool(df[key].is_unique),
        "nunique": {k: int(v) for k, v in df.nunique().items()},
        "skew": {k: round(float(v), 3) for k, v in num.skew().items()},
        "date_range": {c: [str(dates[c].min().date()), str(dates[c].max().date())] for c in dates},
        "invalid_dates": {c: int(dates[c].isna().sum()) for c in dates},
    }
    return out


def iqr_outliers(s: pd.Series, k: float = 1.5) -> pd.Series:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return (s < q1 - k * iqr) | (s > q3 + k * iqr)


# --------------------------------------------------------------------------------------------------
# Unified customer view (one row per Shukran customer, as of a reference date)
# --------------------------------------------------------------------------------------------------
def _safe_div(a, b):
    b = np.where(np.asarray(b, dtype=float) == 0, np.nan, b)
    return np.asarray(a, dtype=float) / b


def build_customer_view(cust: pd.DataFrame, tx: pd.DataFrame, ref_date: pd.Timestamp,
                        exclude_category: str | None = None, prefix: str = "") -> pd.DataFrame:
    """
    Engineer one row per customer from transactions dated <= ref_date.

    exclude_category: drop that category's transactions BEFORE computing behaviour. Used for banner routing so
    the model only sees what a customer does OUTSIDE the target category (prevents direct AND indirect leakage,
    e.g. a total-spend denominator inflated by Home Centre's large tickets).
    prefix: prepended to behavioural feature names (e.g. 'nh_' = non-home) so leakage-safe columns are explicit.
    """
    ref_date = pd.Timestamp(ref_date)
    t = tx[tx["date"] <= ref_date]
    if exclude_category is not None:
        t = t[t["category"] != exclude_category]
    g = t.groupby("customer_id")
    p = prefix
    f = pd.DataFrame({
        f"{p}recency_days": (ref_date - g["date"].max()).dt.days,
        f"{p}frequency": g.size(),
        f"{p}monetary": g["amount"].sum(),
        f"{p}avg_ticket": g["amount"].mean(),
        f"{p}median_ticket": g["amount"].median(),
        f"{p}max_ticket": g["amount"].max(),
        f"{p}discount_rate": g["discount_used"].mean(),
        f"{p}n_banners": g["banner"].nunique(),
        f"{p}n_categories": g["category"].nunique(),
        f"{p}active_months": g["date"].apply(lambda s: s.dt.to_period("M").nunique()),
        f"{p}days_since_first": (ref_date - g["date"].min()).dt.days,
    })
    life = np.maximum(f[f"{p}days_since_first"].to_numpy(dtype=float), 30.0)
    f[f"{p}purchase_rate_30d"] = f[f"{p}frequency"] / life * 30
    f[f"{p}spend_rate_30d"] = f[f"{p}monetary"] / life * 30
    f[f"{p}mean_gap_days"] = _safe_div(f[f"{p}days_since_first"], (f[f"{p}frequency"] - 1))
    f[f"{p}recency_gap_ratio"] = _safe_div(f[f"{p}recency_days"], f[f"{p}mean_gap_days"])
    for w in (90, 180):
        tw = t[t["date"] > ref_date - pd.Timedelta(days=w)].groupby("customer_id")["amount"]
        f[f"{p}f{w}"] = tw.size()
        f[f"{p}spend{w}"] = tw.sum()
    prev = t[(t["date"] <= ref_date - pd.Timedelta(days=90)) &
             (t["date"] > ref_date - pd.Timedelta(days=180))].groupby("customer_id").size()
    f[f"{p}f_prev90"] = prev
    for c in [f"{p}f90", f"{p}spend90", f"{p}f180", f"{p}spend180", f"{p}f_prev90"]:
        f[c] = f[c].fillna(0)
    f[f"{p}trend90"] = f[f"{p}f90"] - f[f"{p}f_prev90"]

    b = t.pivot_table(index="customer_id", columns="banner", values="amount", aggfunc="sum", fill_value=0)
    k = t.pivot_table(index="customer_id", columns="category", values="amount", aggfunc="sum", fill_value=0)
    kt = t.pivot_table(index="customer_id", columns="category", values="amount", aggfunc="mean")
    bshare = b.div(b.sum(axis=1), axis=0)
    f[f"{p}banner_hhi"] = (bshare ** 2).sum(axis=1)
    view = cust.set_index("customer_id")
    view = view.join(f)
    view = view.join(bshare.add_prefix(f"{p}bshare_")).join(k.div(k.sum(axis=1), axis=0).add_prefix(f"{p}cshare_"))
    view = view.join(b.add_prefix(f"{p}bspend_")).join(k.add_prefix(f"{p}cspend_"))
    view = view.join(kt.add_prefix(f"{p}cticket_"))  # NaN = never bought that category (handled by imputer)
    zero_fill = [c for c in view.columns if c.startswith((f"{p}bshare_", f"{p}cshare_", f"{p}bspend_", f"{p}cspend_"))]
    zero_fill += [f"{p}frequency", f"{p}monetary", f"{p}n_banners", f"{p}n_categories", f"{p}active_months",
                  f"{p}f90", f"{p}f180", f"{p}spend90", f"{p}spend180", f"{p}f_prev90", f"{p}trend90",
                  f"{p}purchase_rate_30d", f"{p}spend_rate_30d"]
    view[zero_fill] = view[zero_fill].fillna(0)
    view[f"{p}has_tx"] = (view[f"{p}frequency"] > 0).astype(int)
    for c in ("monetary", "avg_ticket", "spend90", "spend180", "spend_rate_30d"):
        view[f"{p}log_{c}"] = np.log1p(view[f"{p}{c}"])
    view[f"{p}log_frequency"] = np.log1p(view[f"{p}frequency"])
    # profile / cohort features (identical whatever the prefix)
    view["days_member_ref"] = (ref_date - view["signup_date"]).dt.days
    view["signup_year"] = view["signup_date"].dt.year
    view["signup_quarter"] = view["signup_date"].dt.to_period("Q").astype(str)
    view["tenure_band"] = pd.cut(view["tenure_months"], [0, 12, 24, 48, 72, 1000],
                                 labels=["0-12m", "13-24m", "25-48m", "49-72m", "73m+"], include_lowest=True)
    view["age_band"] = pd.cut(view["age"], [0, 29, 39, 49, 59, 120], labels=["20-29", "30-39", "40-49", "50-59", "60+"])
    view["household"] = np.where(view["has_kids"] == 1, "With kids", "No kids")
    if not prefix:
        view["recency_band"] = pd.cut(view["recency_days"], [-1, 30, 60, 90, 180, 10_000],
                                      labels=["0-30d", "31-60d", "61-90d", "91-180d", "180d+"])
        view["top_banner"] = b.idxmax(axis=1).reindex(view.index)
    return view


def routing_label(tx: pd.DataFrame, premium: str, value: str, lo: float = 0.4, hi: float = 0.6):
    """
    Dominant-banner label for a premium/value pair, from observed spend in the two banners only.
    share = premium spend / (premium + value spend).  >= hi -> 1 (premium), <= lo -> 0 (value),
    (lo, hi) -> ambiguous, excluded and counted.  Customers with no spend in either banner are not in scope.
    """
    s = tx[tx["banner"].isin([premium, value])].pivot_table(
        index="customer_id", columns="banner", values="amount", aggfunc="sum", fill_value=0)
    for col in (premium, value):
        if col not in s:
            s[col] = 0.0
    share = s[premium] / (s[premium] + s[value])
    y = pd.Series(np.where(share >= hi, 1, np.where(share <= lo, 0, -1)), index=share.index)
    info = {"in_scope": int(len(share)), "premium_1": int((y == 1).sum()), "value_0": int((y == 0).sum()),
            "ambiguous_excluded": int((y == -1).sum()),
            "shop_both_banners": int(((s[premium] > 0) & (s[value] > 0)).sum())}
    return y[y >= 0].astype(int), share, info


# --------------------------------------------------------------------------------------------------
# Financial helpers (recorded revenue = sum of observed `amount`; NOT audited financial statements)
# --------------------------------------------------------------------------------------------------
def combined_transactions(tx: pd.DataFrame, viva: pd.DataFrame) -> pd.DataFrame:
    """Stack Shukran-banner and Viva rows for GROUP-LEVEL revenue only (never for customer-level joins)."""
    a = tx[["transaction_id", "date", "banner", "category", "amount", "discount_used"]].assign(source="Shukran (7 banners)")
    v = viva[["transaction_id", "date", "category", "amount"]].assign(banner="Viva", discount_used=np.nan,
                                                                      source="Viva (separate IDs)")
    return pd.concat([a, v[a.columns]], ignore_index=True)


def period_coverage(df: pd.DataFrame, freq: str = "Q", lo=None, hi=None) -> pd.DataFrame:
    """Revenue per calendar period with a partial-period flag based on the data-collection window [lo, hi]
    (defaults: the frame's own first and last transaction dates)."""
    lo = pd.Timestamp(lo).normalize() if lo is not None else df["date"].min().normalize()
    hi = pd.Timestamp(hi).normalize() if hi is not None else df["date"].max().normalize()
    per = df["date"].dt.to_period(freq)
    out = df.groupby(per).agg(revenue=("amount", "sum"), transactions=("amount", "size"),
                              avg_ticket=("amount", "mean")).reset_index().rename(columns={"date": "period"})
    out["period_start"] = out["period"].map(lambda p: p.start_time.normalize())
    out["period_end"] = out["period"].map(lambda p: p.end_time.normalize())
    out["covered_start"] = out["period_start"].clip(lower=lo)
    out["covered_end"] = out["period_end"].clip(upper=hi)
    out["days_in_period"] = (out["period_end"] - out["period_start"]).dt.days + 1
    out["days_covered"] = (out["covered_end"] - out["covered_start"]).dt.days + 1
    out["coverage_pct"] = (out["days_covered"] / out["days_in_period"] * 100).round(1)
    out["partial"] = out["days_covered"] < out["days_in_period"]
    out["period"] = out["period"].astype(str)
    return out


def banner_ranking(tx: pd.DataFrame, viva: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Rank all eight banners on recorded revenue and transaction volume within [start, end]."""
    allx = combined_transactions(tx, viva)
    if start is not None:
        allx = allx[allx["date"] >= pd.Timestamp(start)]
    if end is not None:
        allx = allx[allx["date"] <= pd.Timestamp(end)]
    r = allx.groupby("banner").agg(revenue_aed=("amount", "sum"), transactions=("amount", "size"),
                                   avg_ticket_aed=("amount", "mean"), discount_rate=("discount_used", "mean"))
    r = r.reindex(ALL_BANNERS)
    r["revenue_share_pct"] = r["revenue_aed"] / r["revenue_aed"].sum() * 100
    txw = tx if start is None else tx[tx["date"] >= pd.Timestamp(start)]
    txw = txw if end is None else txw[txw["date"] <= pd.Timestamp(end)]
    r["reach"] = txw.groupby("banner")["customer_id"].nunique().reindex(ALL_BANNERS)
    vw = allx[allx["banner"] == "Viva"]
    vids = viva.loc[viva["transaction_id"].isin(vw["transaction_id"]), "viva_member_id"].dropna()
    r.loc["Viva", "reach"] = vids.nunique()
    r["reach_basis"] = ["Unique Shukran IDs"] * 7 + ["Unique Viva Rewards IDs (+ anonymous walk-ins) — NOT comparable"]
    r["rank_revenue"] = r["revenue_aed"].rank(ascending=False, method="min").astype(int)
    r["rank_transactions"] = r["transactions"].rank(ascending=False, method="min").astype(int)
    r["rank_avg_ticket"] = r["avg_ticket_aed"].rank(ascending=False, method="min").astype(int)
    r["category"] = r.index.map(BANNER_CATEGORY)
    r["positioning"] = r.index.map(BANNER_POSITION)
    return r.reset_index().rename(columns={"index": "banner"}).sort_values("rank_revenue")


def contribution_scenario(ranking: pd.DataFrame, margins: dict) -> pd.DataFrame:
    """ASSUMPTION-DRIVEN contribution proxy = recorded revenue x user-entered margin. Not actual profit."""
    s = ranking[["banner", "revenue_aed", "rank_revenue"]].copy()
    s["assumed_margin_pct"] = s["banner"].map(margins).astype(float)
    s["contribution_proxy_aed"] = s["revenue_aed"] * s["assumed_margin_pct"] / 100
    s["rank_contribution_proxy"] = s["contribution_proxy_aed"].rank(ascending=False, method="min").astype(int)
    s["rank_change_vs_revenue"] = s["rank_revenue"] - s["rank_contribution_proxy"]
    return s.sort_values("rank_contribution_proxy")


# --------------------------------------------------------------------------------------------------
# Epsilon-greedy offer sandbox — SIMULATION ONLY (no exposure / redemption data exists)
# --------------------------------------------------------------------------------------------------
def epsilon_greedy_sim(true_p, eps: float, steps: int = 2000, runs: int = 200, decay: bool = False,
                       seed: int = SEED) -> dict:
    """
    Vectorised epsilon-greedy bandit over `runs` independent simulations.
    true_p: HYPOTHETICAL redemption probabilities per offer (user-entered). Rewards are Bernoulli draws.
    Returns average reward per step, cumulative regret, and share of pulls on the best arm.
    """
    rng = np.random.default_rng(seed)
    p = np.asarray(true_p, dtype=float)
    k = len(p)
    q = np.zeros((runs, k))
    n = np.zeros((runs, k))
    best = int(np.argmax(p))
    rew = np.zeros(steps)
    opt = np.zeros(steps)
    regret = np.zeros(steps)
    idx = np.arange(runs)
    for t in range(steps):
        e = eps / (1 + t / 200) if decay else eps
        explore = rng.random(runs) < e
        greedy = np.argmax(q + rng.random((runs, k)) * 1e-9, axis=1)  # random tie-break
        a = np.where(explore, rng.integers(0, k, runs), greedy)
        r = (rng.random(runs) < p[a]).astype(float)
        n[idx, a] += 1
        q[idx, a] += (r - q[idx, a]) / n[idx, a]
        rew[t] = r.mean()
        opt[t] = (a == best).mean()
        regret[t] = p[best] - p[a].mean()
    return {"avg_reward": rew, "pct_optimal": opt, "cum_regret": np.cumsum(regret),
            "final_estimates": q.mean(axis=0), "pulls_share": n.sum(axis=0) / n.sum()}


# --------------------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------------------
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return str(o)


def _strict_json(o):
    """Replace NaN/inf (not valid JSON) with null so every artifact parses in any JSON reader, not only Python's."""
    if isinstance(o, dict):
        return {k: _strict_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_strict_json(v) for v in o]
    if isinstance(o, np.ndarray):
        return _strict_json(o.tolist())
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    return o


def save_json(obj, path: Path):
    Path(path).write_text(json.dumps(_strict_json(obj), indent=2, default=_json_default, allow_nan=False))


def load_json(path: Path):
    return json.loads(Path(path).read_text())


TEAM = ["Sparsh Saraya", "Shashaank Sharma", "Sanjay Seemakurti", "Shirley Waeni Kioko", "Apurba Dash"]


def sort_rules(d: pd.DataFrame, by: list, ascending: list) -> pd.DataFrame:
    """Sort association rules reproducibly. Rule mining returns mirror rules (A → B, B → A) with identical lift in an order
    that depends on Python's string hashing, so exact ties are broken by rule name; float keys are rounded to 9 decimals
    so that last-digit floating-point noise cannot reorder them."""
    tmp, keys = d.copy(), []
    for c in by:
        k = f"__key_{c}"
        tmp[k] = tmp[c].round(9) if pd.api.types.is_float_dtype(tmp[c]) else tmp[c]
        keys.append(k)
    tie = [c for c in ("antecedent", "consequent") if c in tmp.columns and c not in by]
    return tmp.sort_values(keys + tie, ascending=list(ascending) + [True] * len(tie), kind="mergesort").drop(columns=keys)


def fmt_small_p(p: float) -> str:
    """A probability that may round to zero: '< 0.001' instead of '0.0000'."""
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def fmt_aed(x, dp: int = 0) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    if abs(x) >= 1e6:
        return f"AED {x / 1e6:,.2f}M"
    if abs(x) >= 1e3 and dp == 0:
        return f"AED {x / 1e3:,.1f}K"
    return f"AED {x:,.{dp}f}"


# --------------------------------------------------------------------------------------------------
# Readable feature names (used by the notebook reflections and the dashboard)
# --------------------------------------------------------------------------------------------------
FEATURE_LABELS = {"recency_days": "Days since last purchase", "days_since_first": "Days since first purchase", "days_member_ref": "Days as member",
       "purchase_rate_30d": "Purchases per 30 days", "spend_rate_30d": "Spend per 30 days", "log_spend_rate_30d": "Spend per 30 days (log)",
       "recency_gap_ratio": "Recency ÷ usual gap", "mean_gap_days": "Usual gap between purchases (days)", "discount_rate": "Share of purchases discounted",
       "log_avg_ticket": "Average ticket (log)", "log_monetary": "Total spend (log)", "log_frequency": "Purchases (log)", "median_ticket": "Median ticket (AED)",
       "max_ticket": "Largest ticket (AED)", "n_banners": "Banners used", "n_categories": "Categories bought", "banner_hhi": "Banner concentration (HHI)",
       "f90": "Purchases, last 90 days", "f180": "Purchases, last 180 days", "trend90": "Change in purchases vs prior 90 days",
       "log_spend90": "Spend, last 90 days (log)", "log_spend180": "Spend, last 180 days (log)", "tenure_months": "Tenure (months)",
       "family_size": "Family size", "has_kids": "Has kids", "age": "Age", "gender": "Gender", "emirate": "Emirate"}


def feature_label(f: str) -> str:
    """Readable label for a model feature; nh_/nf_ = computed on non-home / non-fashion transactions only."""
    scope = " (outside home)" if f.startswith("nh_") else " (outside fashion)" if f.startswith("nf_") else ""
    b = f[3:] if scope else f
    for pre, txt in (("bshare_", "Spend share · {}"), ("cshare_", "Category share · {}"), ("cticket_", "Avg ticket · {}"),
                     ("bspend_", "Spend · {}"), ("cspend_", "Category spend · {}")):
        if b.startswith(pre):
            return txt.format(b[len(pre):]) + scope
    return FEATURE_LABELS.get(b, b.replace("_", " ")) + scope


# --------------------------------------------------------------------------------------------------
# Validation helpers — bootstrap confidence intervals, calibration, scoring guard, inactivity labels
# --------------------------------------------------------------------------------------------------
def bootstrap_indices(y, n_boot: int = 1000, seed: int = SEED, stratify: bool = True) -> np.ndarray:
    """Row indices for n_boot resamples of a fixed test set. Stratified resampling keeps the class mix
    (no resample can lose all positives); the same indices are reused for every model, so differences are paired."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    n = len(y)
    if not stratify:
        return rng.integers(0, n, size=(n_boot, n))
    out = np.empty((n_boot, n), dtype=int)
    pos = 0
    for c in np.unique(y):
        g = np.flatnonzero(y == c)
        out[:, pos:pos + len(g)] = g[rng.integers(0, len(g), size=(n_boot, len(g)))]
        pos += len(g)
    return out


def bootstrap_ci(y, preds: dict, metric, idx: np.ndarray, alpha: float = 0.05):
    """Percentile CI of metric(y, pred) for each named prediction vector, on the SAME resamples.
    Returns (table, samples) — samples[name] holds the bootstrap distribution for paired comparisons."""
    y = np.asarray(y)
    samples, rows = {}, []
    for k, p in preds.items():
        p = np.asarray(p)
        s = np.array([metric(y[i], p[i]) for i in idx])
        samples[k] = s
        rows.append({"model": k, "point": float(metric(y, p)), "ci_low": float(np.quantile(s, alpha / 2)),
                     "ci_high": float(np.quantile(s, 1 - alpha / 2)), "boot_se": float(s.std(ddof=1))})
    return pd.DataFrame(rows), samples


def paired_difference(samples_a, samples_b, point_a: float, point_b: float, alpha: float = 0.05) -> dict:
    """Difference A − B with a paired percentile CI and the share of resamples in which A beats B."""
    d = np.asarray(samples_a) - np.asarray(samples_b)
    lo, hi = float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))
    return {"diff": float(point_a - point_b), "ci_low": lo, "ci_high": hi, "share_A_better": float((d > 0).mean()),
            "verdict": "A better" if lo > 0 else ("B better" if hi < 0 else "no clear difference (CI includes 0)")}


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table with equal-count (quantile) bins: mean predicted probability vs observed rate."""
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "p": np.asarray(p, dtype=float)})
    df["bin"] = pd.qcut(df.p.rank(method="first"), n_bins, labels=False)
    t = df.groupby("bin").agg(mean_predicted=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size")).reset_index()
    t["gap"] = t.observed_rate - t.mean_predicted
    return t


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    t = calibration_table(y, p, n_bins)
    return float((t.n * t.gap.abs()).sum() / t.n.sum())


def clip_to_reference(X: pd.DataFrame, ref: pd.DataFrame, cols) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scoring guard: clip each numeric feature to the range seen in training, so a model is never asked to
    extrapolate (e.g. a 'days since first record' feature that keeps growing after the training date).
    Returns the clipped frame and a report of how many rows fell outside the training range."""
    cols = list(cols)
    lo, hi = ref[cols].min(), ref[cols].max()
    Xc = X.copy()
    Xc[cols] = X[cols].clip(lower=lo, upper=hi, axis=1)
    rep = pd.DataFrame({"train_min": lo, "train_max": hi, "score_min": X[cols].min(), "score_max": X[cols].max(),
                        "share_below_range": (X[cols] < lo).mean(), "share_above_range": (X[cols] > hi).mean()})
    rep["share_clipped"] = rep.share_below_range + rep.share_above_range
    return Xc, rep


def inactivity_label(tx: pd.DataFrame, ids, t: pd.Timestamp, horizon_days: int = 90) -> pd.Series:
    """1 = no Shukran transaction in (t, t + horizon]; the rule the supplied churn flag follows at the as-of date."""
    w = tx[(tx["date"] > t) & (tx["date"] <= t + pd.Timedelta(days=horizon_days))]
    bought = w.groupby("customer_id").size()
    return (bought.reindex(ids).fillna(0) == 0).astype(int)


# --------------------------------------------------------------------------------------------------
# Association rules — activity-adjusted lift and Cochran–Mantel–Haenszel test
# --------------------------------------------------------------------------------------------------
def stratified_lift(basket: pd.DataFrame, antecedent, consequent, strata: pd.Series) -> dict:
    """Lift after holding a confounder fixed: observed co-occurrence / co-occurrence expected if A and C were
    independent *within* each stratum (e.g. activity band). Lift ≈ 1 → the raw affinity is explained by the strata."""
    A = basket[list(antecedent)].all(axis=1)
    C = basket[list(consequent)].all(axis=1)
    df = pd.DataFrame({"A": A.astype(float), "C": C.astype(float), "s": strata.reindex(basket.index)}).dropna()
    g = df.groupby("s")
    observed = float((df.A * df.C).sum())
    expected = float((g.size() * g.A.mean() * g.C.mean()).sum())
    return {"observed": observed, "expected": expected, "adjusted_lift": observed / expected if expected > 0 else np.nan}


def cmh_test(basket: pd.DataFrame, antecedent, consequent, strata: pd.Series) -> float:
    """Cochran–Mantel–Haenszel chi-square p-value (no continuity correction) for A–C association across strata."""
    from scipy.stats import chi2
    A = basket[list(antecedent)].all(axis=1)
    C = basket[list(consequent)].all(axis=1)
    df = pd.DataFrame({"A": A, "C": C, "s": strata.reindex(basket.index)}).dropna()
    num, var = 0.0, 0.0
    for _, g in df.groupby("s"):
        n = len(g)
        if n < 2:
            continue
        a = float((g.A & g.C).sum())
        n1, m1 = float(g.A.sum()), float(g.C.sum())
        n0, m0 = n - n1, n - m1
        num += a - n1 * m1 / n
        var += n1 * n0 * m1 * m0 / (n ** 2 * (n - 1))
    return float(chi2.sf(num ** 2 / var, 1)) if var > 0 else np.nan


# --------------------------------------------------------------------------------------------------
# Forward CLV — two-part (hurdle) regressor and a probabilistic BG/NBD + Gamma-Gamma benchmark
# --------------------------------------------------------------------------------------------------
from sklearn.base import BaseEstimator, RegressorMixin, clone as _clone   # noqa: E402


class HurdleRegressor(RegressorMixin, BaseEstimator):
    """Two-part model: P(spend > 0) from a classifier × E[spend | spend > 0] from a regressor."""

    def __init__(self, classifier=None, regressor=None):
        self.classifier = classifier
        self.regressor = regressor

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        pos = y > 0
        self.clf_ = _clone(self.classifier).fit(X, pos.astype(int))
        self.reg_ = _clone(self.regressor).fit(X[pos], y[pos])
        return self

    def predict(self, X):
        return self.clf_.predict_proba(X)[:, 1] * np.clip(self.reg_.predict(X), 0, None)


def rfm_summary(tx: pd.DataFrame, cutoff: pd.Timestamp, unit_days: float = 7.0) -> pd.DataFrame:
    """Per-member inputs for BG/NBD and Gamma-Gamma from transactions up to `cutoff`.
    A 'transaction' is a purchase day (several receipts on one day count once). Time is in weeks.
    x = repeat purchase days, t_x = time of last purchase since the first, T = time from first purchase to cutoff,
    m = mean spend of the repeat purchase days (0 when x = 0)."""
    t = tx[tx["date"] <= cutoff]
    daily = t.groupby(["customer_id", t["date"].dt.normalize()]).amount.sum().reset_index()
    g = daily.groupby("customer_id")
    first, last, n = g.date.min(), g.date.max(), g.size()
    out = pd.DataFrame({"x": n - 1, "t_x": (last - first).dt.days / unit_days, "T": (cutoff.normalize() - first).dt.days / unit_days})
    daily = daily.merge(first.rename("first"), left_on="customer_id", right_index=True)
    rep = daily[daily.date > daily["first"]]
    out["m"] = rep.groupby("customer_id").amount.mean().reindex(out.index).fillna(0.0)
    return out


def _bgnbd_ll(params, x, t_x, T):
    from scipy.special import gammaln, betaln
    r, alpha, a, b = params
    A1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    A2 = betaln(a, b + x) - betaln(a, b) - (r + x) * np.log(alpha + T)
    with np.errstate(divide="ignore", invalid="ignore"):
        A3 = np.where(x > 0, betaln(a + 1, b + np.maximum(x, 1) - 1) - betaln(a, b) - (r + x) * np.log(alpha + t_x), -np.inf)
    return A1 + np.logaddexp(A2, A3)


LOG_BOUNDS = (-8.0, 12.0)          # parameters kept within [3e-4, 1.6e5]: beyond that gammaln() loses precision


def _bounded_fit(nll, starts, k):
    """Nelder-Mead (penalised outside the bounds) followed by bounded L-BFGS-B, from several starting points."""
    from scipy.optimize import minimize
    lo, hi = LOG_BOUNDS
    pen = lambda lp: nll(np.clip(lp, lo, hi)) + 1e3 * np.sum(np.clip(np.abs(lp - np.clip(lp, lo, hi)), 0, None))
    best = None
    for s in starts:
        nm = minimize(pen, np.log(s), method="Nelder-Mead", options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-8})
        res = minimize(nll, np.clip(nm.x, lo, hi), method="L-BFGS-B", bounds=[LOG_BOUNDS] * k)
        res.success = bool(nm.success or res.success)        # L-BFGS-B may stop on line-search precision at the optimum
        if best is None or res.fun < best.fun:
            best = res
    at_bound = [bool(np.isclose(v, lo, atol=1e-3) or np.isclose(v, hi, atol=1e-3)) for v in best.x]
    return best, at_bound


def bgnbd_fit(x, t_x, T, starts=((1, 1, 1, 1), (0.3, 5, 0.8, 2.5), (2, 20, 1, 5))) -> dict:
    """Maximum-likelihood BG/NBD (Fader, Hardie & Lee 2005), fitted on bounded log-parameters from several starts.
    `params_at_bound` flags parameters the data cannot identify (e.g. r, α → ∞ when purchase rates are homogeneous)."""
    x, t_x, T = (np.asarray(v, dtype=float) for v in (x, t_x, T))
    best, at_bound = _bounded_fit(lambda lp: -np.sum(_bgnbd_ll(np.exp(lp), x, t_x, T)), starts, 4)
    r, alpha, a, b = np.exp(best.x)
    return {"r": float(r), "alpha": float(alpha), "a": float(a), "b": float(b), "mean_rate_per_unit": float(r / alpha),
            "mean_dropout_p": float(a / (a + b)), "log_likelihood": float(-best.fun), "converged": bool(best.success),
            "params_at_bound": [n for n, f in zip(["r", "alpha", "a", "b"], at_bound) if f]}


def bgnbd_expected(p: dict, t: float, x, t_x, T) -> np.ndarray:
    """Expected number of repeat transactions in (T, T + t] for each member (FHL 2005, eq. 10)."""
    from scipy.special import hyp2f1
    r, alpha, a, b = p["r"], p["alpha"], p["a"], p["b"]
    x, t_x, T = (np.asarray(v, dtype=float) for v in (x, t_x, T))
    hyp = hyp2f1(r + x, b + x, a + b + x - 1, t / (alpha + T + t))
    first = (a + b + x - 1) / (a - 1) * (1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp)
    with np.errstate(divide="ignore", invalid="ignore"):
        second = np.where(x > 0, a / (b + x - 1) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0.0)
    return first / (1 + second)


def bgnbd_p_alive(p: dict, x, t_x, T) -> np.ndarray:
    r, alpha, a, b = p["r"], p["alpha"], p["a"], p["b"]
    x, t_x, T = (np.asarray(v, dtype=float) for v in (x, t_x, T))
    with np.errstate(divide="ignore", invalid="ignore"):
        second = np.where(x > 0, a / (b + x - 1) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0.0)
    return 1 / (1 + second)


def gamma_gamma_fit(x, m) -> dict:
    """Gamma-Gamma spend model (Fader & Hardie) on members with at least one repeat purchase."""
    from scipy.special import gammaln
    x, m = np.asarray(x, dtype=float), np.asarray(m, dtype=float)
    keep = (x > 0) & (m > 0)
    x, m = x[keep], m[keep]

    def nll(lp):
        p, q, v = np.exp(lp)
        ll = (gammaln(p * x + q) - gammaln(p * x) - gammaln(q) + q * np.log(v) + (p * x - 1) * np.log(m)
              + (p * x) * np.log(x) - (p * x + q) * np.log(x * m + v))
        return -np.sum(ll)
    best, at_bound = _bounded_fit(nll, ((1, 1, 1), (5, 3, 10), (2, 4, float(np.median(m)))), 3)
    p, q, v = np.exp(best.x)
    return {"p": float(p), "q": float(q), "v": float(v), "log_likelihood": float(-best.fun), "converged": bool(best.success),
            "n_used": int(keep.sum()), "params_at_bound": [n for n, f in zip(["p", "q", "v"], at_bound) if f]}


def gamma_gamma_expected(gg: dict, x, m) -> np.ndarray:
    """Conditional expected spend per transaction; members with no repeat purchase get the population mean."""
    p, q, v = gg["p"], gg["q"], gg["v"]
    x, m = np.asarray(x, dtype=float), np.asarray(m, dtype=float)
    pop = p * v / (q - 1)
    cond = p * (v + x * m) / (p * x + q - 1)
    return np.where(x > 0, cond, pop)


def bgnbd_simulate(params: dict, n: int, T, seed: int = SEED, horizon: float = 0.0):
    """Simulate the BG/NBD process (used by the unit tests to check that bgnbd_fit recovers known parameters).
    Returns x, t_x within (0, T] and the number of repeat purchases in (T, T + horizon]."""
    rng = np.random.default_rng(seed)
    lam = rng.gamma(params["r"], 1 / params["alpha"], n)
    pdrop = rng.beta(params["a"], params["b"], n)
    T = np.broadcast_to(np.asarray(T, dtype=float), (n,))
    x, t_x, fut = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        t, alive = 0.0, True
        while alive:
            t += rng.exponential(1 / lam[i])
            if t > T[i] + horizon:
                break
            if t <= T[i]:
                x[i] += 1; t_x[i] = t
            else:
                fut[i] += 1
            alive = rng.random() > pdrop[i]
    return x, t_x, T.copy(), fut
