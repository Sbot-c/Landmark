"""
app.py — Landmark Retail Group · Customer Intelligence Engine (Streamlit dashboard)

Reads ONLY the artifacts written by notebooks/Landmark_Customer_Intelligence.ipynb (artifacts/) plus the raw CSVs
for interactive revenue filtering. It never retrains a model, so every metric shown matches the notebook.

    streamlit run app.py
"""
from __future__ import annotations

import inspect
import json
import textwrap
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import lci_core as L

st.set_page_config(page_title="Landmark · Customer Intelligence Engine", page_icon=":material/insights:",
                   layout="wide", initial_sidebar_state="expanded")

NAVY, NAVY2, GOLD, IVORY, SLATE, INK, LINE = L.UI["navy"], L.UI["navy2"], L.UI["gold"], L.UI["ivory"], L.UI["slate"], L.UI["ink"], "#E6E2D8"
BC, SERIES, STATUS = L.BANNER_COLORS, L.SERIES, L.STATUS
FAMILY_COLORS = {"Single": SERIES[0], "Bagging": SERIES[2], "Boosting": SERIES[3], "Voting": SERIES[4], "Stacking": SERIES[1], "Regression": SERIES[0]}
FAMILY_SYMBOLS = {"Single": "circle", "Bagging": "square", "Boosting": "diamond", "Voting": "triangle-up", "Stacking": "x"}

# ----------------------------------------------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------------------------------------------
st.markdown(f"""
<style>
:root {{ --navy:{NAVY}; --navy2:{NAVY2}; --gold:{GOLD}; --ivory:{IVORY}; --slate:{SLATE}; --ink:{INK}; --line:{LINE}; }}
.block-container {{ padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1420px; }}
h1, h2, h3 {{ color: var(--navy); letter-spacing: .1px; }}
.lm-hero {{ background: linear-gradient(118deg, #14213D 0%, #1F3A5F 68%, #29507E 100%); border-radius: 14px;
           padding: 20px 26px 16px; color: #F7F5F0; border-bottom: 3px solid var(--gold); margin-bottom: 14px; }}
.lm-hero.compact {{ padding: 12px 22px 10px; }}
.lm-eyebrow {{ text-transform: uppercase; letter-spacing: .16em; font-size: .7rem; color: #DCC59C; font-weight: 600; }}
.lm-hero h1 {{ font-size: 1.62rem; margin: 2px 0 0; color: #FFFFFF; font-weight: 700; }}
.lm-hero.compact h1 {{ font-size: 1.15rem; }}
.lm-sub {{ color: #D5DBE6; font-size: .93rem; margin-top: 4px; max-width: 980px; }}
.lm-chips span {{ display: inline-block; border: 1px solid rgba(255,255,255,.26); border-radius: 999px; padding: 2px 10px;
                 margin: 9px 6px 0 0; font-size: .74rem; color: #E8ECF3; }}
.lm-kpis {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; margin-top: 14px; }}
.lm-kpi {{ background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.13); border-radius: 10px; padding: 10px 12px; }}
.lm-kpi .l {{ font-size: .68rem; color: #C6CFDC; text-transform: uppercase; letter-spacing: .07em; }}
.lm-kpi .v {{ font-size: 1.28rem; font-weight: 700; color: #FFFFFF; margin-top: 2px; }}
.lm-kpi .v.gold {{ color: #E6C894; }}
.lm-kpi .s {{ font-size: .72rem; color: #AEB9CA; }}
.lm-note {{ font-size: .74rem; color: #B9C3D3; margin-top: 10px; }}
.card {{ background: #FFFFFF; border: 1px solid var(--line); border-radius: 12px; padding: 13px 15px; height: 100%; }}
.card .l {{ font-size: .7rem; text-transform: uppercase; letter-spacing: .07em; color: var(--slate); }}
.card .v {{ font-size: 1.32rem; font-weight: 700; color: var(--navy); margin: 2px 0; }}
.card .s {{ font-size: .78rem; color: var(--slate); }}
.callout {{ border-left: 4px solid var(--gold); background: #FBF8F1; padding: 10px 14px; border-radius: 6px; font-size: .9rem; margin: 6px 0 10px; color: var(--ink); }}
.callout.risk {{ border-left-color: #d03b3b; background: #FDF3F2; }}
.callout.sim {{ border-left-color: #8E5A9B; background: #F6F1F8; }}
.callout.good {{ border-left-color: #2A9D8F; background: #EFF7F5; }}
.badge {{ display: inline-block; padding: 1px 9px; border-radius: 999px; font-size: .68rem; font-weight: 700; letter-spacing: .05em; vertical-align: middle; margin-left: 6px; }}
.b-measured {{ background: #E3F1EE; color: #1B6A60; }}
.b-model {{ background: #E6EEF8; color: #1F4E8C; }}
.b-scenario {{ background: #FBEFDC; color: #85570F; }}
.b-sim {{ background: #F1E6F5; color: #6B3E78; }}
.b-proposed {{ background: #ECEAE5; color: #4E4A42; }}
.sec-title {{ font-size: 1.32rem; font-weight: 700; color: var(--navy); margin: .25rem 0 0; }}
.sec-sub {{ color: var(--slate); font-size: .9rem; margin: .1rem 0 .7rem; }}
.small {{ font-size: .8rem; color: var(--slate); }}
[data-testid="stSidebar"] {{ background: #EFEBE2; }}
@media (max-width: 1000px) {{ .lm-kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
</style>
""", unsafe_allow_html=True)


def badge(kind: str) -> str:
    text = {"measured": "MEASURED", "model": "MODEL OUTPUT", "scenario": "SCENARIO · ASSUMPTION", "sim": "SIMULATION", "proposed": "PROPOSED"}[kind]
    cls = {"measured": "b-measured", "model": "b-model", "scenario": "b-scenario", "sim": "b-sim", "proposed": "b-proposed"}[kind]
    return f'<span class="badge {cls}">{text}</span>'


def section(title: str, sub: str = "", kind: str | None = None):
    st.markdown(f'<div class="sec-title">{title}{badge(kind) if kind else ""}</div>' + (f'<div class="sec-sub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


def callout(text: str, kind: str = ""):
    st.markdown(f'<div class="callout {kind}">{text}</div>', unsafe_allow_html=True)


def cards(items):
    cols = st.columns(len(items))
    for c, (label, value, sub) in zip(cols, items):
        c.markdown(f'<div class="card"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>', unsafe_allow_html=True)


def aed(x, dp=0):
    return L.fmt_aed(x, dp)


def style(fig, title=None, height=380, legend_bottom=True, xt=None, yt=None, legend_below=False):
    fig.update_layout(template="plotly_white", height=height, paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", size=12, color=INK),
                      title=dict(text=title, x=0, xanchor="left", y=0.98, yanchor="top", font=dict(size=14, color=NAVY)) if title else None,
                      margin=dict(l=10, r=14, t=(82 if legend_bottom and not legend_below else 50) if title else 30, b=110 if legend_below else 10),
                      legend=(dict(orientation="h", yref="container", yanchor="bottom", y=0.01, x=0, xanchor="left", title=None, font=dict(size=11)) if legend_below else
                              dict(orientation="h", yanchor="bottom", y=1.01, x=0, xanchor="left", title=None, font=dict(size=11))) if legend_bottom else dict(title=None),
                      hoverlabel=dict(bgcolor="white", font_size=12))
    fig.update_xaxes(showgrid=False, linecolor="#C3C2B7", title=xt, automargin=True)
    if xt in ("Quarter", "Calendar quarter", "Signup quarter"):
        fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(gridcolor="#ECE8DF", linecolor="#C3C2B7", title=yt, automargin=True, zeroline=False)
    fig.update_traces(cliponaxis=False, selector=dict(type="bar"))       # outside value labels are never clipped
    return fig


feat_label = L.feature_label


def cm_figure(key, title=None, height=330):
    """Test-set confusion matrix drawn from the notebook's cm_<key>.json (counts + share of the actual class)."""
    c = js(f"cm_{key}.json")
    z = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    pct = z / z.sum(axis=1, keepdims=True) * 100
    names = ["true negative", "false positive", "false negative", "true positive"]
    txt = [[f"<b>{z[i, j]:,}</b><br>{pct[i, j]:.0f}% of actual<br>{names[2 * i + j]}" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(z=pct, x=[f"Predicted {l}" for l in c["labels"]], y=[f"Actual {l}" for l in c["labels"]], text=txt, texttemplate="%{text}",
                               colorscale="Blues", zmin=0, zmax=100, showscale=False, hovertemplate="%{y} · %{x}<br>%{text}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return style(fig, title or f"{c['model']} · test set · threshold {c['threshold']:.2f}", height, legend_bottom=False)


def _full_width(fn) -> dict:
    """Full-width keyword for the installed Streamlit: width="stretch" on current releases, use_container_width=True on
    older ones (e.g. 1.45, where `width` is an integer or absent). Keeps the app working across the tested version range."""
    try:
        p = inspect.signature(fn).parameters.get("width")
        if p is not None and "Width" in str(p.annotation):
            return {"width": "stretch"}
    except (TypeError, ValueError):
        pass
    return {"use_container_width": True}


W_CHART, W_TABLE, W_EDITOR = _full_width(st.plotly_chart), _full_width(st.dataframe), _full_width(st.data_editor)


AFFINITY_SHORT = {"banner affinity (survives activity and household)": "genuine banner affinity", "household-driven (explained by kids)": "household-driven (kids)"}


def short_affinity(df):
    """Short labels for the 'what drives it' column so it fits the table width."""
    return df.assign(affinity_type=df.affinity_type.map(AFFINITY_SHORT).fillna(df.affinity_type)) if "affinity_type" in df.columns else df


def show(fig):
    st.plotly_chart(fig, **W_CHART, config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def table(df, height=None, **kw):
    if height is not None:
        kw["height"] = height
    cfg = kw.get("column_config") or {}
    for c in df.columns:                      # missing numbers show as an em dash, not "None"
        if c not in cfg and pd.api.types.is_float_dtype(df[c]) and df[c].isna().any():
            df = df.assign(**{c: df[c].map(lambda v: "—" if pd.isna(v) else f"{v:,.3f}".rstrip("0").rstrip("."))})
    st.dataframe(df, **W_TABLE, hide_index=kw.pop("hide_index", True), **kw)


# ----------------------------------------------------------------------------------------------------------------
# Data access (cached) — artifacts only; nothing is retrained
# ----------------------------------------------------------------------------------------------------------------
REQUIRED = ["customer_master.csv", "model_summary.json", "viva_gap.json", "banner_ranking_full.csv", "banner_ranking_ttm.csv", "fin_quarterly.csv",
            "segment_profiles.csv", "leaderboard_churn.csv", "leaderboard_home.csv", "leaderboard_fashion.csv", "threshold_curve.csv",
            "threshold_choice.json", "rules_all.csv", "regression_historical.csv", "regression_next_period.csv", "diagnostics.csv",
            "oot_summary.json", "calibration_summary.csv", "test_ci.csv", "window_feature_ablation.json", "clv_benchmark.csv", "clv_intervals.json",
            "style_choice.json", "style_profiles.csv"]


@st.cache_data(show_spinner=False)
def csv(name, index_col=None, parse_dates=None):
    p = L.METRICS / name
    if not p.exists():
        raise FileNotFoundError(name)
    df = pd.read_csv(p, index_col=index_col, parse_dates=parse_dates)
    if "status" in df.columns and df["status"].dtype == object:
        df["status"] = df["status"].fillna("")
    return df


@st.cache_data(show_spinner=False)
def js(name):
    return json.loads((L.METRICS / name).read_text())


@st.cache_data(show_spinner="Loading supplied transaction files…")
def raw():
    c, t, v, _ = L.load_raw()
    return c, t, v


@st.cache_data(show_spinner=False)
def ranking(start, end):
    _, t, v = raw()
    return L.banner_ranking(t, v, start, end)


missing = [f for f in REQUIRED if not (L.METRICS / f).exists()]
if missing:
    st.error("Artifacts are missing — run the notebook first:\n\n`jupyter nbconvert --to notebook --execute notebooks/Landmark_Customer_Intelligence.ipynb --inplace`"
             f"\n\nMissing: {', '.join(missing)}")
    st.stop()
try:
    CUST, TX, VIVA = raw()
except FileNotFoundError as e:
    st.error(f"Supplied CSVs not found in data/: {e}")
    st.stop()

CM = csv("customer_master.csv", index_col=0, parse_dates=["signup_date"])
GAP, MS, THR = js("viva_gap.json"), js("model_summary.json"), js("threshold_choice.json")
RANK_FULL, RANK_TTM = csv("banner_ranking_full.csv"), csv("banner_ranking_ttm.csv")
SEGP = csv("segment_profiles.csv", index_col=0)
SEG_ORDER = CM.segment.value_counts().index.tolist()          # from the member table itself, so filters always match it
SEG_COLORS = {s: SERIES[i] for i, s in enumerate(SEG_ORDER)}
SEG_SYMBOLS = {s: sym for s, sym in zip(SEG_ORDER, ["circle", "square", "diamond", "triangle-up", "x", "cross", "star", "pentagon"])}
OBS = pd.Timestamp(TX.date.max())
DMIN = min(TX.date.min(), VIVA.date.min()).date()
SRC_START = {"Shukran": TX.date.min().normalize(), "Viva": VIVA.date.min().normalize()}
DEND = max(TX.date.max(), VIVA.date.max()).normalize()


def coverage(sub, freq="Q", F=None):
    """Partial-period flags against the source's data-collection window (and the sidebar period),
    not the series' own first/last sale — a banner with no sale on the last day is not 'partial'."""
    lo = SRC_START["Viva"] if (sub.banner == "Viva").all() else SRC_START["Shukran"]
    hi = DEND
    if F is not None:
        lo, hi = max(lo, pd.Timestamp(F["p0"]).normalize()), min(hi, pd.Timestamp(F["p1"]).normalize())
    return L.period_coverage(sub, freq, lo=lo, hi=hi)

# ----------------------------------------------------------------------------------------------------------------
# Global filters (sidebar) — shared by every page
# ----------------------------------------------------------------------------------------------------------------
def sidebar_filters():
    sb = st.sidebar
    sb.markdown("#### Filters")
    sb.caption("Financial pages use **period** and **banners**; customer pages use the **customer** filters.")
    period = sb.date_input("Period (financial pages)", value=(TX.date.min().date(), OBS.date()), min_value=DMIN, max_value=OBS.date(), key="f_period")
    banners = sb.multiselect("Banners", L.ALL_BANNERS, default=L.ALL_BANNERS, key="f_banners")
    sb.markdown("---")
    segs = sb.multiselect("Segment", SEG_ORDER, default=SEG_ORDER, key="f_segs")
    status = sb.multiselect("Churn status", ["Active", "Churned"], default=["Active", "Churned"], key="f_status")
    vmax = float(np.ceil(CM.monetary.max() / 100) * 100)
    value = sb.slider("Historical spend (AED)", 0.0, vmax, (0.0, vmax), step=100.0, key="f_value")
    dd_levels = ["Discount-only (100%)", "Discount-dependent (75–99%)", "Mixed (26–74%)", "Mostly full-price (1–25%)", "Full-price loyalist (0%)", "Insufficient history (<5 tx)"]
    dd = sb.multiselect("Discount behaviour", [d for d in dd_levels if d in set(CM.discount_dependency)], default=[d for d in dd_levels if d in set(CM.discount_dependency)], key="f_dd")
    sb.markdown("---")
    sb.caption(f"Data as of **{OBS.date()}** (latest recorded transaction — not today's date). Revenue = sum of supplied transaction amounts, not audited financials.")
    sb.caption("**Team:** " + " · ".join(L.TEAM))
    if isinstance(period, (tuple, list)) and len(period) == 2:
        p0, p1 = pd.Timestamp(period[0]), pd.Timestamp(period[1])
    else:
        p0, p1 = TX.date.min(), OBS
    return {"p0": p0, "p1": p1, "banners": banners or L.ALL_BANNERS, "segs": segs or SEG_ORDER, "status": status or ["Active", "Churned"],
            "value": value, "dd": dd}


def filt_customers(F):
    m = CM.segment.isin(F["segs"]) & CM.churned.map({0: "Active", 1: "Churned"}).isin(F["status"])
    m &= CM.monetary.between(F["value"][0], F["value"][1]) & CM.discount_dependency.isin(F["dd"])
    return CM[m]


def empty_guard(d) -> bool:
    if len(d) == 0:
        st.info("No members match the current sidebar filters — widen the segment, status, spend or discount filters.")
        return True
    return False


def filter_note(F, kind="customers"):
    if kind == "customers":
        n = len(filt_customers(F))
        st.caption(f"Customer filters applied → **{n:,} of {len(CM):,} members** in view.")
    else:
        st.caption(f"Period {F['p0'].date()} → {F['p1'].date()} · banners: {', '.join(F['banners']) if len(F['banners']) < 8 else 'all eight'}")


def hero(compact=False):
    shk_rev, viva_rev = GAP["core_7_banner_revenue_aed"], GAP["viva_revenue_aed"]
    act, chn = int((CM.churned == 0).sum()), int((CM.churned == 1).sum())
    lead = RANK_TTM.sort_values("rank_revenue").iloc[0]
    hi_risk = int(CM.retention_action.str.startswith("Retain").sum())
    hi_pri = int((CM.retention_action == "Retain now — priority").sum())
    if compact:
        st.markdown(f'<div class="lm-hero compact"><div class="lm-eyebrow">Landmark Retail Group</div><h1>Customer Intelligence Engine</h1></div>', unsafe_allow_html=True)
        return
    st.markdown(f"""
<div class="lm-hero">
  <div class="lm-eyebrow">Landmark Retail Group · Shukran member analytics</div>
  <h1>Landmark Retail Group — Customer Intelligence Engine</h1>
  <div class="lm-sub">Who our members are, who may leave, who is most valuable, which banner fits each member, and what to offer them — one engine across eight banners.</div>
  <div class="lm-chips"><span>Shukran banners: {GAP['core_coverage'][0]} → {GAP['core_coverage'][1]}</span><span>Viva: {GAP['viva_coverage'][0]} → {GAP['viva_coverage'][1]}</span>
  <span>As-of date {OBS.date()} (not today)</span><span>Q4-2026 = 1 day · partial</span><span>{len(CM):,} Shukran member records</span></div>
  <div class="lm-kpis">
    <div class="lm-kpi"><div class="l">Recorded revenue · total</div><div class="v gold">{aed(shk_rev + viva_rev)}</div><div class="s">8 banners, supplied records</div></div>
    <div class="lm-kpi"><div class="l">Shukran banners (7)</div><div class="v">{aed(shk_rev)}</div><div class="s">{len(TX):,} transactions</div></div>
    <div class="lm-kpi"><div class="l">Viva (separate IDs)</div><div class="v">{aed(viva_rev)}</div><div class="s">{GAP['viva_share_common_window_pct']}% of revenue since Viva data begin</div></div>
    <div class="lm-kpi"><div class="l">Members active / churned</div><div class="v">{act:,} / {chn:,}</div><div class="s">churn rate {chn/len(CM)*100:.1f}%</div></div>
    <div class="lm-kpi"><div class="l">Early-warning list</div><div class="v">{hi_risk:,}</div><div class="s">active members above the cost threshold · {hi_pri:,} at ≥ 0.50</div></div>
    <div class="lm-kpi"><div class="l">Leading banner (TTM)</div><div class="v">{lead.banner}</div><div class="s">{aed(lead.revenue_aed)} · {lead.revenue_share_pct:.1f}% share</div></div>
  </div>
  <div class="lm-note">Revenue is calculated from the supplied transaction records (a 3,000-member Shukran sample plus Viva baskets) — it is not audited financial data and not Landmark's actual group revenue.</div>
</div>""", unsafe_allow_html=True)


def monthly_figure(df, key, cmap):
    """Monthly revenue per series: complete months joined by a line, partial months as hollow markers only."""
    fig = go.Figure()
    for k, sub in df.groupby(key):
        cov = coverage(sub, freq="M")
        cov["month"] = pd.PeriodIndex(cov.period, freq="M").to_timestamp()
        full, part = cov[~cov.partial], cov[cov.partial]
        fig.add_scatter(x=full.month, y=full.revenue, name=k, mode="lines+markers", line=dict(color=cmap.get(k, SLATE), width=2.2), marker=dict(size=5),
                        legendgroup=k, hovertemplate=f"{k}<br>%{{x|%b %Y}}: AED %{{y:,.0f}}<extra></extra>")
        if len(part):
            fig.add_scatter(x=part.month, y=part.revenue, mode="markers", legendgroup=k, showlegend=False, customdata=part.coverage_pct,
                            marker=dict(size=10, symbol="circle-open", color=cmap.get(k, SLATE), line=dict(width=2)),
                            hovertemplate=f"{k} (PARTIAL: %{{customdata:.0f}}% of month)<br>%{{x|%b %Y}}: AED %{{y:,.0f}}<extra></extra>")
    return fig


# ================================================================================================================
# A · Executive overview
# ================================================================================================================
def page_overview():
    hero()
    F = st.session_state["F"]
    c1, c2 = st.columns([1.05, 1])
    with c1:
        segd = csv("segment_dashboard.csv", index_col=0).reindex(SEG_ORDER)
        fig = go.Figure()
        fig.add_bar(y=segd.index, x=segd.share_of_members_pct, orientation="h", name="% of members", marker_color="#B9B4A8",
                    hovertemplate="%{y}<br>%{x:.1f}% of members<extra></extra>")
        fig.add_bar(y=segd.index, x=segd.share_of_revenue_pct, orientation="h", name="% of recorded revenue", marker_color=SERIES[0],
                    hovertemplate="%{y}<br>%{x:.1f}% of revenue<extra></extra>", text=[f"{v:.0f}%" for v in segd.share_of_revenue_pct], textposition="outside")
        fig.update_layout(barmode="group", bargap=0.25)
        fig.update_xaxes(range=[0, max(segd.share_of_members_pct.max(), segd.share_of_revenue_pct.max()) * 1.22])
        show(style(fig, "Segments: share of members vs share of revenue", 330, xt="%"))
    with c2:
        r = RANK_TTM.sort_values("revenue_aed")
        fig = go.Figure(go.Bar(y=r.banner, x=r.revenue_aed, orientation="h", marker_color=[BC[b] for b in r.banner],
                               text=[f"#{k} · {aed(v)}" for k, v in zip(r.rank_revenue, r.revenue_aed)], textposition="outside",
                               hovertemplate="%{y}: AED %{x:,.0f}<extra></extra>"))
        fig.update_xaxes(range=[0, r.revenue_aed.max() * 1.32])
        show(style(fig, "Eight banners — recorded revenue, 2025-Q4 → 2026-Q3 (like-for-like)", 330, xt="AED", legend_bottom=False))
    # risks & opportunities — computed from artifacts
    rules = csv("rules_all.csv"); bo = csv("bundle_offers.csv"); mig = csv("migration_candidates.csv"); dd = csv("discount_dependency.csv", index_col=0)
    hi = CM[CM.retention_action.str.startswith("Retain")]; lapsed = CM[CM.churned == 1]
    top_rule = rules.iloc[0] if len(rules) else None
    dep = dd.loc[[i for i in dd.index if i.startswith("Discount-only") or i.startswith("Discount-dependent")]]
    st.markdown("#### Principal risks and opportunities")
    a, b = st.columns(2)
    with a:
        callout(f"<b>▲ Retention risk.</b> {len(hi):,} active members are above the cost-justified 90-day risk threshold ({THR['operational_threshold']:.2f}), "
                f"{int((hi.churn_risk_next90d >= 0.5).sum()):,} of them at 0.50 or more (historical spend of the flagged group {aed(hi.monetary.sum())}); "
                f"{len(lapsed):,} members are already lapsed and are win-back targets.", "risk")
        callout(f"<b>▲ Blind spot.</b> Viva is {GAP['viva_share_common_window_pct']}% of recorded revenue since its data begin, yet its "
                f"{GAP['viva_identified_members']:,} Viva Rewards IDs (and {GAP['viva_anonymous_pct']}% anonymous baskets) cannot be linked to any Shukran member.", "risk")
        callout(f"<b>▲ Margin exposure.</b> {int(dep.members.sum()):,} members buy on discount at least 75% of the time "
                f"({dep.share_of_revenue_pct.sum():.1f}% of Shukran revenue). Actual margin impact is unmeasurable without discount amounts.", "risk")
    with b:
        if top_rule is not None:
            gen = rules[rules.affinity_type.str.startswith("banner")].sort_values("lift_adj_activity_household", ascending=False) if "affinity_type" in rules else rules.iloc[:0]
            hh = rules[rules.affinity_type.str.startswith("household")] if "affinity_type" in rules else rules.iloc[:0]
            if len(gen):
                g0 = gen.iloc[0]; elig = bo[bo.rule == f"{g0.antecedent} → {g0.consequent}"].eligible_members.sum()
                txt = (f"<b>● Bundle.</b> Strongest affinity that survives allowing for activity and household: <b>{g0.antecedent} → {g0.consequent}</b> "
                       f"(adjusted lift {g0.lift_adj_activity_household:.2f}); {int(elig):,} members shop the first but not yet the second.")
            else:
                elig = bo[bo.rule == f"{top_rule.antecedent} → {top_rule.consequent}"].eligible_members.sum()
                txt = (f"<b>● Bundle.</b> Strongest affinity: <b>{top_rule.antecedent} → {top_rule.consequent}</b> (lift {top_rule.lift:.2f}); "
                       f"{int(elig):,} members shop the first but not yet the second.")
            if len(hh):
                h0 = hh.sort_values("lift", ascending=False).iloc[0]
                txt += (f" Higher raw lifts such as {h0.antecedent} → {h0.consequent} ({h0.lift:.2f}) are household-driven: target families who do not yet shop "
                        "the destination banner rather than shoppers of the other banner.")
            callout(txt, "good")
        act_m = mig[mig.churned == 0]
        callout(f"<b>● Upsell.</b> {int((act_m.pair == 'Homebox → Home Centre').sum()):,} active Homebox shoppers and "
                f"{int((act_m.pair == 'Max → Splash').sum()):,} active Max shoppers behave like premium-tier customers elsewhere — candidates for a controlled premium-introduction test.", "good")
        sel = MS["churn"]
        oot = js("oot_summary.json"); oci = {r["model"]: r for r in oot["pr_auc_ci_test_members"]}
        if oot.get("transfers", False):
            callout(f"<b>● Early warning.</b> The churn model ({sel['selected_model']}) ranks lapse with test PR-AUC {sel['selected_test']['pr_auc']:.2f} "
                    f"(no-skill {sel['positive_rate']:.2f}) and keeps skill when trained on earlier periods ({oci['out-of-time']['point']:.2f}); still validate with a random control group.", "good")
        else:
            callout(f"<b>● Early warning — pilot first.</b> Within this period the churn model ({sel['selected_model']}) ranks lapse with test PR-AUC "
                    f"{sel['selected_test']['pr_auc']:.2f} (no-skill {sel['positive_rate']:.2f}), mostly by recognising lapses already under way. Trained on earlier periods it scores "
                    f"{oci['out-of-time']['point']:.2f}, so it has not yet been shown to transfer: use it in a monitored pilot with a random control group before scaling.", "")
    st.markdown("#### What the data cannot tell us")
    st.markdown("- **Profit** — no cost or margin fields; only assumption-based contribution scenarios are shown.\n"
                "- **Member savings and Shukran's causal effect** — `discount_used` is a yes/no flag and there is no non-member comparison group.\n"
                "- **Churn timing** — the lapse flag has no date; departures by quarter are unknown.\n"
                + ("- **Whether the churn model works next quarter** — the out-of-time test found no transfer from earlier periods; only a monitored pilot can show it.\n"
                   if not js("oot_summary.json").get("transfers", False) else "") +
                "- **Viva customers** — no shared key; the Key Ring proposal (Methodology page) describes how to fix this with consent.")
    filter_note(F, "fin")


# ================================================================================================================
# B · Group financials
# ================================================================================================================
def page_financials():
    hero(compact=True)
    F = st.session_state["F"]
    section("Group financials", "Recorded revenue from the supplied transactions, by calendar period. Partial periods are marked and never used for growth rates.", "measured")
    filter_note(F, "fin")
    allx = L.combined_transactions(TX, VIVA)
    w = allx[(allx.date >= F["p0"]) & (allx.date <= F["p1"]) & allx.banner.isin(F["banners"])]
    if w.empty:
        st.warning("No transactions in the selected period/banners."); return
    s_rev, v_rev = w[w.banner != "Viva"].amount.sum(), w[w.banner == "Viva"].amount.sum()
    cards([("Recorded revenue · selection", aed(w.amount.sum()), f"{len(w):,} transactions"),
           ("Shukran banners", aed(s_rev), f"{(w.banner != 'Viva').sum():,} transactions"),
           ("Viva", aed(v_rev), "separate identity system"),
           ("Average ticket", aed(w.amount.mean(), 2), "revenue ÷ transactions"),
           ("Discounted transactions", f"{w.discount_used.mean()*100:.1f}%" if w.discount_used.notna().any() else "n/a", "Shukran banners only (flag)")])
    st.write("")
    split = st.radio("Show revenue by", ["Source", "Banner"], horizontal=True, key="fin_split")
    key = "source" if split == "Source" else "banner"
    cmap = {"Shukran (7 banners)": SERIES[0], "Viva (separate IDs)": BC["Viva"]} if split == "Source" else BC
    quarters = sorted(w.date.dt.to_period("Q").astype(str).unique())
    fig = go.Figure()
    for k, sub in w.groupby(key):
        cov = coverage(sub, F=F)                         # partial flags vs the source's collection window
        full, part = cov[~cov.partial], cov[cov.partial]
        fig.add_scatter(x=full.period, y=full.revenue, name=k, mode="lines+markers", line=dict(color=cmap.get(k, SLATE), width=2.2),
                        marker=dict(size=7, color=cmap.get(k, SLATE)), legendgroup=k, hovertemplate=f"{k}<br>%{{x}}: AED %{{y:,.0f}}<extra></extra>")
        if len(part):
            fig.add_scatter(x=part.period, y=part.revenue, name=f"{k} · partial quarter", mode="markers", legendgroup=k, showlegend=False,
                            marker=dict(size=11, symbol="circle-open", color=cmap.get(k, SLATE), line=dict(width=2)),
                            customdata=part.coverage_pct, hovertemplate=f"{k} (PARTIAL: %{{customdata:.0f}}% of quarter covered)<br>%{{x}}: AED %{{y:,.0f}}<extra></extra>")
    fig.update_xaxes(categoryorder="array", categoryarray=quarters)
    show(style(fig, "Quarterly recorded revenue — line = complete quarters; hollow marker = partial quarter (not joined to the trend)", 400, xt="Calendar quarter", yt="AED"))
    c1, c2 = st.columns(2)
    with c1:
        fq = csv("fin_quarterly.csv")
        fq = fq[["source", "period", "revenue", "transactions", "avg_ticket", "active_customers", "coverage_pct", "partial", "qoq_growth_pct", "yoy_same_quarter_pct"]]
        st.markdown("**Quarter table (full data)** — growth only between complete quarters")
        table(fq.round(1), height=320, column_config={"revenue": st.column_config.NumberColumn("revenue (AED)", format="localized"),
                                                       "partial": st.column_config.CheckboxColumn("partial?")})
    with c2:
        st.markdown("**Calendar years (coverage stated)**")
        table(csv("fin_annual.csv").round(0), height=170)
        st.markdown("**Like-for-like comparisons**")
        table(csv("fin_like_for_like.csv").round(1), height=150)
    sh = w[w.banner != "Viva"]
    if len(sh):
        tq = coverage(sh, F=F)
        tf, tp_ = tq[~tq.partial], tq[tq.partial]
        c3, c4 = st.columns(2)
        with c3:
            fig = go.Figure(go.Bar(x=tf.period, y=tf.transactions, marker_color=SERIES[0], name="complete quarter", hovertemplate="%{x}: %{y:,} transactions<extra></extra>"))
            if len(tp_):
                fig.add_bar(x=tp_.period, y=tp_.transactions, name="partial quarter", marker=dict(color="rgba(0,0,0,0)", line=dict(color=SERIES[0], width=1.5)),
                            customdata=tp_.coverage_pct, hovertemplate="%{x} (PARTIAL: %{customdata:.0f}% covered): %{y:,} transactions<extra></extra>")
            fig.update_xaxes(categoryorder="array", categoryarray=tq.period.tolist())
            show(style(fig, "Shukran transactions per quarter (outlined = partial quarter)", 320, xt="Quarter", yt="Transactions"))
        with c4:
            fig = go.Figure(go.Scatter(x=tf.period, y=tf.avg_ticket, mode="lines+markers", name="complete quarter", line=dict(color=SERIES[2], width=2.2),
                                       hovertemplate="%{x}: AED %{y:,.0f}<extra></extra>"))
            if len(tp_):
                fig.add_scatter(x=tp_.period, y=tp_.avg_ticket, mode="markers", name="partial quarter", marker=dict(size=11, symbol="circle-open", color=SERIES[2], line=dict(width=2)),
                                customdata=tp_.coverage_pct, hovertemplate="%{x} (PARTIAL: %{customdata:.0f}% covered): AED %{y:,.0f}<extra></extra>")
            fig.update_xaxes(categoryorder="array", categoryarray=tq.period.tolist())
            show(style(fig, "Shukran average ticket per quarter (hollow = partial quarter)", 320, xt="Quarter", yt="AED per transaction"))
    st.divider()
    section("Revenue is measured — profit is not", "No cost, margin or expense fields exist in the supplied data. Enter margin assumptions to see a contribution proxy and how the ranking would change.", "scenario")
    win = st.radio("Revenue window for the scenario", ["Like-for-like 2025-Q4 → 2026-Q3", "Sidebar period"], horizontal=True, key="scen_win")
    base = RANK_TTM if win.startswith("Like") else ranking(F["p0"], F["p1"])
    default = pd.DataFrame({"banner": L.ALL_BANNERS, "assumed_margin_pct": [30.0] * 8})
    ed = st.data_editor(default, key="margins", hide_index=True, **W_EDITOR,
                        column_config={"assumed_margin_pct": st.column_config.NumberColumn("assumed margin %", min_value=0.0, max_value=100.0, step=1.0)},
                        disabled=["banner"])
    sc = L.contribution_scenario(base, dict(zip(ed.banner, ed.assumed_margin_pct)))
    c5, c6 = st.columns([1.2, 1])
    with c5:
        table(sc[["banner", "revenue_aed", "assumed_margin_pct", "contribution_proxy_aed", "rank_revenue", "rank_contribution_proxy", "rank_change_vs_revenue"]].round({"revenue_aed": 0, "contribution_proxy_aed": 0}), height=320,
              column_config={"banner": st.column_config.TextColumn("Banner"), "revenue_aed": st.column_config.NumberColumn("Revenue (AED)", format="localized"),
                             "assumed_margin_pct": st.column_config.NumberColumn("Margin %", format="%.0f"),
                             "contribution_proxy_aed": st.column_config.NumberColumn("Proxy (AED)", format="localized"),
                             "rank_revenue": st.column_config.NumberColumn("Rank · rev"), "rank_contribution_proxy": st.column_config.NumberColumn("Rank · proxy"),
                             "rank_change_vs_revenue": st.column_config.NumberColumn("Δ rank", format="%+d")})
    with c6:
        s2 = sc.sort_values("contribution_proxy_aed")
        show(style(go.Figure(go.Bar(y=s2.banner, x=s2.contribution_proxy_aed, orientation="h", marker_color=[BC[b] for b in s2.banner],
                                    text=[f"#{r} ({'+' if d > 0 else ''}{d} vs revenue rank)" if d else f"#{r}" for r, d in zip(s2.rank_contribution_proxy, s2.rank_change_vs_revenue)],
                                    textposition="outside")), "Contribution proxy (scenario)", 320, xt="AED (hypothetical)", legend_bottom=False).update_xaxes(range=[0, s2.contribution_proxy_aed.max() * 1.45]))
    callout("Contribution proxy = recorded revenue × your assumed margin. It is <b>not actual profit</b>; with equal margins the ranking cannot differ from the revenue ranking.", "")


# ================================================================================================================
# C · Eight-banner performance
# ================================================================================================================
def page_banners():
    hero(compact=True)
    F = st.session_state["F"]
    section("Eight-banner performance", "Numerical ranking of all eight banners. Viva's data begin on 28 Aug 2025, so the like-for-like window is the default.", "measured")
    win = st.radio("Ranking window", ["Like-for-like: 2025-Q4 → 2026-Q3 (all banners complete)", "Full available period (each banner's own coverage)", "Sidebar period"],
                   horizontal=False, key="rank_win")
    r = RANK_TTM if win.startswith("Like") else (RANK_FULL if win.startswith("Full") else ranking(F["p0"], F["p1"]))
    r = r[r.banner.isin(F["banners"])].copy()
    if r.empty:
        st.warning("No banners selected."); return
    show_cols = ["rank_revenue", "banner", "positioning", "revenue_aed", "revenue_share_pct", "rank_transactions", "transactions", "rank_avg_ticket", "avg_ticket_aed", "reach", "reach_basis", "discount_rate"]
    table(r.sort_values("rank_revenue")[show_cols].round(2), column_config={
        "rank_revenue": st.column_config.NumberColumn("Rank · revenue"), "revenue_aed": st.column_config.NumberColumn("Revenue (AED)", format="localized"),
        "revenue_share_pct": st.column_config.NumberColumn("Share %", format="%.1f"), "rank_transactions": st.column_config.NumberColumn("Rank · volume"),
        "rank_avg_ticket": st.column_config.NumberColumn("Rank · ticket"), "avg_ticket_aed": st.column_config.NumberColumn("Avg ticket (AED)", format="%.0f"),
        "discount_rate": st.column_config.NumberColumn("Discounted tx share", format="%.2f")})
    with st.expander("How each rank is calculated"):
        st.markdown("- **Revenue** = Σ `amount` per banner in the window; **share** = banner revenue ÷ Σ eight banners.\n- **Transactions** = row count; **average ticket** = revenue ÷ transactions.\n"
                    "- **Reach** = distinct Shukran IDs for the seven Shukran banners; for Viva it is distinct Viva Rewards IDs (anonymous baskets excluded) — **not comparable** with Shukran counts.\n"
                    "- Ranks use the *minimum* method (ties share the better rank). Rank 1 = highest.")
    c1, c2, c3 = st.columns(3)
    for col, metric, ttl, xt in [(c1, "revenue_aed", "Recorded revenue", "AED"), (c2, "transactions", "Transactions", "Transactions"), (c3, "avg_ticket_aed", "Average ticket", "AED")]:
        s = r.sort_values(metric)
        fig = go.Figure(go.Bar(y=s.banner, x=s[metric], orientation="h", marker_color=[BC[b] for b in s.banner],
                               text=[aed(v) if "aed" in metric else f"{v:,.0f}" for v in s[metric]], textposition="outside"))
        fig.update_xaxes(range=[0, s[metric].max() * 1.5])
        with col:
            show(style(fig, ttl, 340, xt=xt, legend_bottom=False))
    st.markdown("#### Monthly trend")
    pick = st.multiselect("Banners to plot (up to four keeps the chart readable)", L.ALL_BANNERS, default=["Home Centre", "Centrepoint", "Viva"], max_selections=4, key="trend_pick")
    allx = L.combined_transactions(TX, VIVA)
    fig = monthly_figure(allx[allx.banner.isin(pick)], "banner", BC)
    show(style(fig, "Monthly recorded revenue — complete months on the line; hollow = partial month", 380, xt="Month", yt="AED"))
    callout("Revenue growth in this sample partly reflects more of the 3,000 sample members becoming active over time; it is not a like-for-like Group growth rate.")


# ================================================================================================================
# D / E · Pair pages
# ================================================================================================================
def pair_page(premium, value, key, label):
    hero(compact=True)
    section(f"{premium} vs {value}", f"Same business, different customer tier. Descriptive comparison of recorded figures, then the leakage-free routing model ({label}).", "measured")
    pc = csv("pair_comparison.csv", index_col=0).loc[[premium, value]]
    c = st.columns(2)
    for col, ban in zip(c, (premium, value)):
        p = pc.loc[ban]
        with col:
            st.markdown(f"<div class='card' style='border-top:4px solid {BC[ban]}'><div class='l'>{L.BANNER_POSITION[ban]}</div><div class='v'>{ban}</div>"
                        f"<div class='s'>Revenue (full) <b>{aed(p.revenue_full_aed)}</b> · TTM {aed(p.revenue_ttm_aed)} · share {p.revenue_share_full_pct:.1f}%<br>"
                        f"{int(p.transactions_full):,} transactions · avg ticket <b>{aed(p.avg_ticket_aed, 0)}</b> · discounted {p.discount_rate_pct:.1f}% of tx<br>"
                        f"{int(p.shoppers):,} shoppers · median total spend {aed(p.shopper_total_spend_median_aed)} · churn {p.shopper_churn_rate_pct:.1f}%</div></div>", unsafe_allow_html=True)
    hi = pc.revenue_full_aed.idxmax()
    callout(f"<b>{hi}</b> records the higher revenue ({aed(pc.revenue_full_aed.max())} vs {aed(pc.revenue_full_aed.min())}) with "
            f"{'fewer' if pc.loc[hi].transactions_full < pc.drop(hi).transactions_full.iloc[0] else 'more'} transactions — the difference is ticket size. "
            "These are descriptive associations, not causes.")
    q = TX[TX.banner.isin([premium, value])].assign(quarter=TX.date.dt.to_period("Q").astype(str))
    q = q[(q.quarter > "2024Q2") & (q.quarter < "2026Q4")].groupby(["quarter", "banner"]).amount.agg(["sum", "mean"]).reset_index()
    c1, c2 = st.columns(2)
    with c1:
        show(style(px.line(q, x="quarter", y="sum", color="banner", color_discrete_map=BC, markers=True), "Quarterly revenue (complete quarters)", 320, xt="Quarter", yt="AED"))
    with c2:
        show(style(px.line(q, x="quarter", y="mean", color="banner", color_discrete_map=BC, markers=True), "Average ticket (complete quarters)", 320, xt="Quarter", yt="AED per transaction"))
    st.markdown("**Shopper behaviour and profile**")
    labels = {"shoppers": "Shoppers (members)", "shopper_total_spend_median_aed": "Median total spend across all banners (AED)", "shopper_frequency_mean": "Transactions per shopper (all banners)",
              "shopper_discount_rate_pct": "Share of their transactions discounted (%)", "shopper_n_banners_mean": "Banners visited", "shopper_age_mean": "Average age",
              "shopper_has_kids_pct": "Households with kids (%)", "shopper_churn_rate_pct": "Churn rate (%)"}
    table(pc[list(labels)].rename(columns=labels).T.round(2).reset_index().rename(columns={"index": "measure"}))
    st.divider()
    section("Banner routing — which tier does a member belong to?", "Predicted from behaviour outside the category plus demographics; the category's own banner spend is never used.", "model")
    s = MS[key]; lb = csv(f"leaderboard_{key}.csv"); leak = csv(f"leakage_audit_{key}.csv")
    cards([("Selected model", s["selected_model"], "1-SE rule on CV ROC-AUC"), ("Test ROC-AUC", f"{s['selected_test']['roc_auc']:.3f}", "0.5 = coin flip"),
           ("Test accuracy", f"{s['selected_test']['accuracy']:.3f}", f"premium share {s['positive_rate']:.2f}"), ("Test F1", f"{s['selected_test']['f1']:.3f}", "premium = positive class")])
    st.write("")
    c1, c2 = st.columns([1.1, 1.15])
    with c1:
        lk = leak.copy(); lk["feature_set"] = lk.feature_set.str.replace(r"^\d ", "", regex=True).str.replace(r":\s*", ":<br>", regex=True).str.replace(r" \((permitted|forbidden)\)", r"<br>(\1)", regex=True)
        fig = px.bar(lk, y="feature_set", x="roc_auc", color="model", barmode="group", orientation="h", color_discrete_sequence=[SERIES[0], SERIES[2]],
                     text=lk.roc_auc.map(lambda v: f"{v:.2f}"))
        fig.update_traces(textposition="outside")
        fig.add_vline(x=0.5, line_color="#898781", line_width=1, line_dash="dot", annotation_text="chance", annotation_position="bottom right")
        show(style(fig, "Leakage audit — score of each feature set", 420, xt="CV ROC-AUC", yt="").update_xaxes(range=[0, 1.18], dtick=0.25))
    with c2:
        imp = csv(f"perm_importance_{key}.csv").head(10)[::-1]
        fig = go.Figure(go.Bar(y=imp.feature.map(feat_label), x=imp.importance_mean, orientation="h", marker_color=SERIES[0],
                               error_x=dict(type="data", array=imp.importance_std, color=SLATE)))
        show(style(fig, "Signals that reveal the tier (permutation importance, test set)", 420, xt="Drop in ROC-AUC when shuffled", legend_bottom=False).update_xaxes(tickangle=0, nticks=5))
    callout("Direct or indirect use of the category's own spend gives a near-perfect but meaningless score. With leakage removed the signal is moderate, and demographics alone carry almost none: "
            "<b>a member's price tier is revealed by how they shop elsewhere — ticket sizes, discount use and their tier in the other category — not by who they are.</b>")
    st.markdown("#### Route a member")
    feats = csv(f"routing_features_{key}.csv", index_col=0)
    pref = "home" if key == "home" else "fashion"
    pool = filt_customers(st.session_state["F"])
    cid = st.selectbox("Member (type to search)", pool.index.tolist(), key=f"route_{key}")
    if cid:
        r = CM.loc[cid]; p = r[f"{pref}_p_premium"]
        lab = r[f"{pref}_label"]
        actual = (premium if lab == 1 else value) if pd.notna(lab) else f"has not shopped this category yet"
        pred = premium if p >= 0.5 else value
        cards([("P(premium tier)", f"{p:.2f}", r[f"{pref}_scored_by"]), ("Model suggests", pred, "threshold 0.5"), ("Observed banner", actual, "from transactions"),
               ("Segment", r.segment, f"value tier {r.value_tier}")])
        num_feats = [f for f in csv(f"perm_importance_{key}.csv").feature if f in feats.columns][:7]
        labels = CM[f"{pref}_label"]
        sig = pd.DataFrame({"this member": feats.loc[cid, num_feats], f"{premium} shoppers (mean)": feats.loc[labels == 1, num_feats].mean(),
                            f"{value} shoppers (mean)": feats.loc[labels == 0, num_feats].mean()})
        sig["closer to"] = np.where((sig["this member"] - sig.iloc[:, 1]).abs() < (sig["this member"] - sig.iloc[:, 2]).abs(), premium, value)
        sig.loc[sig["this member"].isna(), "closer to"] = "no data (imputed)"
        sig.index = [feat_label(f) for f in sig.index]
        table(sig.round(3).reset_index().rename(columns={"index": "signal (non-target behaviour)"}))
    st.markdown("#### Migration / upsell candidates")
    mig = csv("migration_candidates.csv"); mig = mig[(mig.current_banner == value) & mig.customer_id.isin(filt_customers(st.session_state["F"]).index)]
    only_active = st.toggle("Active members only", value=True, key=f"mig_active_{key}")
    if only_active:
        mig = mig[mig.churned == 0]
    st.caption(f"{len(mig):,} {value} shoppers in the top decile of the premium-likeness score — **candidates for a controlled test, not predicted conversions.**")
    table(mig.sort_values("migration_score", ascending=False).round(3), height=300)
    st.download_button("Download candidates (CSV)", mig.to_csv(index=False).encode(), f"migration_{value}_to_{premium}.csv".replace(" ", "_"), "text/csv", key=f"dl_mig_{key}")
    with st.expander("Model leaderboard for this target"):
        table(lb.sort_values("cv_roc_auc", ascending=False)[["model", "family", "cv_roc_auc", "cv_accuracy", "cv_f1", "test_roc_auc", "test_accuracy", "status"]].round(3))
        show(cm_figure(key))


def page_home():
    pair_page("Home Centre", "Homebox", "home", "Home Centre vs Homebox")


def page_fashion():
    pair_page("Splash", "Max", "fashion", "Splash vs Max")


# ================================================================================================================
# F · Other banners
# ================================================================================================================
def page_other():
    hero(compact=True)
    section("Other banners", "Babyshop, Centrepoint, Funcity and Viva, each on its own terms (not forced into premium/value pairs).", "measured")
    ob = csv("other_banners.csv", index_col=0)
    tabs = st.tabs(["Babyshop", "Centrepoint", "Funcity", "Viva"])
    for t, ban in zip(tabs, ["Babyshop", "Centrepoint", "Funcity", "Viva"]):
        with t:
            p = ob.loc[ban]
            cards([("Revenue · full period", aed(p.revenue_full_aed), L.BANNER_POSITION[ban]), ("Revenue · 2025-Q4→2026-Q3", aed(p.revenue_ttm_aed), "like-for-like"),
                   ("Transactions", f"{int(p.transactions):,}", f"avg ticket {aed(p.avg_ticket_aed, 0)}"), ("Reach", p.reach.split(" Shukran")[0].split(" (")[0] if ban != "Viva" else f"{GAP['viva_identified_members']:,} VR IDs",
                    ("Shukran members" + (f" ({p.reach.split('(')[1]}" if "(" in p.reach else "")) if ban != "Viva" else "+ anonymous baskets; not comparable")])
            st.write("")
            src = VIVA.assign(banner="Viva") if ban == "Viva" else TX[TX.banner == ban]
            c1, c2 = st.columns(2)
            with c1:
                show(style(monthly_figure(src, "banner", BC), f"{ban} — monthly revenue (hollow = partial month)", 300, xt="Month", yt="AED", legend_bottom=False))
            with c2:
                if ban == "Viva":
                    vc = VIVA.groupby("category").amount.sum().sort_values()
                    show(style(go.Figure(go.Bar(y=vc.index, x=vc.values, orientation="h", marker_color=BC["Viva"], text=[aed(v) for v in vc.values], textposition="outside")),
                               "Viva revenue by category", 300, xt="AED", legend_bottom=False).update_xaxes(range=[0, vc.max() * 1.35]))
                else:
                    sh = CM[CM[f"bspend_{ban}"] > 0]
                    prof = pd.DataFrame({"shoppers": [sh.has_kids.mean() * 100, sh.churned.mean() * 100, (sh.n_banners >= 4).mean() * 100],
                                         "all members": [CM.has_kids.mean() * 100, CM.churned.mean() * 100, (CM.n_banners >= 4).mean() * 100]},
                                        index=["% with kids", "% churned", "% visiting 4+ banners"])
                    fig = go.Figure([go.Bar(y=prof.index, x=prof.shoppers, name=f"{ban} shoppers", orientation="h", marker_color=BC[ban]),
                                     go.Bar(y=prof.index, x=prof["all members"], name="all members", orientation="h", marker_color="#B9B4A8")])
                    show(style(fig.update_layout(barmode="group"), f"{ban} shoppers vs all members", 300, xt="%"))
            if ban == "Viva":
                callout(f"Viva is the highest-frequency, lowest-ticket banner (avg {aed(GAP['viva_avg_ticket_aed'], 2)}). {GAP['viva_anonymous_pct']}% of its baskets are anonymous and none can be linked to Shukran. "
                        "No discount flag or customer attributes exist for Viva.", "risk")


# ================================================================================================================
# G · Segmentation & PCA
# ================================================================================================================
def style_view(F):
    """Second, independent segmentation on how members shop (no banner mix) and how it relates to the primary segments."""
    sc = js("style_choice.json"); sp = csv("style_profiles.csv", index_col=0); ct = csv("style_vs_banner_crosstab.csv", index_col=0)
    st.caption(f"{sc['algorithm']}, K = {sc['k']} on {', '.join(sc['features'])} — banner shares, concentration, recency and trend deliberately excluded. "
               "Descriptive only: never used as a model input.")
    styles = sp.index.tolist(); scol = {n: SERIES[(i + 3) % len(SERIES)] for i, n in enumerate(styles)}
    cols = st.columns(len(styles))
    for c, n in zip(cols, styles):
        r = sp.loc[n]
        c.markdown(f"<div class='card' style='border-top:4px solid {scol[n]}'><div class='l'>Shopping style</div><div class='v' style='font-size:1.05rem'>{n}</div>"
                   f"<div class='s'><b>{int(r.members):,}</b> members ({r.share_of_members_pct:.0f}%) · <b>{r.share_of_revenue_pct:.0f}%</b> of revenue<br>"
                   f"ticket {aed(r.avg_ticket_aed)} · discount {r.discount_rate_pct:.0f}% · {r.banners_mean:.1f} banners<br>churn {r.churn_rate_pct:.0f}% · kids {r.has_kids_pct:.0f}%</div></div>",
                   unsafe_allow_html=True)
    cards([("Agreement with banner-mix segments", f"ARI {sc['ari_vs_banner_segments']:.2f}", f"NMI {sc['nmi_vs_banner_segments']:.2f} (1 = identical, 0 = unrelated)"),
           ("In the typical style of their segment", f"{sc['share_in_modal_style_of_their_banner_segment']*100:.0f}%", "the rest shop differently from their banner group"),
           ("Churn differs by style?", "No" if sc["churn_by_style_chi2_p"] >= 0.05 else "Yes", f"chi-square p = {sc['churn_by_style_chi2_p']:.3f}")])
    c1, c2 = st.columns([1.15, 1])
    with c1:
        d = filt_customers(F)
        fig = go.Figure()
        for n in styles:
            g = d[d.style_segment == n]
            fig.add_scattergl(x=g.PC1, y=g.PC2, mode="markers", name=n, marker=dict(size=5, color=scol[n], opacity=0.6),
                              customdata=np.stack([g.index, g.segment], axis=-1), hovertemplate="%{customdata[0]}<br>%{customdata[1]}<extra>" + n + "</extra>")
        pe = csv("pca_explained.csv")
        show(style(fig, "Shopping styles on the same PC1–PC2 map (filtered)", 480, xt=f"PC1 ({pe.explained_var_pct[0]:.1f}%)", yt=f"PC2 ({pe.explained_var_pct[1]:.1f}%)", legend_below=True))
    with c2:
        pct = ct.div(ct.sum(axis=1), axis=0) * 100
        pct.index = ["<br>".join(textwrap.wrap(i.replace(" shoppers", ""), 18)) for i in pct.index]
        pct.columns = ["<br>".join(textwrap.wrap(c_.replace(" shoppers", ""), 12)) for c_ in pct.columns]
        fig = px.imshow(pct.round(0), text_auto=".0f", color_continuous_scale="Blues", zmin=0, zmax=100, aspect="auto", labels=dict(color="% of row", x="", y=""))
        fig.update_xaxes(tickangle=0)
        show(style(fig, "Banner-mix segment (rows) × shopping style (columns), row %", 480, legend_bottom=False))
    zc = [c for c in sp.columns if c.startswith("z_")]
    table(sp.drop(columns=zc).reset_index().rename(columns={"style_segment": "Shopping style"}).round(1))
    callout("Two lenses on the same members: the banner-mix segments say <i>where</i> members shop (and drive bundles and targeting); the shopping-style view says <i>how</i> they shop. "
            "Members whose style differs from their banner group — for example value-home shoppers with a big-ticket style — are the natural premium-introduction prospects; "
            "the Acquisition & migration page cross-checks them against the routing model.")


def page_segments():
    hero(compact=True)
    F = st.session_state["F"]
    cc = js("cluster_choice.json")
    section("Customer segmentation and PCA", f"{cc['algorithm']} on {cc['representation']}, K = {cc['k']} — chosen by a like-for-like silhouette with elbow, size and bootstrap-stability checks.", "model")
    filter_note(F)
    view = st.radio("Segmentation view", ["Banner-mix segments (primary)", "Shopping-style segments (second view)"], horizontal=True, key="seg_view")
    if view.startswith("Shopping"):
        style_view(F)
        return
    segd = csv("segment_dashboard.csv", index_col=0).reindex(SEG_ORDER)
    cols = st.columns(len(SEG_ORDER))
    for c, s in zip(cols, SEG_ORDER):
        p = segd.loc[s]
        c.markdown(f"<div class='card' style='border-top:4px solid {SEG_COLORS[s]}'><div class='l'>Segment</div><div class='v' style='font-size:1.05rem'>{s}</div>"
                   f"<div class='s'><b>{int(p.members):,}</b> members ({p.share_of_members_pct:.0f}%) · <b>{p.share_of_revenue_pct:.0f}%</b> of revenue<br>"
                   f"avg ticket {aed(p.avg_ticket_aed)} · discount {p.discount_rate_pct:.0f}%<br>churn {p.churn_rate_pct:.0f}% · avg 90-day risk {p.avg_risk_next90d:.2f}</div></div>", unsafe_allow_html=True)
    sel = st.selectbox("Inspect a segment", SEG_ORDER, key="seg_inspect")
    callout(f"<b>Why this name:</b> {SEGP.set_index('segment').loc[sel, 'evidence']}<br><b>Recommended bundle:</b> {segd.loc[sel, 'recommended_bundle']}")
    c1, c2 = st.columns([1.15, 1])
    with c1:
        d = filt_customers(F)
        fig = go.Figure()
        for s in SEG_ORDER:
            g = d[d.segment == s]
            fig.add_scattergl(x=g.PC1, y=g.PC2, mode="markers", name=s, marker=dict(size=5, color=SEG_COLORS[s], symbol=SEG_SYMBOLS[s], opacity=0.6),
                              customdata=np.stack([g.index, g.monetary], axis=-1), hovertemplate="%{customdata[0]}<br>spend AED %{customdata[1]:,.0f}<extra>" + s + "</extra>")
        pe = csv("pca_explained.csv")
        show(style(fig, "Members in PC1–PC2 space (filtered)", 500, xt=f"PC1 ({pe.explained_var_pct[0]:.1f}% of variance)", yt=f"PC2 ({pe.explained_var_pct[1]:.1f}%)", legend_below=True))
    with c2:
        reach = SEGP.set_index("segment").loc[SEG_ORDER, [f"reach_{b}_pct" for b in L.CORE_BANNERS]]
        reach.columns = L.CORE_BANNERS
        short = ["<br>".join(textwrap.wrap(s.replace(" shoppers", ""), 11)) for s in SEG_ORDER]
        rt = reach.T.round(0); rt.columns = short
        fig = px.imshow(rt, text_auto=".0f", color_continuous_scale="Blues", zmin=0, zmax=100, aspect="auto", labels=dict(color="% reach", x="", y=""))
        fig.update_xaxes(tickangle=0)
        show(style(fig, "Banner reach by segment (% of members)", 500, legend_bottom=False))
    prof_cols = {"recency_days_mean": "recency (days)", "frequency_mean": "transactions", "monetary_mean_aed": "spend (AED)", "avg_ticket_aed": "avg ticket (AED)",
                 "discount_rate_pct": "discount %", "n_banners_mean": "banners", "has_kids_pct": "% with kids", "age_mean": "age", "churn_rate_pct": "churn rate %"}
    prof_cols = {k: v for k, v in prof_cols.items() if k in SEGP.columns}
    pt = SEGP.set_index("segment").loc[SEG_ORDER, list(prof_cols)].rename(columns=prof_cols).round(1)
    st.markdown("**Profile comparison (all members in each segment)**")
    table(pt.reset_index().rename(columns={"segment": "Segment"}))
    st.divider()
    pe, ld = csv("pca_explained.csv"), csv("pca_loadings.csv", index_col=0)
    c3, c4 = st.columns([1, 1.5])
    with c3:
        fig = go.Figure([go.Bar(x=pe.PC, y=pe.explained_var_pct, name="individual", marker_color=SERIES[0]),
                         go.Scatter(x=pe.PC, y=pe.cumulative_pct, name="cumulative", mode="lines+markers", line=dict(color=NAVY, width=2))])
        fig.add_hline(y=90, line_color=STATUS["critical"], line_width=1, annotation_text=f"90% → {cc['n_pcs']} PCs", annotation_position="bottom right")
        show(style(fig, "Scree and cumulative explained variance", 340, xt="Component", yt="% of variance"))
    with c4:
        ldl = ld.iloc[:, :6].round(2); ldl.index = [feat_label(f) for f in ldl.index]
        fig = px.imshow(ldl, text_auto=True, color_continuous_scale="RdBu", zmin=-0.7, zmax=0.7, aspect="auto", labels=dict(x="", y="", color="loading"))
        fig.update_traces(textfont_size=11)
        show(style(fig, "Loadings PC1–PC6 (what each axis means)", 520, legend_bottom=False))
    with st.expander("Cluster-selection evidence (elbow, silhouette, Euclidean vs Manhattan, raw vs PCA)"):
        g = csv("cluster_grid.csv"); g["solution"] = g.algorithm + " · " + g.representation.str.split(" ").str[0]
        c5, c6 = st.columns(2)
        with c5:
            ycol = "sil_common" if "sil_common" in g.columns else "silhouette"
            show(style(px.line(g, x="k", y=ycol, color="solution", markers=True, line_dash="algorithm", color_discrete_sequence=SERIES),
                       "Silhouette by K — common basis (raw space, mean of Euclidean & Manhattan)", 460, xt="K", yt="Mean silhouette", legend_below=True))
        with c6:
            show(style(px.line(g, x="k", y="min_cluster_share_pct", color="solution", markers=True, line_dash="algorithm", color_discrete_sequence=SERIES).add_hline(y=5, line_color=STATUS["critical"]),
                       "Smallest segment's share (5% floor)", 460, xt="K", yt="% of members", legend_below=True))
        st.markdown(f"**Decision:** {cc['rule']} → **{cc['algorithm']} on {cc['representation']}, K = {cc['k']}** (common-basis silhouette {cc['silhouette']:.3f}"
                    + (f" — Euclidean {cc['silhouette_common_euclid']:.3f}, Manhattan {cc['silhouette_common_manhattan']:.3f}" if "silhouette_common_euclid" in cc else "")
                    + f"; bootstrap ARI mean {cc['bootstrap_ari_mean']:.2f}, min {cc['bootstrap_ari_min']:.2f}).")
        if "own_metric_ranking_would_pick" in cc:
            st.caption(f"Each algorithm's own-metric silhouette is not comparable across Euclidean and Manhattan distance; ranking that way would have picked "
                       f"{cc['own_metric_ranking_would_pick']}.")
        table(csv("cluster_agreement_ari.csv").rename(columns={"Unnamed: 0": "solution"}))


# ================================================================================================================
# H · Churn & retention
# ================================================================================================================
def churn_trust():
    """Out-of-time test, calibration, confidence intervals and the data-window feature — the evidence for how far to trust the churn scores."""
    st.markdown("#### How much can these scores be trusted?")
    t1, t2, t3, t4 = st.tabs(["Out-of-time test", "Calibration", "Uncertainty (95% CI)", "Data-window feature"])
    with t1:
        o = js("oot_summary.json"); ci = pd.DataFrame(o["pr_auc_ci_test_members"]).set_index("model"); tm = o["churn_timing"]; pr = pd.DataFrame(o["predictability_by_date"])
        cards([("In-time test PR-AUC", f"{ci.loc['in-time', 'point']:.2f}", f"95% CI {ci.loc['in-time', 'ci_low']:.2f}–{ci.loc['in-time', 'ci_high']:.2f}"),
               ("Out-of-time PR-AUC", f"{ci.loc['out-of-time', 'point']:.2f}", f"95% CI {ci.loc['out-of-time', 'ci_low']:.2f}–{ci.loc['out-of-time', 'ci_high']:.2f}"),
               ("No-skill PR-AUC", f"{MS['churn']['positive_rate']:.2f}", "= churn prevalence"),
               ("Median churner: days since last purchase", f"{tm['median_days_since_last_purchase_at_reference']:.0f}", "at the model's reference date"),
               ("Earlier 90-day gap → churn flag", f"{tm['p_flag_given_earlier_90d_gap']*100:.0f}%", f"vs {tm['p_flag_base']*100:.0f}% of all members")])
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(x=["Trained in-time", "Trained on earlier periods"], y=ci.point.values, marker_color=[SERIES[0], SERIES[1]],
                                   error_y=dict(type="data", symmetric=False, array=(ci.ci_high - ci.point).values, arrayminus=(ci.point - ci.ci_low).values, color=SLATE),
                                   text=[f"{v:.2f}" for v in ci.point.values], textposition="inside", insidetextanchor="start",   # value at the bar base, clear of the CI whisker
                                   textfont=dict(color="white")))
            fig.add_hline(y=MS["churn"]["positive_rate"], line_dash="dot", line_color="#898781")
            show(style(fig, "PR-AUC on the same test members (95% CI)", 340, yt=f"PR-AUC (dotted = no skill, {MS['churn']['positive_rate']:.2f})", legend_bottom=False).update_yaxes(range=[0, 1]))
        with c2:
            fig = go.Figure(go.Scatter(x=pr.feature_date, y=pr.cv_roc_auc, mode="lines+markers", line=dict(color=SERIES[0], width=2.2), marker=dict(size=9),
                                       error_y=dict(type="data", array=pr.cv_roc_auc_std, color=SLATE), name="CV ROC-AUC"))
            fig.add_hline(y=0.5, line_dash="dot", line_color="#898781", annotation_text="chance", annotation_position="bottom right")
            show(style(fig, "How predictable is 'no purchase in the next 90 days', by date?", 340, xt="Feature date", yt="CV ROC-AUC", legend_bottom=False).update_yaxes(range=[0.4, 1]))
        transfers = bool(o.get("transfers", ci.loc["out-of-time", "ci_low"] > MS["churn"]["positive_rate"] + 0.05))
        callout(("The relationship carries over to a new period." if transfers else
                 "<b>The model does not transfer across periods.</b> Trained on earlier inactivity spells, it has almost no skill on the recorded churn status. Earlier 90-day gaps were common and mostly temporary, "
                 "while the recorded churn is a recent, concentrated lapse. ") +
                " Use the forward scores as a <b>ranking hypothesis</b>: contact a random half of the flagged members, compare lapse rates after 90 days, and retrain every quarter.", "risk" if not transfers else "good")
    with t2:
        cc_ = csv("calibration_curve.csv"); cs_ = csv("calibration_summary.csv")
        c1, c2 = st.columns([1.1, 1])
        with c1:
            fig = go.Figure()
            fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#898781", dash="dot"), name="perfect calibration")
            for i, (tgt, g) in enumerate(cc_.groupby("target", sort=False)):
                fig.add_scatter(x=g.mean_predicted, y=g.observed_rate, mode="lines+markers", name=tgt, line=dict(color=SERIES[i], width=2), marker=dict(size=7))
            show(style(fig, "Reliability: predicted vs observed (test sets, decile bins)", 380, xt="Mean predicted probability", yt="Observed rate").update_xaxes(range=[0, 1]).update_yaxes(range=[0, 1]))
        with c2:
            table(cs_[["target", "model", "version", "brier", "brier_skill", "ece", "mean_predicted", "observed_rate"]].round(3))
            st.caption("Brier skill: 1 = perfect, 0 = no better than always predicting the base rate. ECE = average gap between predicted and observed rates. "
                       "Re-calibrated versions are a what-if (fitted on training folds); the deployed models are unchanged.")
        table(csv("risk_band_check.csv").round(3))
    with t3:
        table(csv("churn_operational_ci.csv")[["model", "metric", "point", "ci_low", "ci_high"]].round(3))
        pairs = csv("test_ci_pairs.csv")
        st.markdown("**Paired comparisons on the test set** (same 1,000 resamples for both models)")
        table(pairs[["target", "comparison", "A", "B", "diff", "ci_low", "ci_high", "verdict"]].round(4))
        st.caption("A difference whose 95% interval includes zero is not established by a ~700-member test set — which is why model selection used the one-standard-error rule on cross-validation.")
    with t4:
        wa = js("window_feature_ablation.json"); ge = js("scoring_guard_churn.json")
        cards([("CV PR-AUC with / without", f"{wa['cv_pr_auc_with']:.3f} / {wa['cv_pr_auc_without']:.3f}", f"paired diff {wa['cv_diff']:+.3f} (SE {wa['cv_diff_paired_se']:.3f})"),
               ("Test PR-AUC with / without", f"{wa['test_pr_auc_with']:.3f} / {wa['test_pr_auc_without']:.3f}", f"95% CI of diff {wa['test_diff_ci'][0]:+.3f} to {wa['test_diff_ci'][1]:+.3f}"),
               ("Members beyond training range", f"{wa['share_scored_above_training_range']*100:.0f}%", f"train max {wa['training_max']:.0f} d · scoring max {wa['scoring_max']:.0f} d"),
               ("Scoring guard effect", f"ρ = {ge['spearman_guarded_vs_unguarded']:.3f}", f"{ge['members_changing_band']} members cross the contact threshold")])
        table(csv("window_feature_bands.csv").rename(columns={"Unnamed: 0": "days since first purchase in file"}).round(3))
        st.caption("`days_since_first` counts days since the member's first purchase in the transaction file (which starts on 2024-04-15), so for long-standing members it records where their history "
                   "in the file begins, not their loyalty. The scoring guard clips every feature to its training range before forward scoring.")


def page_churn():
    hero(compact=True)
    F = st.session_state["F"]
    s = MS["churn"]
    section("Churn and retention", "Early-warning design: the model only sees behaviour up to 90 days before the as-of date and predicts whether the member is lapsed 90 days later. "
            "In this period it mostly recognises lapses already under way, and the out-of-time test found no evidence yet that it transfers to a new period — see 'How much can these scores be trusted?'.", "model")
    ci_c = csv("test_ci.csv").query("target == 'churn' and metric == 'pr_auc'").set_index("model")
    sel_ci = ci_c.loc[s["selected_model"]] if s["selected_model"] in ci_c.index else None
    cards([("Selected model", s["selected_model"], "1-SE rule on CV PR-AUC"),
           ("Test PR-AUC", f"{s['selected_test']['pr_auc']:.3f}", (f"95% CI {sel_ci.ci_low:.2f}–{sel_ci.ci_high:.2f} · " if sel_ci is not None else "") + f"no-skill {s['positive_rate']:.2f}"),
           ("Test ROC-AUC", f"{s['selected_test']['roc_auc']:.3f}", "0.5 = coin flip"), ("Operational threshold", f"{THR['operational_threshold']:.2f}", "cost-tuned, 5:1 base case"),
           ("High-value threshold", f"{THR['high_value_threshold']:.2f}", "top-25% value tier")])
    with st.expander("Why not use recency? (leakage audit)"):
        la = csv("leakage_audit_churn.csv")
        fig = px.bar(la, y="design", x="pr_auc", color="model", barmode="group", orientation="h", color_discrete_sequence=[SERIES[0], SERIES[2]], text=la.pr_auc.map(lambda v: f"{v:.2f}"))
        show(style(fig, "PR-AUC by design (prevalence ≈ 0.25)", 330, xt="Cross-validated PR-AUC"))
        st.markdown("The churn flag is almost exactly *no purchase in the last 90 days*. Designs that see that window reconstruct the label (high score, no foresight). "
                    "Design D predicts it 90 days ahead from information available at that date — see the out-of-time test below for how far that carries.")
    st.markdown("#### Choose the operating threshold")
    tc = csv("threshold_curve.csv")
    c0, c1, c2 = st.columns([1.2, 1, 1])
    t = c0.slider("Decision threshold on P(churn)", 0.05, 0.95, float(THR["operational_threshold"]), 0.01, key="thr")
    cfn = c1.number_input("Relative cost of a missed churner", 0.5, 50.0, 5.0, 0.5, key="cfn")
    cfp = c2.number_input("Relative cost of a false alarm", 0.1, 10.0, 1.0, 0.1, key="cfp")
    row = tc.iloc[(tc.threshold - t).abs().argmin()]
    tc["cost_user"] = (cfn * tc.fn + cfp * tc.fp) / (tc.tp + tc.fp + tc.fn + tc.tn)
    best = tc.loc[tc.cost_user.idxmin()]
    cards([("Churners caught (TP)", f"{int(row.tp):,}", f"recall {row.recall:.2f}"), ("Churners missed (FN)", f"{int(row.fn):,}", "costliest error"),
           ("False alarms (FP)", f"{int(row.fp):,}", f"precision {row.precision:.2f}"), ("Flagged per 1,000", f"{row.flagged_share*1000:.0f}", "retention-team workload"),
           ("Cost-optimal threshold", f"{best.threshold:.2f}", f"for {cfn:g}:{cfp:g} costs (scenario)")])
    st.caption("Counts are out-of-fold predictions on the training set (validation data); the test set was used once, at the end, for the matrix below.")
    c3, c4 = st.columns(2)
    with c3:
        fig = go.Figure()
        for col, name, color in [("precision", "Precision", SERIES[0]), ("recall", "Recall", SERIES[1]), ("f1", "F1", SERIES[2]), ("flagged_share", "Share flagged", "#898781")]:
            fig.add_scatter(x=tc.threshold, y=tc[col], name=name, mode="lines", line=dict(color=color, width=2, dash="dot" if col == "flagged_share" else "solid"))
        fig.add_vline(x=t, line_color=NAVY, line_width=2, annotation_text=f"{t:.2f}")
        show(style(fig, "Threshold vs metrics (validation)", 340, xt="Threshold", yt="Value").update_yaxes(range=[0, 1.02]))
    with c4:
        fig = go.Figure(go.Scatter(x=tc.threshold, y=tc.cost_user, mode="lines", line=dict(color=SERIES[3], width=2.4), name="relative cost"))
        fig.add_vline(x=best.threshold, line_color=STATUS["critical"], annotation_text=f"min at {best.threshold:.2f}")
        fig.add_vline(x=t, line_color=NAVY)
        show(style(fig, "Expected relative cost per member (scenario units)", 340, xt="Threshold", yt="Relative cost", legend_bottom=False))
    c5, c6 = st.columns([1, 1.1])
    with c5:
        show(cm_figure("churn", f"Test-set confusion matrix · {MS['churn']['selected_model']} · operational threshold {THR['operational_threshold']:.2f}"))
    with c6:
        st.markdown("**Reading the matrix**")
        st.markdown("- **True positive** — lapsing member flagged in time → retention offer.\n- **False negative** — lapsing member missed → future spend lost (the costliest error).\n"
                    "- **False positive** — active member flagged → an unnecessary offer (cost of the incentive).\n- **True negative** — active member left alone.\n\n"
                    "Lowering the threshold raises recall and the contact list together; the cost curve shows where the trade-off is best for your cost assumptions.")
        table(pd.DataFrame(THR["value_tiers"]).round(3))
    churn_trust()
    st.divider()
    section("At-risk members", "Active members ranked by value at risk (forward 90-day risk × historical spend). Forward scores apply the model to today's behaviour and cannot yet be validated.", "model")
    d = filt_customers(F)
    c7, c8, c9 = st.columns(3)
    tiers = c7.multiselect("Value tier", ["Q4 high", "Q3", "Q2", "Q1 low"], default=["Q4 high", "Q3", "Q2", "Q1 low"], key="ar_tier")
    min_risk = c8.slider("Minimum 90-day risk", 0.0, 1.0, float(t), 0.01, key="ar_min")
    view = c9.radio("List", ["Early warning (active)", "Win-back (already lapsed)"], key="ar_view")
    if view.startswith("Early"):
        lst = d[(d.churned == 0) & (d.churn_risk_next90d >= min_risk) & d.value_tier.isin(tiers)]
    else:
        lst = d[(d.churned == 1) & d.value_tier.isin(tiers)]
    lst = lst.sort_values("value_at_risk_proxy_aed", ascending=False)
    st.caption(f"**{len(lst):,} members** · combined historical spend {aed(lst.monetary.sum())}")
    rb = csv("risk_band_check.csv")
    st.caption("How to read the risk bands (test set): " + " · ".join(f"{r.band}: {r.observed_churn_rate*100:.0f}% actually lapsed (n = {int(r.test_members)})"
                                                                     for r in rb.itertuples() if r.test_members > 0)
               + ". Treat the bands as a ranking (highest risk first), not as exact probabilities.")
    cols = ["segment", "churn_risk_next90d", "risk_band", "value_tier", "monetary", "value_at_risk_proxy_aed", "clv_next180_pred_aed", "recency_days", "discount_dependency", "retention_action", "recommended_bundle"]
    table(lst[cols].reset_index().round(2), height=360, column_config={
        "churn_risk_next90d": st.column_config.ProgressColumn("90-day risk", min_value=0.0, max_value=1.0, format="%.2f"),
        "monetary": st.column_config.NumberColumn("historical spend (AED)", format="localized"),
        "value_at_risk_proxy_aed": st.column_config.NumberColumn("value at risk (AED, proxy)", format="localized"),
        "clv_next180_pred_aed": st.column_config.NumberColumn("pred. next-180-day spend", format="localized")})
    st.download_button("Download list (CSV)", lst[cols].to_csv().encode(), "landmark_at_risk_members.csv", "text/csv", key="dl_risk")
    st.markdown("#### Churn by driver (association, not cause)")
    dr = csv("churn_drivers.csv")
    dim = st.selectbox("Dimension", dr.dimension.unique().tolist(), key="drv_dim")
    g = dr[dr.dimension == dim]
    fig = go.Figure(go.Bar(y=g.level.astype(str), x=g.churn_rate_pct, orientation="h", marker_color=SERIES[0], text=[f"{v:.0f}% (n={n:,})" for v, n in zip(g.churn_rate_pct, g.members)], textposition="outside"))
    fig.add_vline(x=CM.churned.mean() * 100, line_color=STATUS["critical"], annotation_text="overall")
    show(style(fig, f"Churn rate by {dim.replace('_', ' ')}", 340, xt="% churned", legend_bottom=False).update_xaxes(range=[0, max(g.churn_rate_pct.max() * 1.35, 10)]))
    if (L.METRICS / "churn_association_tests.csv").exists():
        ca = csv("churn_association_tests.csv")
        st.markdown("**Are these differences real?** Chi-square test of churn vs each grouping (groups ≥ 50 members); Bonferroni-corrected for testing them all at once.")
        table(ca.round(4))


# ================================================================================================================
# I · Customer value & regression
# ================================================================================================================
def page_value():
    hero(compact=True)
    F = st.session_state["F"]
    section("Customer value (CLV)", "Historical spend is measured; next-180-day value is a model forecast validated on a past cutoff.", "model")
    filter_note(F)
    d = filt_customers(F)
    if empty_guard(d):
        return
    rh, rn = csv("regression_historical.csv"), csv("regression_next_period.csv")
    bestN = rn[~rn.model.str.startswith("Baseline")].sort_values("cv_rmse").iloc[0]
    persist = rn[rn.model.str.startswith("Baseline · persistence (")].iloc[0]
    iv = js("clv_intervals.json"); bench = csv("clv_benchmark.csv")
    dep = bench[bench.deployed].iloc[0]
    cards([("Members in view", f"{len(d):,}", "after filters"), ("Historical spend (sum)", aed(d.monetary.sum()), "2024-04-15 → 2026-10-01"),
           ("Predicted next 180 days (sum)", aed(d.clv_next180_pred_aed.sum()), f"{bestN.model}"),
           ("Forward model vs persistence", f"RMSE {bestN.test_rmse:,.0f} vs {persist.test_rmse:,.0f}", "test set, AED (lower is better)"),
           ("80% range — test coverage", f"{iv['test_coverage_conformal']*100:.0f}%", f"target 80% · median width {aed(iv['median_width_aed'])}"),
           ("Ranking power", f"top 10% → {dep.top_decile_capture*100:.0f}%", f"of actual next-180-day spend · ρ {dep.spearman:.2f}")])
    h1, h2 = st.columns(2)
    with h1:
        show(style(go.Figure(go.Histogram(x=d.monetary, nbinsx=40, marker_color=SERIES[0])), "Historical spend per member (Apr 2024 → Oct 2026, measured)", 300,
                   xt="AED per member", yt="Members", legend_bottom=False))
    with h2:
        show(style(go.Figure(go.Histogram(x=d.clv_next180_pred_aed, nbinsx=40, marker_color=SERIES[2])), "Predicted spend in the next 180 days (model forecast)", 300,
                   xt="AED per member", yt="Members", legend_bottom=False))
    st.caption("Different horizons (≈29 months measured vs 180 days forecast) — compare rankings, not amounts.")
    c1, c2 = st.columns(2)
    with c2:
        st.markdown("**Top members**")
        by = st.radio("Rank by", ["Historical spend", "Predicted next 180 days"], horizontal=True, key="top_by")
        col = "monetary" if by.startswith("Hist") else "clv_next180_pred_aed"
        top = d.sort_values(col, ascending=False).head(25).copy()
        top["range_80"] = [f"{a:,.0f}–{b:,.0f}" for a, b in zip(top.clv_next180_low_aed, top.clv_next180_high_aed)]
        table(top[["segment", "monetary", "clv_next180_pred_aed", "range_80", "churn_risk_next90d", "risk_band", "value_tier"]].reset_index(), height=380,
              column_config={"customer_id": st.column_config.TextColumn("Member"), "segment": st.column_config.TextColumn("Segment"),
                             "monetary": st.column_config.NumberColumn("Historical (AED)", format="localized"),
                             "clv_next180_pred_aed": st.column_config.NumberColumn("Next 180 d (AED)", format="%.0f"),
                             "range_80": st.column_config.TextColumn("80% range (AED)"),
                             "churn_risk_next90d": st.column_config.ProgressColumn("90-day risk", min_value=0.0, max_value=1.0, format="%.2f"),
                             "risk_band": st.column_config.TextColumn("Risk band"), "value_tier": st.column_config.TextColumn("Value tier")})
    with c1:
        tp = csv("clv_test_predictions.csv")
        mdl = st.radio("Model", tp.model.unique().tolist(), horizontal=True, key="clv_model")
        g = tp[tp.model == mdl]
        fig = go.Figure(go.Scattergl(x=g.actual_aed, y=g.predicted_aed, mode="markers", marker=dict(size=5, color=SERIES[0], opacity=0.55), name="members"))
        lim = float(max(g.actual_aed.max(), g.predicted_aed.max()))
        fig.add_scatter(x=[0, lim], y=[0, lim], mode="lines", line=dict(color=STATUS["critical"], width=1.4), name="perfect prediction")
        show(style(fig, "Actual vs predicted (test set)", 340, xt="Actual (AED)", yt="Predicted (AED)"))
    st.markdown("**Historical CLV — circularity audit across feature sets** (set C is the trap: monetary = frequency × ticket)")
    rp = rh.pivot_table(index="model", columns="feature_set", values="cv_r2").round(3)
    table(rp.reset_index())
    st.markdown("**Next-180-day spend — models vs naive baselines** (including the log-target, Poisson-loss and two-part variants)")
    table(rn[["model", "best_params", "cv_rmse", "cv_r2", "train_r2", "test_mae", "test_rmse", "test_r2"]].round(3))
    st.markdown("#### Is the forward model worth using? Benchmarks, ranking and ranges")
    b1, b2 = st.columns(2)
    with b1:
        cb = bench.sort_values("test_rmse", ascending=False)
        colr = [NAVY if r.deployed else ("#898781" if r.kind == "baseline" else (SERIES[1] if r.kind == "probabilistic" else SERIES[2])) for r in cb.itertuples()]
        fig = go.Figure(go.Bar(y=cb.model, x=cb.test_rmse, orientation="h", marker_color=colr,
                               error_x=dict(type="data", symmetric=False, array=(cb.rmse_ci_high - cb.test_rmse).fillna(0), arrayminus=(cb.test_rmse - cb.rmse_ci_low).fillna(0), color=SLATE),
                               hovertemplate="%{y}: RMSE %{x:,.0f}<extra></extra>"))
        show(style(fig, "Test RMSE with 95% CI (navy = deployed; lower is better)", 420, xt="RMSE (AED)", legend_bottom=False))
    with b2:
        cr = bench.dropna(subset=["spearman"]).sort_values("top_decile_capture")
        fig = go.Figure(go.Bar(y=cr.model, x=cr.top_decile_capture * 100, orientation="h", marker_color=[NAVY if d_ else SERIES[0] for d_ in cr.deployed],
                               text=[f"{v*100:.0f}% · ρ {r:.2f}" for v, r in zip(cr.top_decile_capture, cr.spearman)], textposition="outside"))
        fig.add_vline(x=10, line_dash="dot", line_color="#898781")
        show(style(fig, "Share of actual future spend in the top 10% of predictions", 420, xt="% captured (dotted line = random ranking, 10%)", legend_bottom=False).update_xaxes(range=[0, cr.top_decile_capture.max() * 140]))
    pb = js("clv_probabilistic.json")
    st.caption(f"BG/NBD × Gamma-Gamma (probabilistic benchmark): mean drop-out probability {L.fmt_small_p(pb['bgnbd']['mean_dropout_p'])}, "
               f"predicted {pb['test_mean_purchase_days_predicted']:.2f} vs actual {pb['test_mean_purchase_days_actual']:.2f} purchase days per test member in 180 days — "
               "it assumes a stable purchase and drop-out process, which this period does not show (churn arrived as a wave after the cutoff).")
    ti = csv("clv_test_intervals.csv").sort_values("predicted_aed").reset_index(drop=True)
    inside = (ti.actual_aed >= ti.low_aed) & (ti.actual_aed <= ti.high_aed)
    fig = go.Figure()
    fig.add_scatter(x=ti.index, y=ti.high_aed, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=ti.index, y=ti.low_aed, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(42,157,143,0.22)", name="80% conformal range")
    fig.add_scatter(x=ti.index, y=ti.predicted_aed, mode="lines", line=dict(color=NAVY, width=1.6), name="point forecast")
    fig.add_scatter(x=ti.index[inside], y=ti.actual_aed[inside], mode="markers", marker=dict(size=4, color=SERIES[0]), name="actual — inside range")
    fig.add_scatter(x=ti.index[~inside], y=ti.actual_aed[~inside], mode="markers", marker=dict(size=5, color=SERIES[1], symbol="x"), name="actual — outside range")
    show(style(fig, f"Test members sorted by forecast — {inside.mean()*100:.0f}% inside their 80% range", 380, xt="Test members (sorted by forecast)", yt="AED, next 180 days"))
    lc = csv("learning_curves.csv"); lc = lc[lc.target == "clv"]
    lc_cols = st.columns(2)
    for col_, (prefix, ttl) in zip(lc_cols, [("Set B", "historical CLV"), ("Next-180", "next-180-day CLV")]):
        fig = go.Figure()
        for i, (m, g) in enumerate(lc[lc.model.str.startswith(prefix)].groupby("model")):
            nm = m.split(" · ", 1)[1] if " · " in m else m
            fig.add_scatter(x=g.n, y=g.train, name=f"{nm} · train", line=dict(color=SERIES[i % 8], dash="dash"))
            fig.add_scatter(x=g.n, y=g.cv, name=f"{nm} · CV", line=dict(color=SERIES[i % 8]))
        with col_:
            show(style(fig, f"Learning curves (R²) — {ttl}", 400, xt="Training members", yt="R²", legend_below=True))
    st.caption("A persistent train–CV gap = variance (the model memorises); both curves low and flat = bias (the features cannot explain more).")
    callout("Historical spend answers <i>who has been valuable</i>; the next-180-day model answers <i>who is likely to be valuable next</i> and must beat the 'same as last 180 days' baseline to be worth using. "
            "Neither includes Viva grocery spend (blind spot).")


# ================================================================================================================
# J · Shukran loyalty intelligence
# ================================================================================================================
def page_shukran():
    hero(compact=True)
    F = st.session_state["F"]
    sm = js("shukran_summary.json")
    section("Shukran loyalty intelligence", "Coverage, cohorts, retention and discount behaviour of the 3,000 member records supplied — not the programme's full membership.", "measured")
    filter_note(F)
    d = filt_customers(F)
    if empty_guard(d):
        return
    cards([("Member records (dataset)", f"{sm['member_records_in_dataset']:,}", "not total programme users"), ("Active / churned", f"{sm['active_members']:,} / {sm['churned_members']:,}", f"churn {sm['churn_rate_pct']}%"),
           ("Visit 3+ banners", f"{sm['members_visiting_3plus_banners_pct']}%", f"mean {sm['mean_banners_per_member']} banners"),
           ("Discounted transactions", f"{sm['share_transactions_discounted_pct']}%", f"{sm['share_of_spend_discounted_pct']}% of spend"),
           ("Measured member savings", "Unavailable", "flag only — no amounts")])
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        sq = csv("shukran_signups_by_quarter.csv")
        qx = sq.iloc[:, 0].astype(str)
        fig = go.Figure(go.Bar(x=qx, y=sq.new_members, marker_color=SERIES[0], hovertemplate="%{x}: %{y} new members<extra></extra>"))
        q1 = [q for q in qx if q.endswith("Q1")]
        show(style(fig, "New members by signup quarter (signups, not departures)", 320, xt="Signup quarter (tick = Q1 of each year)", yt="Members", legend_bottom=False)
             .update_xaxes(tickangle=0, tickmode="array", tickvals=q1, ticktext=[q[:4] for q in q1]))
    with c2:
        part = d.n_banners.value_counts().sort_index()
        bycs = d.groupby("n_banners").churned.mean() * 100
        fig = go.Figure(go.Bar(x=part.index.astype(str), y=part.values, marker_color=SERIES[2], text=[f"churn {bycs[i]:.0f}%" for i in part.index], textposition="outside"))
        show(style(fig, "Cross-banner participation (filtered) and churn rate", 320, xt="Shukran banners visited", yt="Members", legend_bottom=False).update_yaxes(range=[0, part.max() * 1.2]))
    st.markdown("#### Discount dependency")
    dd = csv("discount_dependency.csv", index_col=0)
    c3, c4 = st.columns([1.2, 1])
    with c3:
        table(dd.reset_index().round(1), height=260)
    with c4:
        sd = csv("discount_by_segment.csv", index_col=0)
        fig = go.Figure(go.Bar(y=sd.index, x=sd.discounted_spend_share_pct, orientation="h", marker_color=[SEG_COLORS.get(s, SLATE) for s in sd.index],
                               text=[f"{v:.1f}%" for v in sd.discounted_spend_share_pct], textposition="outside"))
        show(style(fig, "Share of spend made on discount, by segment", 260, xt="%", legend_bottom=False).update_xaxes(range=[0, sd.discounted_spend_share_pct.max() * 1.3]))
    section("Hypothetical saving proxy", "Discounted spend is what members paid, not what they saved. Enter an assumed average discount depth to see an illustrative saving.", "scenario")
    depth = st.slider("Assumed average discount depth (%)", 1, 60, 20, key="depth")
    sp = sm["spend_on_discounted_transactions_aed"]
    cards([("Spend on discounted transactions (measured)", aed(sp), "Shukran banners, full period"), (f"Hypothetical saving at {depth}% depth", aed(sp * depth / (100 - depth)), "= spend × d / (1 − d)"),
           ("Funding source", "Unknown", "not every discount is Shukran-funded")])
    st.markdown("#### Has Shukran helped or harmed Landmark?")
    st.table(pd.DataFrame([
        ["Customer", "Members use discounts on a large minority of transactions and spread spend across banners", "How much they saved or valued the programme"],
        ["Group · retention", "Churn rates by segment, breadth, discount use and value are tested on the Churn page — differences are small and mostly within noise", "That Shukran caused lower churn — no non-member comparison group"],
        ["Group · revenue", "All seven banners' spend captured under one ID → cross-banner targeting is possible", "How much revenue is incremental"],
        ["Group · margin", "An identifiable discount-dependent group exists", "Actual margin erosion — no discount amounts or costs"]],
        columns=["perspective", "evidence (measured)", "cannot be concluded"]).set_index("perspective"))
    callout(f"<b>Verdict:</b> {sm['verdict_helped_or_harmed']} <b>Recommended evaluation:</b> " + "; ".join(sm["recommended_evaluation"]) + ".")


# ================================================================================================================
# K · Acquisition & migration
# ================================================================================================================
def page_acquisition():
    hero(compact=True)
    F = st.session_state["F"]
    section("Customer acquisition and migration", "Signup cohorts, who the members are, and which value-tier shoppers look premium elsewhere.", "measured")
    filter_note(F)
    d = filt_customers(F)
    if empty_guard(d):
        return
    sy = d.groupby("signup_year").agg(members=("churned", "size"), churn=("churned", "mean"), spend=("monetary", "median")).reset_index()
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(x=sy.signup_year.astype(str), y=sy.members, marker_color=SERIES[0], text=[f"churn {c*100:.0f}%" for c in sy.churn], textposition="outside"))
        s0, s1 = pd.to_datetime(CM.signup_date).min(), pd.to_datetime(CM.signup_date).max()
        show(style(fig, f"Members by signup year ({s0.year} = from {s0:%d %b}; {s1.year} = to {s1:%d %b} — partial years)", 320, xt="Signup year", yt="Members", legend_bottom=False)
             .update_yaxes(range=[0, sy.members.max() * 1.2]))
    with c2:
        dim = st.selectbox("Profile by", ["age_band", "household", "emirate", "tenure_band", "gender"], key="acq_dim")
        ct = (pd.crosstab(d[dim], d.segment, normalize="columns") * 100).round(1)
        fig = go.Figure([go.Bar(x=ct.index.astype(str), y=ct[s], name=s, marker_color=SEG_COLORS[s]) for s in SEG_ORDER if s in ct])
        show(style(fig.update_layout(barmode="group"), f"Segment composition by {dim.replace('_', ' ')} (% of segment)", 380, xt=dim.replace("_", " "), yt="% of segment", legend_below=True))
    st.caption("Profiles describe observed groups; they are not used to target individuals by demographics.")
    st.markdown("#### Value-to-premium migration candidates")
    mig = csv("migration_candidates.csv"); mig = mig[mig.customer_id.isin(d.index)]
    pair = st.radio("Pair", ["Homebox → Home Centre", "Max → Splash"], horizontal=True, key="mig_pair")
    m = mig[mig.pair == pair]
    c3, c4 = st.columns([1, 1.25])        # wider table side so the segment column is not cut off
    with c3:
        fig = go.Figure()
        for lab, g, sym in [("Active", m[m.churned == 0], "circle"), ("Lapsed", m[m.churned == 1], "x")]:
            fig.add_scatter(x=g.migration_score, y=g.monetary, mode="markers", name=lab, marker=dict(size=8, symbol=sym, color=SERIES[0] if lab == "Active" else STATUS["critical"], opacity=0.7),
                            customdata=g.customer_id, hovertemplate="%{customdata}<br>score %{x:.2f}<br>AED %{y:,.0f}<extra></extra>")
        show(style(fig, "Candidates: premium-likeness score vs historical spend", 360, xt="Migration score (P premium, out-of-fold)", yt="Historical spend (AED)"))
    with c4:
        pr = m[m.churned == 0].assign(priority=lambda x: x.migration_score * (1 - x.churn_risk_next90d))
        st.markdown("**Prioritised actions** — active members ranked by priority = migration score × (1 − 90-day churn risk)")
        table(pr.sort_values("priority", ascending=False).head(15)[["customer_id", "priority", "migration_score", "churn_risk_next90d", "monetary", "segment"]]
              .rename(columns={"migration_score": "score", "churn_risk_next90d": "churn risk", "monetary": "spend (AED)"}).round(2), height=360,
              column_config={c: st.column_config.NumberColumn(c, format="%.2f", width="small") for c in ["priority", "score", "churn risk"]}
              | {"spend (AED)": st.column_config.NumberColumn("spend (AED)", format="%.0f", width="small"), "customer_id": st.column_config.TextColumn("member", width="small")})
    xc = csv("migration_style_crosscheck.csv"); xr = xc[xc.pair == pair]
    if len(xr):
        xr = xr.iloc[0]
        st.caption(f"Independent cross-check with the shopping-style view: {xr.share_in_style_candidates*100:.0f}% of these candidates have the *{xr.style}* style, "
                   f"against {xr.share_in_style_all_value_shoppers*100:.0f}% of all {pair.split(' → ')[0]} shoppers ({xr.concentration_ratio:.1f}×) — two unrelated methods point to the same members.")
    callout("Test before scaling: randomise active candidates into an offer group and a control group; measure first premium-banner purchase within 90 days and net margin after the offer cost.", "")
    st.markdown("#### Cross-banner engagement by segment")
    eng = pd.crosstab(d.segment, d.n_banners, normalize="index") * 100
    fig = go.Figure([go.Bar(y=eng.index, x=eng[c], name=f"{c} banner{'' if int(c) == 1 else 's'}", orientation="h", marker_color=px.colors.sequential.Blues[min(2 + int(c), 8)]) for c in eng.columns])
    show(style(fig.update_layout(barmode="stack"), "Distribution of banners visited within each segment", 300, xt="% of segment"))


# ================================================================================================================
# L · Cross-banner bundles & epsilon sandbox
# ================================================================================================================
def page_bundles():
    hero(compact=True)
    F = st.session_state["F"]
    section("Cross-banner bundles", "Apriori on member-level banner baskets: which banners the same member shops. Affinity, not same-checkout purchase, and not causation.", "model")
    rules, screened, lift = csv("rules_all.csv"), csv("rules_screened.csv"), csv("pair_lift.csv")
    tl = js("tuning_log.json").get("apriori", {})
    st.caption(f"Thresholds: support ≥ {tl.get('min_support')}, confidence ≥ {tl.get('min_confidence')}, lift ≥ {tl.get('min_lift')}, {tl.get('significance')}, {tl.get('actionable')}.")
    c1, c2 = st.columns([1.3, 1])
    with c1:
        ml = st.slider("Minimum lift to display", 1.0, float(max(rules.lift.max(), 1.1)), 1.05, 0.01, key="minlift")
        r = rules[rules.lift >= ml]
        rc_ = [c for c in ["antecedent", "consequent", "support", "confidence", "lift", "lift_adj_activity", "lift_adj_activity_household", "affinity_type", "members_with_rule", "p_value"] if c in r.columns]
        table(short_affinity(r[rc_]).round(4), height=330, column_config={"lift": st.column_config.NumberColumn("lift", format="%.2f"),
                                                            "lift_adj_activity": st.column_config.NumberColumn("lift | activity", format="%.2f"),
                                                            "lift_adj_activity_household": st.column_config.NumberColumn("lift | activity + kids", format="%.2f"),
                                                            "affinity_type": st.column_config.TextColumn("what drives it"),
                                                            "support": st.column_config.NumberColumn("support", format="%.3f")})
        if "affinity_type" in rules.columns:
            n_hh = int(rules.affinity_type.str.startswith("household").sum()); n_g = int(rules.affinity_type.str.startswith("banner").sum())
            st.caption(f"Adjusted lifts hold activity (transaction tercile) and then activity + kids fixed. {n_hh} of {len(rules)} rules are household-driven — "
                       f"their banners co-occur because families shop both, not because one banner leads to the other; {n_g} survive as genuine banner affinities.")
        with st.expander(f"Rejected rules ({(~screened.kept).sum()}) and why"):
            table(screened[~screened.kept][["antecedent", "consequent", "support", "confidence", "lift", "rejection_reason"]].round(3), height=280)
    with c2:
        m = pd.DataFrame(index=L.CORE_BANNERS, columns=L.CORE_BANNERS, dtype=float)
        for _, x in lift.iterrows():
            m.loc[x.banner_a, x.banner_b] = m.loc[x.banner_b, x.banner_a] = x.lift
        fig = px.imshow(m.round(2), text_auto=True, color_continuous_scale="RdBu", zmin=0, zmax=2, aspect="auto", labels=dict(color="lift"))
        show(style(fig, "Pairwise co-visit lift (1 = independent; 0 = never together)", 380, legend_bottom=False))
    st.markdown("#### Bundles by segment")
    bo = csv("bundle_offers.csv"); bo = bo[bo.segment.isin(F["segs"])]
    seg = st.selectbox("Segment", [s for s in SEG_ORDER if s in F["segs"]], key="bund_seg")
    b = bo[bo.segment == seg].sort_values("opportunity_score", ascending=False)
    st.caption("Opportunity score = base rate of the destination banner × (adjusted lift − 1) × eligible members: the extra co-shoppers the affinity implies after allowing for activity and "
               "household (the unadjusted score is shown alongside). A sizing heuristic, not forecast uplift.")
    bcols = [c for c in ["rule", "offer_idea", "affinity_type", "eligible_members", "opportunity_score", "opportunity_score_raw", "lift", "lift_adj_activity_household",
                         "avg_churn_risk_next90d", "avg_historical_value_aed", "avg_pred_next180_aed", "audience_note"] if c in b.columns]
    table(short_affinity(b[bcols]).round(3), height=280, column_config={"affinity_type": st.column_config.TextColumn("what drives it"), "opportunity_score": st.column_config.NumberColumn("opportunity (adjusted)", format="%.1f"),
                                                        "opportunity_score_raw": st.column_config.NumberColumn("opportunity (unadjusted)", format="%.1f"),
                                                        "lift_adj_activity_household": st.column_config.NumberColumn("lift | activity + kids", format="%.2f")})
    rs = csv("rules_by_segment.csv")
    with st.expander("Segment-specific rules (mined within the segment)"):
        table(rs[rs.segment == seg].round(3))
    with st.expander("Robustness: do the rules hold for same-month baskets?"):
        rr = csv("rules_month_robustness.csv")
        if "lift_above_1_within_month" in rr.columns:
            st.caption(f"Each kept rule re-scored on customer-month baskets: {int(rr.lift_above_1_within_month.sum())} of {len(rr)} keep lift > 1 within the same month "
                       f"(co-occurring in {int(rr.member_months_with_all.min()):,}–{int(rr.member_months_with_all.max()):,} member-months). Rules that drop below 1 are "
                       "portfolio habits — the same member uses both banners at different times — so they suit a next-visit introduction, not a same-day bundle.")
        table(rr.round(3))
    callout("Rules show where co-shopping already happens across the year — mostly in different months (see the robustness check), so each offer is a "
            "<b>next-visit introduction</b>, not a same-day bundle. Any offer must be tested against a randomised hold-out before rollout.", "")
    st.divider()
    section("ε-greedy offer sandbox", "How a live test would balance exploring offers vs exploiting the best one. Redemption rates are YOUR hypothetical inputs — the data contain no offer exposure or redemption.", "sim")
    arms = csv("epsilon_arms.csv")
    ed = st.data_editor(arms[["arm", "hypothetical_redemption_p"]], key="arms_ed", hide_index=True, **W_EDITOR, disabled=["arm"],
                        column_config={"hypothetical_redemption_p": st.column_config.NumberColumn("hypothetical redemption probability", min_value=0.0, max_value=1.0, step=0.01)})
    c3, c4, c5 = st.columns(3)
    eps = c3.slider("ε (exploration rate)", 0.0, 1.0, 0.10, 0.01, key="eps")
    decay = c4.toggle("Decay ε over time", value=False, key="decay")
    steps = c5.select_slider("Contacts simulated", [500, 1000, 2000, 4000], value=2000, key="steps")
    res = sim(tuple(ed.hypothetical_redemption_p.astype(float)), eps, steps, decay)
    base = sim(tuple(ed.hypothetical_redemption_p.astype(float)), 0.0, steps, False)
    cards([("Share of contacts on the best offer (last 20%)", f"{res['pct_optimal'][-steps // 5:].mean():.0%}", "SIMULATION"),
           ("Average redemption per contact", f"{res['avg_reward'].mean():.3f}", f"pure exploitation: {base['avg_reward'].mean():.3f}"),
           ("Cumulative regret", f"{res['cum_regret'][-1]:.1f}", "redemptions lost vs always-best"), ("Status", "Simulation", "no real uplift claimed")])
    fig = go.Figure([go.Scatter(y=pd.Series(res["pct_optimal"]).rolling(50, min_periods=1).mean(), name=f"ε = {eps:.2f}{' (decaying)' if decay else ''}", line=dict(color=SERIES[0], width=2.2)),
                     go.Scatter(y=pd.Series(base["pct_optimal"]).rolling(50, min_periods=1).mean(), name="ε = 0 (pure exploitation)", line=dict(color="#898781", width=2, dash="dot"))])
    show(style(fig, "SIMULATION — share of contacts receiving the best offer (rolling 50)", 330, xt="Contact number", yt="Share").update_yaxes(range=[0, 1.02]))
    callout("Next step for real evidence: run a live A/B or bandit pilot that records which offer each member saw and whether they redeemed it.", "sim")


@st.cache_data(show_spinner=False)
def sim(p, eps, steps, decay):
    return L.epsilon_greedy_sim(list(p), eps, steps=steps, runs=200, decay=decay)


# ================================================================================================================
# M · Model laboratory
# ================================================================================================================
def page_lab():
    hero(compact=True)
    section("Model laboratory", "Seventeen models per target — LR, decision tree, KNN, linear & RBF SVM, Gaussian NB, MLP, plus bagging (RF, Extra Trees, bagged trees), boosting (AdaBoost, gradient boosting, XGBoost), soft/hard voting and stacking — under one protocol: stratified split, 5-fold CV tuning on train, one final test evaluation.", "model")
    tgt = st.radio("Target", ["churn", "home", "fashion"], horizontal=True, format_func={"churn": "Churn (early-warning)", "home": "Home Centre vs Homebox", "fashion": "Splash vs Max"}.get, key="lab_tgt")
    s = MS[tgt]; lb = csv(f"leaderboard_{tgt}.csv"); pm = s["primary_metric"]
    callout(re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s["summary"]))
    st.caption(f"Selection rule: the least complex model within one standard error ({s['one_se']:.4f}) of the best CV {pm}. Rank 1 = best; ties share the better rank.")
    metric = st.selectbox("Compare models on", list(dict.fromkeys(["cv_" + pm, "cv_accuracy", "cv_precision", "cv_recall", "cv_f1", "cv_balanced_accuracy", "cv_roc_auc", "cv_pr_auc", "test_" + pm])), key="lab_metric")
    g = lb.sort_values(metric, ascending=True, na_position="first")
    err = g[metric + "_std"] if metric + "_std" in g else None
    fig = go.Figure()
    for fam, gg in g.groupby("family", sort=False):        # dot plot: the axis is zoomed, so bars would exaggerate differences
        fig.add_scatter(y=gg.model, x=gg[metric], mode="markers", name=fam, marker=dict(size=11, color=FAMILY_COLORS[fam], line=dict(color="white", width=1.5)),
                        error_x=dict(type="data", array=gg[metric + "_std"], color=SLATE, thickness=1.2) if err is not None else None,
                        hovertemplate="%{y}: %{x:.3f}<extra>" + fam + "</extra>")
    sel_row = g[g.model == s["selected_model"]]
    if len(sel_row) and sel_row[metric].notna().all():
        fig.add_annotation(y=s["selected_model"], x=float(sel_row[metric].iloc[0]), text="◀ selected", showarrow=False, xanchor="left", xshift=14, font=dict(size=11, color=NAVY))
    fig.update_layout(yaxis=dict(categoryorder="array", categoryarray=g.model.tolist()))
    lo = float(np.nanmin(g[metric])) if g[metric].notna().any() else 0
    show(style(fig, f"{metric} by model (colour = family; whiskers = ±1 SD across folds)", 520, xt=metric)
         .update_xaxes(range=[max(0, lo - 0.06), min(1.02, float(np.nanmax(g[metric] + (g[metric + '_std'] if err is not None else 0))) + 0.05)], showgrid=True, gridcolor="#ECE8DF"))
    if g[metric].isna().any():
        st.caption(f"No value for {', '.join(g[g[metric].isna()].model)}: hard voting outputs class labels only, so probability-based metrics (ROC-AUC, PR-AUC) do not exist for it.")
    t1, t2, t3, t4 = st.tabs(["Leaderboard", "Rankings", "Family winners", "Test results with 95% CI"])
    with t4:
        tci = csv("test_ci.csv"); tci = tci[(tci.target == tgt) & (tci.metric == pm)].sort_values("point")
        fig = go.Figure(go.Scatter(x=tci.point, y=tci.model, mode="markers", marker=dict(size=10, color=[NAVY if m == s["selected_model"] else SERIES[0] for m in tci.model]),
                                   error_x=dict(type="data", symmetric=False, array=tci.ci_high - tci.point, arrayminus=tci.point - tci.ci_low, color=SLATE, thickness=1.2),
                                   hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        show(style(fig, f"Test {pm} with 95% bootstrap interval (navy = deployed)", 520, xt=f"test {pm}", legend_bottom=False).update_xaxes(showgrid=True, gridcolor="#ECE8DF"))
        pr_ = csv("test_ci_pairs.csv"); pr_ = pr_[pr_.target == tgt]
        if len(pr_):
            table(pr_[["comparison", "A", "B", "diff", "ci_low", "ci_high", "share_A_better", "verdict"]].round(4))
        st.caption("1,000 stratified resamples of the test set, identical for every model, so the comparisons are paired. Overlapping intervals mean the test set cannot separate the models.")
    with t1:
        cols = ["model", "family", "best_params", f"cv_{pm}", f"cv_{pm}_std", "cv_accuracy", "cv_precision", "cv_recall", "cv_f1", "cv_balanced_accuracy", "cv_roc_auc", "cv_pr_auc",
                f"train_{pm}", "gap_train_minus_cv", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc", "test_pr_auc", "tune_and_cv_seconds", "status"]
        cols = list(dict.fromkeys(cols))                      # primary metric may already be in the list
        table(lb.sort_values(f"cv_{pm}", ascending=False)[cols].round(3), height=420)
    with t2:
        rk = lb.set_index("model")[[f"rank_{m}" for m in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]]].sort_values(f"rank_{pm}")
        table(rk.reset_index(), height=420)
    with t3:
        fw = csv(f"family_winners_{tgt}.csv")
        table(fw[["family", "model", f"cv_{pm}", "cv_f1", "cv_recall"]].round(4))
    c1, c2 = st.columns(2)
    with c1:
        show(cm_figure(tgt, f"Confusion matrix — selected model, test set" + (" (operational threshold)" if tgt == "churn" else " (threshold 0.5)")))
    with c2:
        imp = csv(f"perm_importance_{tgt}.csv").head(12)[::-1]
        show(style(go.Figure(go.Bar(y=imp.feature.map(feat_label), x=imp.importance_mean, orientation="h", marker_color=SERIES[0], error_x=dict(type="data", array=imp.importance_std, color=SLATE))),
                   "Permutation importance (test set)", 420, xt=f"Drop in {pm}", legend_bottom=False))
    st.markdown("#### Neural network — activation functions")
    ma = csv("mlp_activations.csv"); ma = ma[ma.target == tgt]
    c3, c4 = st.columns([1, 1.2])
    with c3:
        fig = go.Figure(go.Bar(x=ma.activation, y=ma.cv_score, error_y=dict(type="data", array=ma.cv_std, color=SLATE), marker_color=SERIES[0], name="MLP"))
        fig.add_hline(y=ma.best_non_MLP_cv.iloc[0], line_color=STATUS["critical"], annotation_text="best non-MLP model")
        show(style(fig, f"MLP CV {pm} by activation", 320, xt="activation", yt=f"CV {pm}", legend_bottom=False).update_yaxes(range=[max(0, ma.cv_score.min() - 0.08), None]))
    with c4:
        table(ma[["activation", "best_hidden_layers", "best_alpha", "cv_score", "train_score", "gap_train_minus_cv", "mean_fit_seconds", "mean_epochs", "folds_hitting_iteration_cap"]].round(4))
        st.caption("identity = linear (no activation); logistic = sigmoid; activations exist only inside neural networks — trees, SVMs, KNN and Naive Bayes get their non-linearity elsewhere.")
    st.markdown("#### Bias–variance")
    dg = csv("diagnostics.csv"); dg = dg[dg.target == tgt]
    table(dg[["model", "family", "train", "cv", "test", "gap_train_cv", "diagnosis", "mitigation_applied", "default_cv", "mitigation_worked"]].round(3), height=330)
    bvd = csv("bias_variance_decomposition.csv")
    c5, c6 = st.columns(2)
    for col, (task, g) in zip((c5, c6), bvd.groupby("task", sort=False)):
        fig = go.Figure([go.Bar(y=g.model, x=g.bias, orientation="h", name="bias", marker_color=SERIES[0]),
                         go.Bar(y=g.model, x=g.variance, orientation="h", name="variance", marker_color=SERIES[1], marker_pattern_shape="/")])
        with col:
            show(style(fig.update_layout(barmode="stack"), f"Decomposition · {task}", 320, xt="expected loss"))
    with st.expander("PCA inside the pipeline — with vs without"):
        table(csv("pca_vs_raw_classification.csv").round(4))


# ================================================================================================================
# N · Data quality & methodology
# ================================================================================================================
def validation_summary():
    """One table of every robustness check added in the review, with its result, read from the artifacts."""
    oot = js("oot_summary.json"); oci = {r["model"]: r for r in oot["pr_auc_ci_test_members"]}
    wa = js("window_feature_ablation.json"); cal = csv("calibration_summary.csv"); cal = cal[cal.version == "as deployed"].set_index("target")
    tci = csv("test_ci.csv"); pairs = csv("test_ci_pairs.csv"); cb = csv("clv_benchmark.csv"); iv = js("clv_intervals.json"); pb = js("clv_probabilistic.json")
    rules = csv("rules_all.csv"); sc = js("style_choice.json"); ge = js("scoring_guard_churn.json")
    dep_ci = {t: tci[(tci.target == t) & (tci.model == MS[t]["selected_model"]) & (tci.metric == MS[t]["primary_metric"])] for t in ("churn", "home", "fashion")}
    cbi = cb.set_index("model"); dep = cb[cb.deployed].iloc[0]
    n_tests = len(re.findall(r"^def test_", (ROOT_DIR / "tests" / "test_core.py").read_text(), flags=re.M)) if (ROOT_DIR / "tests" / "test_core.py").exists() else 0
    rows = [
        ("Out-of-time churn test", "Does the churn model work in a period it was not trained on?",
         f"PR-AUC {oci['out-of-time']['point']:.2f} out of time vs {oci['in-time']['point']:.2f} in time (no skill {MS['churn']['positive_rate']:.2f}) — "
         + ("transfers" if oot.get("transfers", oci["out-of-time"]["ci_low"] > MS["churn"]["positive_rate"] + 0.05) else "does not transfer; pilot before relying on forward scores"), "§10.6"),
        ("Data-window feature", "Is a top churn signal an artefact of when the file starts?",
         f"removing it {'lowers' if wa['cv_diff'] > 0 else 'raises'} CV PR-AUC by {abs(wa['cv_diff']):.3f} ({wa['cv_pr_auc_with']:.3f} → {wa['cv_pr_auc_without']:.3f}); "
         f"{wa['share_scored_above_training_range']*100:.0f}% of members beyond the training range at scoring → clipped by the scoring guard "
         f"(rank correlation {ge['spearman_guarded_vs_unguarded']:.3f})", "§10.5, §13.4"),
        ("Calibration", "Can predicted probabilities be read literally?",
         "; ".join(f"{t}: Brier skill {cal.loc[t, 'brier_skill']:.2f}, ECE {cal.loc[t, 'ece']:.3f}" for t in cal.index), "§13.8"),
        ("Bootstrap confidence intervals", "How certain is each test score?",
         "; ".join(f"{t}: {r.iloc[0].point:.3f} ({r.iloc[0].ci_low:.3f}–{r.iloc[0].ci_high:.3f})" for t, r in dep_ci.items() if len(r))
         + f"; {int((pairs.verdict.str.startswith('no clear')).sum())} of {len(pairs)} key model comparisons not established by the test set", "§13.9"),
        ("CLV benchmarks", "Is the forward CLV model better than simple or probabilistic alternatives?",
         f"deployed {dep.model}: R² {dep.test_r2:.2f}, top-10% capture {dep.top_decile_capture*100:.0f}% vs persistence R² {cbi.loc['Baseline · persistence (last 180 d)', 'test_r2']:.2f} "
         f"and BG/NBD × Gamma-Gamma R² {cbi.loc['BG/NBD × Gamma-Gamma (probabilistic)', 'test_r2']:.2f} (mean drop-out {L.fmt_small_p(pb['bgnbd']['mean_dropout_p'])})", "§14.3–§14.4"),
        ("CLV ranges", "Are the 80% ranges honest?", f"test coverage {iv['test_coverage_conformal']*100:.0f}% (raw quantile model {iv['test_coverage_raw_quantile']*100:.0f}%)", "§14.5"),
        ("Apriori robustness", "Are the affinities about the banners, or about activity and household?",
         f"{int(rules.affinity_type.str.startswith('household').sum())} of {len(rules)} rules household-driven; {int(rules.affinity_type.str.startswith('banner').sum())} genuine banner affinities", "§16.2"),
        ("Second segmentation view", "Do the segments reflect behaviour or just banner membership?",
         f"shopping-style view agrees partly (ARI {sc['ari_vs_banner_segments']:.2f}, NMI {sc['nmi_vs_banner_segments']:.2f}); churn by style p = {sc['churn_by_style_chi2_p']:.3f}", "§8.2"),
        ("Unit tests", "Does the shared code do what it claims?", f"{n_tests} tests in tests/test_core.py (incl. BG/NBD parameter recovery); run all checks with tests/run_all_checks.py", "tests/"),
    ]
    # a wrapping markdown table: the results are sentences, which a data grid would truncate
    md = "| Check | Question | Result | Where |\n|---|---|---|---|\n" + "\n".join(f"| **{c}** | {q} | {r.replace('|', '/')} | {w} |" for c, q, r, w in rows)
    st.markdown(md)
    rep = ROOT_DIR / "tests" / "compat_report.md"
    if rep.exists():
        with st.expander("Library-version compatibility report"):
            st.markdown(rep.read_text())


def page_method():
    hero(compact=True)
    section("Data quality and methodology", "How the numbers were produced, what was checked, and what the data cannot support.", "measured")
    t1, t2, t3, t6, t4, t5 = st.tabs(["Data quality", "Pipeline", "Viva & Key Ring", "Validation & robustness", "Assumptions & limits", "Compliance"])
    with t6:
        validation_summary()
    with t1:
        audit = js("data_audit.json"); rec = js("reconciliation.json")
        table(pd.DataFrame([{"file": a["file"], "rows": a["rows"], "columns": a["columns"], "missing cells": sum(a["missing"].values()), "duplicate rows": a["full_row_duplicates"],
                             "key unique": a["key_unique"], "date range": "; ".join(f"{k}: {v[0]} → {v[1]}" for k, v in a["date_range"].items())} for a in audit]))
        st.markdown("**Findings and decisions**"); st.table(csv("quality_findings.csv").set_index("finding"))
        st.markdown("**Transformation log**"); table(pd.DataFrame(js("transformation_log.json")))
        st.markdown("**Reconciliation**"); table(pd.Series(rec).astype(str).reset_index().rename(columns={"index": "check", 0: "value"}))
        with st.expander("Feature dictionary"):
            table(csv("feature_dictionary.csv"))
    with t2:
        p = ROOT_DIR / "pipeline_diagram" / "pipeline.png"
        if p.exists():
            st.image(str(p), caption="Implemented pipeline (solid = built; dashed = proposed or simulated)")
        j = ROOT_DIR / "pipeline_diagram" / "pipeline_justification.md"
        if j.exists():
            st.markdown(j.read_text())
    with t3:
        g = GAP
        cards([("Attempted join matches", f"{g['attempted_join_matches']}", "VR… vs SHK…"), ("Viva revenue", aed(g["viva_revenue_aed"]), f"{g['viva_share_full_period_pct']}% of full-period total"),
               ("Anonymous Viva baskets", f"{g['viva_anonymous_pct']}%", f"{g['viva_anonymous_rows']:,} rows"), ("Viva Rewards IDs", f"{g['viva_identified_members']:,}", "cannot be linked")])
        k = ROOT_DIR / "pipeline_diagram" / "key_ring.png"
        st.markdown(f"#### Key Ring identity bridge {badge('proposed')}", unsafe_allow_html=True)
        if k.exists():
            st.image(str(k))
        kr = js("key_ring_proposal.json")
        st.markdown(f"**Status:** {kr['status']}\n\n**Benefits:** " + "; ".join(kr["benefits"]) + "\n\n**Risks:** " + "; ".join(kr["risks"]))
    with t4:
        st.markdown("""
- **As-of date** is the latest recorded transaction (2026-10-01), not today's date. Q4-2026 contains one day and is never compared with complete quarters.
- **Recorded revenue** = sum of supplied `amount` values for a 3,000-member sample plus Viva baskets — not audited, not Group revenue.
- **Profit** cannot be measured (no cost/margin fields); contribution figures are user-entered scenarios.
- **Savings** cannot be measured (`discount_used` is a flag); the saving figure is a labelled hypothetical proxy.
- **Churn** is an observed status that closely follows a 90-day inactivity rule; the model uses behaviour 90 days earlier and mostly recognises lapses already under way. The out-of-time test found no transfer from earlier periods, so forward scores are a ranking hypothesis for a monitored pilot.
- **Scoring guard**: forward churn and CLV scores clip every feature to its training range (time-based features such as days since the first purchase in the file keep growing after training).
- **Routing** labels have no ties in this data (no member shops both banners of a pair); the 0.4/0.6 tie rule is documented for future data.
- **CLV**: historical spend is circular with frequency × ticket; the next-180-day model is the forward measure, shown with an 80% conformal range whose coverage assumes the next period resembles this one.
- **Apriori** baskets are member-level banner sets (no checkout IDs or SKUs) — affinity, not causation; several strong-looking rules are explained by household (families shop both family banners).
- **Test-set results** carry bootstrap intervals; differences between leading models are mostly within them.
- **Cost ratios, margins, discount depth and redemption probabilities** are editable assumptions, never Landmark data.
- **Pre-signup transactions** (6.3% of rows) are retained and flagged.
- **Viva** cannot be attributed to members; every customer-level result excludes grocery behaviour.""")
    with t5:
        st.markdown("**Final Project rubric (published weights — they sum to 106, not the stated 100; kept as published)**")
        table(csv("rubric_verification.csv"))
        st.markdown("**Automated QA checks (from the notebook run)**")
        qa = js("qa_checks.json")
        table(pd.DataFrame({"check": list(qa), "result": ["✔ pass" if v else "✖ FAIL" for v in qa.values()]}))
        with st.expander("Requirement-traceability matrix"):
            table(csv("traceability.csv"), height=420)
        with st.expander("Environment and artifact manifest"):
            st.json(js("env_versions.json"))
            man = json.loads((L.ART / "manifest.json").read_text())
            table(pd.DataFrame(man["files"]), height=300)


ROOT_DIR = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------------------------------------------
# Navigation
# ----------------------------------------------------------------------------------------------------------------
pages = {
    "Overview": [st.Page(page_overview, title="A · Executive overview", icon=":material/space_dashboard:", default=True, url_path="overview")],
    "Financials": [st.Page(page_financials, title="B · Group financials", icon=":material/account_balance:", url_path="financials"),
                   st.Page(page_banners, title="C · Eight-banner performance", icon=":material/leaderboard:", url_path="banners"),
                   st.Page(page_home, title="D · Home Centre vs Homebox", icon=":material/chair:", url_path="home-centre-vs-homebox"),
                   st.Page(page_fashion, title="E · Splash vs Max", icon=":material/checkroom:", url_path="splash-vs-max"),
                   st.Page(page_other, title="F · Other banners", icon=":material/storefront:", url_path="other-banners")],
    "Customers": [st.Page(page_segments, title="G · Segments & PCA", icon=":material/groups:", url_path="segments"),
                  st.Page(page_churn, title="H · Churn & retention", icon=":material/person_remove:", url_path="churn"),
                  st.Page(page_value, title="I · Customer value (CLV)", icon=":material/payments:", url_path="value"),
                  st.Page(page_shukran, title="J · Shukran loyalty", icon=":material/loyalty:", url_path="shukran"),
                  st.Page(page_acquisition, title="K · Acquisition & migration", icon=":material/trending_up:", url_path="acquisition")],
    "Action & models": [st.Page(page_bundles, title="L · Bundles & ε-sandbox", icon=":material/shopping_basket:", url_path="bundles"),
                        st.Page(page_lab, title="M · Model laboratory", icon=":material/science:", url_path="model-lab"),
                        st.Page(page_method, title="N · Data & methodology", icon=":material/fact_check:", url_path="methodology")],
}
nav = st.navigation(pages, position="sidebar", expanded=True)
st.session_state["F"] = sidebar_filters()
nav.run()
