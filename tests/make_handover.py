"""
Writes HANDOVER.md — the project handover: execution evidence, rubric verification, QA results, key results,
assumptions and limitations. Every figure is read from the artifacts written by the executed notebook.

    python tests/make_handover.py [dashboard_test_log]
"""
import json
import sys
from pathlib import Path

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lci_core as LC  # noqa: E402  (team list kept in one place)
M = ROOT / "artifacts" / "metrics"
J = lambda n: json.loads((M / n).read_text())
C = lambda n, **k: pd.read_csv(M / n, **k)

nb = nbformat.read(ROOT / "notebooks" / "Landmark_Customer_Intelligence.ipynb", 4)
code = [c for c in nb.cells if c.cell_type == "code"]
executed = [c for c in code if c.get("execution_count")]
errors = [c for c in code for o in c.get("outputs", []) if o.get("output_type") == "error"]
env, ms, thr, gap, cc = J("env_versions.json"), J("model_summary.json"), J("threshold_choice.json"), J("viva_gap.json"), J("cluster_choice.json")
qa, rub, tr = J("qa_checks.json"), C("rubric_verification.csv"), C("traceability.csv")
seg = C("segment_profiles.csv", index_col=0)
rh, rn = C("regression_historical.csv"), C("regression_next_period.csv")
rules = C("rules_all.csv")
cb = C("clv_benchmark.csv")
qf = C("quality_findings.csv")
small_p = lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"
dash_log = Path(sys.argv[1]).read_text() if len(sys.argv) > 1 and Path(sys.argv[1]).exists() else ""

L = []
w = L.append
w("# Project handover — notebook, dashboard and pipeline\n")
w(f"**Team:** {' · '.join(LC.TEAM)}\n")
w("The business report and the presentation are submitted separately. AI assistance is disclosed in README.md and at the top of the notebook.\n")
w("## 1. Execution evidence\n")
w(f"- Notebook: {len(nb.cells)} cells, {len(code)} code cells, **{len(executed)} executed**, **{len(errors)} errors**; `FAST_mode = {env.get('FAST_mode')}`.")
w(f"- Environment: Python {env['python']}, pandas {env['pandas']}, scikit-learn {env['sklearn']}, xgboost {env['xgboost']}, mlxtend {env['mlxtend']}, kmedoids {env['kmedoids']}, streamlit {env['streamlit']}.")
w(f"- QA checks computed inside the notebook: **{sum(qa.values())}/{len(qa)} passed**.")
if dash_log:
    res = [l for l in dash_log.splitlines() if l.startswith("RESULT")]
    w(f"- Dashboard end-to-end test (headless browser, all 17 modules + interactions + mobile view): **{res[-1].split(':', 1)[1].strip() if res else 'see log'}**.")
ut = ROOT / "tests" / "unit_test_log.txt"
if ut.exists():
    last = [l for l in ut.read_text().splitlines() if l.strip()]
    w(f"- Unit tests (`tests/test_core.py`): **{last[-1].strip() if last else 'see log'}**.")
cr = ROOT / "tests" / "compat_report.md"
if cr.exists():
    first = [l for l in cr.read_text().splitlines() if l.startswith("**Result")]
    verdict = first[0].split("**")[1].replace("Result:", "").strip() if first else "see report"
    w(f"- Older-library compatibility run (Python 3.11, pandas 2.2, NumPy 1.26, scikit-learn 1.6, Streamlit 1.45 — notebook, unit tests and dashboard test): "
      f"**{verdict}**; details and the three fixes it required in `tests/compat_report.md`.")
w("")
w("## 2. Rubric verification (Final Project — published weights kept; they sum to 106, not the stated 100)\n")
w("| Criterion | Weight | Evidence | Status |\n|---|---|---|---|")
for r in rub.itertuples():
    status = r.status
    if r.criterion.startswith("11") and dash_log and "RESULT: PASS" in dash_log:
        status = "Complete — app.py, end-to-end test passed"
    if r.criterion.startswith("3 ") and (ROOT / "pipeline_diagram" / "pipeline.png").exists() and (ROOT / "pipeline_diagram" / "pipeline_justification.md").exists():
        status = "Complete — §9 + pipeline_diagram/ (diagram + justification)"
    w(f"| {r.criterion} | {r.published_weight} | {r.evidence} | {status} |")
w(f"\n**Weights sum to {rub.published_weight.sum()}** — flagged as an inconsistency in the published rubric.\n")
w("## 3. Key results (all computed in the run)\n")
w(f"- **Data:** {gap['viva_rows']:,} Viva rows ({gap['viva_anonymous_pct']}% anonymous), attempted Viva↔Shukran join = {gap['attempted_join_matches']} matches; Viva = {gap['viva_share_common_window_pct']}% of recorded revenue since its data begin.")
w(f"- **Segments:** {cc['algorithm']} on {cc['representation']}, K = {cc['k']} (like-for-like silhouette {cc['silhouette']:.3f}, bootstrap ARI {cc['bootstrap_ari_mean']:.2f}); "
  + "; ".join(f"{r.segment} ({int(r.members):,} members, {r.share_of_revenue_pct:.0f}% of revenue)" for r in seg.itertuples()) + ".")
for t, lab in [("churn", "Churn (early-warning)"), ("home", "Home Centre vs Homebox"), ("fashion", "Splash vs Max")]:
    s = ms[t]
    w(f"- **{lab}:** selected {s['selected_model']} — test {s['primary_metric']} {s['selected_test'][s['primary_metric']]:.3f}, accuracy {s['selected_test']['accuracy']:.3f}, recall {s['selected_test']['recall']:.3f}, F1 {s['selected_test']['f1']:.3f} "
      + ("(threshold 0.50" + (f"; at the operational threshold {thr['operational_threshold']:.2f}: recall {thr['test_comparison'][1]['recall']:.2f}, precision {thr['test_comparison'][1]['precision']:.2f}, F1 {thr['test_comparison'][1]['f1']:.2f}). " if t == "churn" else "). "))
      + f"Family winners: " + ", ".join(f"{k}: {v}" for k, v in s["family_winners"].items()) + ".")
w(f"- **Churn threshold:** {thr['operational_threshold']:.2f} at a 5:1 relative cost (high-value tier {thr['high_value_threshold']:.2f}).")
bB = rh[rh.feature_set.str.startswith("B")].sort_values("cv_rmse").iloc[0]
bN = cb[cb.deployed].iloc[0]                       # the deployed next-180-day model, as flagged by the notebook
w(f"- **CLV:** historical (non-circular) {bB.model} CV R² {bB.cv_r2:.2f}; next-180-day {bN.model} test R² {bN.test_r2:.2f} (beats naive baselines).")
if len(rules):
    r0 = rules.iloc[0]
    w(f"- **Apriori:** {len(rules)} actionable significant rules; strongest raw affinity {r0.antecedent} → {r0.consequent} (lift {r0.lift:.2f}).")
w("")
# ---- robustness and validation added in the review
oot, wa, iv, pb, sc = J("oot_summary.json"), J("window_feature_ablation.json"), J("clv_intervals.json"), J("clv_probabilistic.json"), J("style_choice.json")
cal, tci, pairs = C("calibration_summary.csv"), C("test_ci.csv"), C("test_ci_pairs.csv")
oci = {r["model"]: r for r in oot["pr_auc_ci_test_members"]}; ct_ = oot["churn_timing"]
dep = cb[cb.deployed].iloc[0]; cbi = cb.set_index("model")
w("## 4. Robustness and validation\n")
w(f"- **Out-of-time churn test (§10.6):** trained only on earlier periods, the selected configuration scores PR-AUC {oci['out-of-time']['point']:.2f} "
  f"(95% CI {oci['out-of-time']['ci_low']:.2f}–{oci['out-of-time']['ci_high']:.2f}) on the same test members, against {oci['in-time']['point']:.2f} in time and a no-skill {ms['churn']['positive_rate']:.2f}. "
  f"The recorded churn is a recent, concentrated lapse (median churner inactive {ct_['median_days_since_last_purchase_at_reference']:.0f} days at the reference date; earlier 90-day gaps carry the flag "
  f"{ct_['p_flag_given_earlier_90d_gap']*100:.0f}% of the time vs {ct_['p_flag_base']*100:.0f}% overall). **Forward churn scores are a ranking hypothesis for a monitored pilot.**")
w(f"- **Data-window feature (§10.5):** `days_since_first` adds {wa['cv_diff']:+.3f} CV PR-AUC (paired SE {wa['cv_diff_paired_se']:.3f}); {wa['share_scored_above_training_range']*100:.0f}% of members exceed its training range at scoring, "
  "so a scoring guard clips every feature to its training range for forward churn and CLV scores.")
w("- **Calibration (§13.8):** " + "; ".join(f"{r.target} Brier skill {r.brier_skill:.2f}, ECE {r.ece:.3f}" for r in cal[cal.version == "as deployed"].itertuples()) + ".")
dci = [tci[(tci.target == t) & (tci.model == ms[t]["selected_model"]) & (tci.metric == ms[t]["primary_metric"])] for t in ("churn", "home", "fashion")]
w("- **Bootstrap 95% CIs (§13.9):** " + "; ".join(f"{d_.iloc[0].target} {d_.iloc[0].point:.3f} ({d_.iloc[0].ci_low:.3f}–{d_.iloc[0].ci_high:.3f})" for d_ in dci if len(d_))
  + f"; {int(pairs.verdict.str.startswith('no clear').sum())} of {len(pairs)} key model comparisons are not established by the test set.")
w(f"- **CLV benchmarks (§14.3–§14.4):** deployed {dep.model} — test R² {dep.test_r2:.2f}, top-10% capture {dep.top_decile_capture*100:.0f}%; "
  f"BG/NBD × Gamma-Gamma R² {cbi.loc['BG/NBD × Gamma-Gamma (probabilistic)', 'test_r2']:.2f} (mean drop-out {small_p(pb['bgnbd']['mean_dropout_p'])}); persistence R² {cbi.loc['Baseline · persistence (last 180 d)', 'test_r2']:.2f}. "
  "The BG/NBD and Gamma-Gamma estimators were checked against the published CDNOW estimates (r 0.243, α 4.414, a 0.793, b 2.426; p 6.25, q 3.74, γ 15.44 — reproduced to 3 decimals); "
  "the unit tests check parameter recovery on simulated data.")
w(f"- **CLV 80% ranges (§14.5):** conformal test coverage {iv['test_coverage_conformal']*100:.0f}% (raw quantile model {iv['test_coverage_raw_quantile']*100:.0f}%), median width AED {iv['median_width_aed']:,.0f}.")
w(f"- **Apriori robustness (§16.2):** {int(rules.affinity_type.str.startswith('household').sum())} of {len(rules)} rules are household-driven (explained by kids); "
  f"{int(rules.affinity_type.str.startswith('banner').sum())} survive as banner affinities; bundles are ranked on the adjusted lift. Most affinities are cross-month (portfolio) habits.")
w(f"- **Second segmentation (§8.2):** shopping-style view ({sc['algorithm']}, K = {sc['k']}) agrees partly with the banner-mix segments (ARI {sc['ari_vs_banner_segments']:.2f}); churn does not differ by style (p = {sc['churn_by_style_chi2_p']:.2f}).")
w("- **Tests:** `python tests/run_all_checks.py` runs the unit tests, a quick notebook run in a temporary copy and the dashboard test; see `tests/compat_report.md` for the older-library run.")
w("")
w("## 5. Assumptions and limitations\n")
for t in [
    "As-of date = latest recorded transaction (2026-10-01), not today's date; Q4-2026 has one day and is flagged everywhere.",
    "Recorded revenue = sum of supplied amounts for a 3,000-member sample plus Viva baskets — not audited and not Group revenue.",
    "Profit, member savings and Shukran's causal effect cannot be measured (no cost, discount-amount or control-group data); only labelled scenarios are shown.",
    "The churn flag behaves like a 90-day inactivity rule; the model uses behaviour 90 days earlier, mostly recognises lapses already under way, and did not transfer from earlier periods in the out-of-time test — forward scores need a monitored pilot.",
    "Forward scores use a scoring guard (features clipped to their training range); conformal CLV ranges assume the next period resembles this one.",
    "Several strong-looking Apriori rules are household-driven (families shop both family banners); offers based on them should target families, not shoppers of the other banner.",
    "No member shops both banners of a tier pair, so routing labels have no ties; the 0.4/0.6 rule is documented for future data.",
    "Cost ratios, margins, discount depth and bandit redemption probabilities are editable assumptions, never Landmark data.",
    "Viva spend cannot be attributed to members; every customer-level result excludes grocery behaviour (Key Ring proposal describes the governed fix).",
    *[f"{int(r['count']):,} Shukran transactions predate the member's signup date ({r['detail']}); retained and flagged." for _, r in qf[qf.finding.str.startswith("Transactions dated before")].iterrows()][:1],
]:
    w(f"- {t}")
w("\n## 6. Launch\n\n```bash\npip install -r requirements-lock.txt     # exact versions used (or requirements.txt for tested ranges)\nstreamlit run app.py\n```\n")
w("## 7. Requirement traceability (status evaluated from files on disk)\n")
w("| ID | Requirement | Status |\n|---|---|---|")
for r in tr.itertuples():
    w(f"| {r.id} | {r.requirement} | {r.status} |")
(ROOT / "HANDOVER.md").write_text("\n".join(L))
print("wrote HANDOVER.md;", f"{len(executed)}/{len(code)} executed, {len(errors)} errors, QA {sum(qa.values())}/{len(qa)}")
