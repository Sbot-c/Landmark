# Project handover — notebook, dashboard and pipeline

**Team:** Sparsh Saraya · Shashaank Sharma · Sanjay Seemakurti · Shirley Waeni Kioko · Apurba Dash

The business report and the presentation are submitted separately. AI assistance is disclosed in README.md and at the top of the notebook.

## 1. Execution evidence

- Notebook: 191 cells, 119 code cells, **119 executed**, **0 errors**; `FAST_mode = False`.
- Environment: Python 3.13.16, pandas 3.0.5, scikit-learn 1.9.1, xgboost 3.4.1, mlxtend 0.25.0, kmedoids 0.5.5, streamlit 1.66.1.
- QA checks computed inside the notebook: **37/37 passed**.
- Dashboard end-to-end test (headless browser, all 14 modules + interactions + mobile view): **PASS**.
- Unit tests (`tests/test_core.py`): **22 passed in 8.82s**.
- Older-library compatibility run (Python 3.11, pandas 2.2, NumPy 1.26, scikit-learn 1.6, Streamlit 1.45 — notebook, unit tests and dashboard test): **PASS**; details and the three fixes it required in `tests/compat_report.md`.

## 2. Rubric verification (Final Project — published weights kept; they sum to 106, not the stated 100)

| Criterion | Weight | Evidence | Status |
|---|---|---|---|
| 1 EDA, cleaning & unified customer view | 10 | §2–§5 | Complete |
| 2 PCA | 10 | §7, §8, §13.5 | Complete |
| 3 Pipeline design & justification | 6 | §9 + pipeline_diagram/ (diagram + justification) | Complete — §9 + pipeline_diagram/ (diagram + justification) |
| 4 Segmentation | 10 | §8 (+ second view §8.2) | Complete |
| 5 Classification incl. SVM, no leakage | 10 | §10–§13 (robustness §10.5–§10.6, calibration §13.8, CIs §13.9) | Complete |
| 6 Ensemble methods | 10 | §10–§12 (four families) + paired CIs §13.9 | Complete — see 'ensembles vs singles' verdicts |
| 7 Bias–variance analysis | 8 | §15 | Complete |
| 8 Regression — CLV | 6 | §14 (benchmarks §14.4, intervals §14.5) | Complete |
| 9 Apriori | 6 | §16 (robustness §16.2) | Complete |
| 10 Viva blind spot + second special analysis | 6 | §4, §18.3, §19 | Complete (discount dependency AND migration) |
| 11 Dashboard | 8 | app.py — modules A–N (tests/test_dashboard.py) | Complete — app.py, end-to-end test passed |
| 12 Business report | 6 | Separate deliverable | Separate deliverable |
| 13 Presentation / viva | 10 | Separate deliverable | Separate deliverable |

**Weights sum to 106** — flagged as an inconsistency in the published rubric.

## 3. Key results (all computed in the run)

- **Data:** 9,000 Viva rows (34.87% anonymous), attempted Viva↔Shukran join = 0 matches; Viva = 12.03% of recorded revenue since its data begin.
- **Segments:** K-means (Euclidean) on raw (15 std. features), K = 3 (like-for-like silhouette 0.218, bootstrap ARI 0.99); Family multi-banner shoppers (978 members, 23% of revenue); Premium home-led shoppers (1,013 members, 52% of revenue); Value home & discount-led shoppers (1,009 members, 25% of revenue).
- **Churn (early-warning):** selected Voting (soft) — test pr_auc 0.548, accuracy 0.769, recall 0.166, F1 0.267 (threshold 0.50; at the operational threshold 0.23: recall 0.80, precision 0.42, F1 0.55). Family winners: Single: SVM (RBF), Bagging: Random Forest, Boosting: XGBoost, Voting: Voting (soft), Stacking: Stacking (LR meta).
- **Home Centre vs Homebox:** selected Logistic Regression — test roc_auc 0.631, accuracy 0.600, recall 0.419, F1 0.486 (threshold 0.50). Family winners: Single: SVM (linear), Bagging: Extra Trees, Boosting: XGBoost, Voting: Voting (soft), Stacking: Stacking (LR meta).
- **Splash vs Max:** selected Logistic Regression — test roc_auc 0.609, accuracy 0.593, recall 0.299, F1 0.393 (threshold 0.50). Family winners: Single: Logistic Regression, Bagging: Extra Trees, Boosting: Gradient Boosting, Voting: Voting (soft), Stacking: Stacking (LR meta).
- **Churn threshold:** 0.23 at a 5:1 relative cost (high-value tier 0.05).
- **CLV:** historical (non-circular) Gradient Boosting CV R² 0.73; next-180-day Gradient Boosting test R² 0.26 (beats naive baselines).
- **Apriori:** 19 actionable significant rules; strongest raw affinity Funcity + Home Centre → Babyshop (lift 1.87).

## 4. Robustness and validation

- **Out-of-time churn test (§10.6):** trained only on earlier periods, the selected configuration scores PR-AUC 0.27 (95% CI 0.25–0.32) on the same test members, against 0.55 in time and a no-skill 0.25. The recorded churn is a recent, concentrated lapse (median churner inactive 59 days at the reference date; earlier 90-day gaps carry the flag 20% of the time vs 25% overall). **Forward churn scores are a ranking hypothesis for a monitored pilot.**
- **Data-window feature (§10.5):** `days_since_first` adds +0.026 CV PR-AUC (paired SE 0.005); 43% of members exceed its training range at scoring, so a scoring guard clips every feature to its training range for forward churn and CLV scores.
- **Calibration (§13.8):** churn Brier skill 0.18, ECE 0.040; home Brier skill 0.05, ECE 0.044; fashion Brier skill 0.04, ECE 0.039.
- **Bootstrap 95% CIs (§13.9):** churn 0.548 (0.487–0.615); home 0.631 (0.591–0.674); fashion 0.609 (0.569–0.651); 4 of 5 key model comparisons are not established by the test set.
- **CLV benchmarks (§14.3–§14.4):** deployed Gradient Boosting — test R² 0.26, top-10% capture 24%; BG/NBD × Gamma-Gamma R² 0.05 (mean drop-out < 0.001); persistence R² -0.56. The BG/NBD and Gamma-Gamma estimators were checked against the published CDNOW estimates (r 0.243, α 4.414, a 0.793, b 2.426; p 6.25, q 3.74, γ 15.44 — reproduced to 3 decimals); the unit tests check parameter recovery on simulated data.
- **CLV 80% ranges (§14.5):** conformal test coverage 75% (raw quantile model 73%), median width AED 1,374.
- **Apriori robustness (§16.2):** 12 of 19 rules are household-driven (explained by kids); 7 survive as banner affinities; bundles are ranked on the adjusted lift. Most affinities are cross-month (portfolio) habits.
- **Second segmentation (§8.2):** shopping-style view (K-means (Euclidean), K = 3) agrees partly with the banner-mix segments (ARI 0.31); churn does not differ by style (p = 0.92).
- **Tests:** `python tests/run_all_checks.py` runs the unit tests, a quick notebook run in a temporary copy and the dashboard test; see `tests/compat_report.md` for the older-library run.

## 5. Assumptions and limitations

- As-of date = latest recorded transaction (2026-10-01), not today's date; Q4-2026 has one day and is flagged everywhere.
- Recorded revenue = sum of supplied amounts for a 3,000-member sample plus Viva baskets — not audited and not Group revenue.
- Profit, member savings and Shukran's causal effect cannot be measured (no cost, discount-amount or control-group data); only labelled scenarios are shown.
- The churn flag behaves like a 90-day inactivity rule; the model uses behaviour 90 days earlier, mostly recognises lapses already under way, and did not transfer from earlier periods in the out-of-time test — forward scores need a monitored pilot.
- Forward scores use a scoring guard (features clipped to their training range); conformal CLV ranges assume the next period resembles this one.
- Several strong-looking Apriori rules are household-driven (families shop both family banners); offers based on them should target families, not shoppers of the other banner.
- No member shops both banners of a tier pair, so routing labels have no ties; the 0.4/0.6 rule is documented for future data.
- Cost ratios, margins, discount depth and bandit redemption probabilities are editable assumptions, never Landmark data.
- Viva spend cannot be attributed to members; every customer-level result excludes grocery behaviour (Key Ring proposal describes the governed fix).
- 1,844 Shukran transactions predate the member's signup date (6.3% of rows; 597 customers; median 41 days early); retained and flagged.

## 6. Launch

```bash
pip install -r requirements-lock.txt     # exact versions used (or requirements.txt for tested ranges)
streamlit run app.py
```

## 7. Requirement traceability (status evaluated from files on disk)

| ID | Requirement | Status |
|---|---|---|
| R01 | Read all five files; detect names with '(1)' | Complete |
| R02 | Audit & traceability before modelling | Complete |
| R03 | Per-file audit: shape, dtypes, missing, duplicates, uniques, stats, skew, outliers, dates, ID formats | Complete |
| R04 | Outliers judged against business context; every transformation logged; raw data preserved | Complete |
| R05 | Relationship checks: PK, FK, unique transaction IDs, 7 banners, separate Viva IDs, no join inflation, reconciliation | Complete |
| R06 | Unified one-row-per-customer view: profile, RFM, avg ticket, discount rate, banners, categories, banner & category shares, cohorts, bands | Complete |
| R07 | Explicit observation (as-of) date — not 'today' | Complete |
| R08 | Feature dictionary + correlation / redundancy / low-variance / leakage screen | Complete |
| R09 | Viva blind spot: attempted join, matched IDs, revenue & anonymous share, coverage, ≥2 bias mechanisms, alternative matching | Complete |
| R10 | Stage-0 reflections (skew, data quality, why Viva can't be joined, downstream bias) | Complete (notebook) |
| R11 | EDA visuals E1–E10 with implications | Complete |
| R12 | Group revenue: annual, quarterly, volumes, ticket, growth, partial-period flags, like-for-like | Complete |
| R13 | Rank all eight banners (revenue, volume, ticket, share, reach; Viva reach labelled not comparable) | Complete |
| R14 | Profit unavailable; editable margin scenarios; ranking change under assumptions | Complete |
| R15 | Sections titled exactly 'Home Centre vs Homebox' and 'Splash vs Max'; separate overviews for Babyshop, Centrepoint, Funcity, Viva | Complete |
| R16 | PCA: standardise in a pipeline, scree, cumulative & individual variance, ~90% rule, PC1–PC2 plot, loadings interpreted, cost/benefit | Complete |
| R17 | Clustering raw vs PCA compared | Complete |
| R18 | K-means (Euclidean) vs Manhattan algorithm; silhouette compared like-for-like (same space, both metrics; own-metric score also reported) | Complete |
| R19 | K chosen by elbow + silhouette (+ min share, stability) | Complete |
| R20 | Segment profiles & evidence-based business names | Complete |
| R21 | Churn target (observed status) with leakage audit and early-warning temporal design | Complete |
| R22 | Home Centre vs Homebox target; tie rule; leakage audit; permitted predictors | Complete |
| R23 | Splash vs Max target (advanced) | Complete |
| R24 | Nine single models (LR, DT, KNN, SVM-linear, SVM-RBF, GNB, RF, ET, MLP) with scaling in pipelines | Complete |
| R25 | Bagging, boosting, voting (soft vs hard), stacking; winner per family | Complete |
| R26 | Common protocol: stratified split, CV on train, tuning, untouched test, seeds, timings | Complete |
| R27 | Full metric table + separate numeric rankings (accuracy, precision, recall, F1, discrimination) | Complete |
| R28 | Why the winner wins; materiality; simpler-model check; permutation importance | Complete |
| R29 | PCA inside supervised pipelines (with vs without) | Complete |
| R30 | Churn: baseline, 0.5 threshold, sweep, PR & threshold curves, validation-tuned threshold, cost framework, value tiers, operational threshold | Complete |
| R31 | Confusion matrices (counts & normalised) with TP/TN/FP/FN business meaning | Complete |
| R32 | MLP activation comparison (identity, logistic, tanh, relu) incl. convergence & cost | Complete |
| R33 | CLV historical regression: Linear, Ridge, Lasso, ElasticNet + nonlinear; MAE/RMSE/R²; residuals | Complete |
| R34 | CLV circularity audit + next-period model with cutoff and naive baseline | Complete |
| R35 | Bias–variance: train/CV/test, learning & validation curves, diagnostics table, bagging/boosting link | Complete |
| R36 | Apriori on customer-level banner baskets: threshold sweep, support/confidence/lift, pruning, per segment, offers | Complete |
| R37 | Epsilon method — offer sandbox labelled SIMULATION | Complete |
| R38 | Key Ring identity architecture — PROPOSED, no links manufactured | Complete |
| R39 | Shukran: coverage, cohorts, churn by driver, discount-linked spend, savings unavailable, helped/harmed | Complete |
| R40 | Discount dependency: discount-only vs full-price loyalists; margin-eroding segments | Complete |
| R41 | Segment migration / upsell candidates (Homebox→Home Centre, Max→Splash), labelled candidates | Complete |
| R42 | Answers to the 12 customer-intelligence questions | Complete |
| R43 | Fine-tuning throughout (K, PCs, C/kernel, thresholds, regularisation, support/confidence) | Complete |
| R44 | Bundles per segment (advanced) | Complete |
| R45 | Pipeline diagram + 2–3 sentence dependency justification | Complete |
| R46 | Dashboard modules A–N reading these artifacts | Complete |
| R47 | Business report / memo / presentation | Separate deliverable |
| R48 | Pinned, tested library versions (lock file) and an older-stack compatibility run | Complete |
| R49 | Data-window feature: evidence, paired ablation, extrapolation check, scoring guard for forward scores | Complete |
| R50 | Out-of-time churn validation (train on earlier periods, test on the supplied flag) | Complete |
| R51 | Probability calibration of the selected classifiers (reliability, Brier, ECE, risk-band check) | Complete |
| R52 | Bootstrap confidence intervals for every test result and paired model comparisons | Complete |
| R53 | Forward CLV: alternative losses, two-part model, BG/NBD × Gamma-Gamma benchmark, ranking metrics | Complete |
| R54 | Forward CLV prediction intervals (conformalised) with test coverage | Complete |
| R55 | Apriori robustness: activity- and household-adjusted lift (CMH), customer-month baskets | Complete |
| R56 | Second segmentation view on shopping style only, compared with the banner-mix segments | Complete |
| R57 | Unit tests for the shared module and a one-command verification script | Complete |