# Landmark Retail Group — Customer Intelligence Engine

MAIB Machine Learning · Final Capstone — analysis notebook, Streamlit dashboard and pipeline diagram.

**Team:** Sparsh Saraya · Shashaank Sharma · Sanjay Seemakurti · Shirley Waeni Kioko · Apurba Dash

The business report and the presentation are submitted separately.

**AI-assistance disclosure.** The notebook, the dashboard (`app.py`) and the supporting code were developed with the help of an AI
assistant (Claude, by Anthropic), working from the team's written brief; requirements that brief adds beyond the two project
specifications are labelled *TS (team scope)* in the notebook's §0. Every result is computed by the notebook from the supplied data.
The team is responsible for the analysis choices and their interpretation, and for defending them at the viva.

## What is in this folder

```
Landmark_Customer_Intelligence/
├── data/                         the three supplied CSVs, unchanged (names with "(1)" are also accepted)
├── lci_core.py                   shared logic: loading, schema checks, customer view, financial helpers, ε-greedy simulator,
│                                 bootstrap CIs, calibration, adjusted lift, BG/NBD × Gamma-Gamma, two-part CLV model
├── notebooks/
│   ├── Landmark_Customer_Intelligence.ipynb   fully executed analysis (outputs visible)
│   ├── Landmark_Customer_Intelligence.html    the same notebook rendered for reading without Jupyter
│   └── src/                                   cell sources + build_notebook.py (regenerates the .ipynb)
├── app.py                        Streamlit dashboard (modules A–Q, theme-aware light/dark); reads artifacts only, never retrains
├── artifacts/
│   ├── metrics/                  every table / JSON the dashboard shows (written by the notebook)
│   ├── models/                   fitted pipelines (.joblib): churn, routing ×2, CLV
│   ├── visualisations/           every notebook figure (.png)
│   └── manifest.json             file list with MD5 hashes and row counts
├── pipeline_diagram/             pipeline + Key Ring diagrams (.dot editable, .png, .svg) and the dependency justification
├── tests/                        test_core.py (unit tests), run_all_checks.py (one-command verification), test_dashboard.py
│                                 (headless end-to-end test), check_notebook.py, make_handover.py, compat_report.md, screenshots/
├── HANDOVER.md                   execution evidence, rubric verification, QA results, limitations (generated)
├── .streamlit/config.toml        dashboard theme
├── requirements.txt              tested version ranges (current stack and an older stack — see tests/compat_report.md)
├── requirements-lock.txt         exact versions used for the submitted run
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt # exact versions of the submitted run (Python 3.13)
# or: pip install -r requirements.txt  (ranges also tested on Python 3.11 with older libraries)
```
Graphviz (`dot`) is only needed to re-draw the pipeline diagram (`brew install graphviz` / `apt install graphviz`).

## Data placement
Put `customers.csv`, `transactions.csv` and `viva_transactions.csv` in `data/`. Uploaded copies named
`customers(1).csv` etc. are detected automatically. The raw files are never modified — the notebook records
their MD5 hashes at load time and verifies them again at the end.

## Re-run the analysis (optional — artifacts are already included)

```bash
cd notebooks
jupyter nbconvert --to notebook --execute Landmark_Customer_Intelligence.ipynb --inplace --ExecutePreprocessor.timeout=-1
cd ..
python pipeline_diagram/build_diagram.py      # redraws the diagrams from the new artifacts
```
The submitted full run took about 43 minutes on a 2-core machine (three model laboratories with grid search, plus the robustness checks).
Setting `LCI_FAST=1` runs a quick check with smaller grids and 3-fold CV; the submitted artifacts were produced with `FAST = False`.

## Launch the dashboard

```bash
streamlit run app.py
```
Then open the URL Streamlit prints (normally http://localhost:8501). Navigation is in the sidebar
(A Executive overview → O Recommendation & action plan → … → N Data & methodology); global filters (period, banners, segment, churn status,
spend range, discount behaviour) sit below the navigation.

## Verify
```bash
python tests/run_all_checks.py          # one command: unit tests + quick notebook run (temporary copy) + dashboard test
python tests/run_all_checks.py --full   # same with the full notebook run (~45 min)

python -m pytest tests/test_core.py -q  # unit tests of the shared module only (22 tests, ~10 s)
python tests/check_notebook.py          # execution coverage, errors, warnings in the executed notebook
python tests/test_dashboard.py          # launches the app headless, opens all 14 modules, exercises controls, saves screenshots
python tests/make_handover.py           # regenerates HANDOVER.md from the artifacts
```
(`test_dashboard.py` needs Playwright with Chromium: `pip install playwright && playwright install chromium`.)

## Key design decisions (details in the notebook)
* **As-of date** = latest recorded transaction (2026-10-01), not today's date. Q4-2026 has one day and is flagged.
* **Churn**: the supplied flag is almost exactly "no purchase in the last 90 days", so the model uses an
  **early-warning design** — behaviour up to 90 days before the as-of date predicts the status 90 days later. In this
  period it mostly recognises lapses already under way; the out-of-time test shows it does not yet transfer to a new period.
* **Segmentation**: K-means and Manhattan K-medoids on raw and PCA features are compared with a *like-for-like* silhouette
  (every partition scored in the same space under both Euclidean and Manhattan distance), not each algorithm's own metric.
  A second, behaviour-only *shopping-style* view (frequency, spend, ticket size, discount use, banner count) is a descriptive
  cross-check of the banner-mix segments; it is never used as a model input.
* **Banner routing**: features are recomputed on **non-home / non-fashion** transactions only; the leakage
  audit shows how direct and indirect leakage would fake a ~99% score.
* **Model selection**: nine single learners + bagging, boosting, voting, stacking under one CV protocol;
  the deployed model is chosen by the one-standard-error rule.
* **CLV**: historical spend (with a circularity audit) plus a forward next-180-day model against naive baselines.
* **Robustness layer** (added after review): out-of-time churn test, paired ablation of the one feature tied to the data
  window, probability calibration, bootstrap confidence intervals for every test score, a scoring guard that clips features to
  their training range, activity/household-adjusted Apriori lift, CLV benchmarks (incl. BG/NBD × Gamma-Gamma) and conformal
  80% CLV ranges. Key finding: the churn model does **not** transfer from earlier periods, so forward scores need a monitored pilot.
* **Review addendum (§22):** churn-label audit, class balance of all targets, an imbalance-aware re-test of the ensembles (`class_weight` / `scale_pos_weight`) and a nested-CV check of the churn leaderboard. None of them changes a selection or threshold; results are in `artifacts/metrics/` and the dashboard.
* **Viva** is never joined to Shukran customers (no shared key); the *Key Ring* identity bridge is a proposal.
* **ε-greedy** offer sandbox is a **simulation** with hypothetical redemption rates.
* Profit, member savings and Shukran's causal effect are **not measurable** from these files; only labelled
  scenarios are shown.


## Dashboard additions (post-notebook)
- `scripts/compute_member_drivers.py` re-scores the saved churn model with one feature group at a time set to the median member and writes
  `artifacts/metrics/member_churn_drivers.csv` (used by **Member 360 → "Why this risk score?"**). It reproduces the stored risk scores exactly (max difference ~1e-16).
  Re-run it whenever `churn_final.joblib` or `customer_master.csv` is regenerated.
- **What-if simulator** (module Q) includes a sensitivity tornado and a save-rate × contact-cost break-even map; every response rate is a user assumption.
- Filters are local to each page (expander at the top, with a reset button); Overview, Recommendation and Member 360 are whole-base views.
