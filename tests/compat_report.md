# Library-version compatibility report

**Result: PASS** — the notebook (quick mode), the 22 unit tests and the dashboard end-to-end test all run on an older stack (Python 3.11.17, pandas 2.2.3, NumPy 1.26.4, scikit-learn 1.6.1, Streamlit 1.45.1) as well as on the locked current stack. Three small code changes were needed; they are listed below and are part of the delivered code.

## Why this check exists

The submitted run used the newest libraries (`requirements-lock.txt`). A marker or a colleague may have an older environment, so the project was re-run end to end in a separate Python 3.11 virtual environment with older releases, and `requirements.txt` now states the tested range (lower bound = the older stack below).

## Environments

| Package | Older stack (this check) | Current stack (submitted run, `requirements-lock.txt`) |
|---|---|---|
| Python | 3.11.17 | 3.13 |
| pandas | 2.2.3 | 3.0.5 |
| NumPy | 1.26.4 | 2.5.3 |
| SciPy | 1.13.1 | 1.18.1 |
| scikit-learn | 1.6.1 | 1.9.1 |
| mlxtend | 0.23.4 | 0.25.0 |
| XGBoost | 2.1.4 | 3.4.1 |
| kmedoids | 0.5.3.1 | 0.5.5 |
| matplotlib | 3.8.4 | 3.11.2 |
| Streamlit | 1.45.1 | 1.66.1 |
| Plotly | 5.24.1 | 7.1.0 |

## What was run on the older stack

| Check | Command | Outcome |
|---|---|---|
| Notebook, quick mode (`LCI_FAST=1`), executed by the Python 3.11 kernel | `python -m nbconvert --execute ...` | 191 cells, 119 code, executed 119; no errors, no stderr output; all QA checks passed (37/37) |
| Unit tests | `python -m pytest tests/test_core.py -q` | 22 passed in 6.74s |
| Dashboard end-to-end test, app served by the Python 3.11 environment | `tests/test_dashboard.py` | PASS — 14 modules, 0 exceptions; interactions (cost slider, style view, robustness tabs, model-lab CI tab, mobile view) |

The 404 messages counted as browser *console errors* in the dashboard log are Streamlit's own requests for `<page>/_stcore/health` and `<page>/_stcore/host-config` when a page is opened by its direct URL; they occur on both stacks and do not come from the app.

## Fixes needed for the older stack (all included in the delivered code)

| # | Symptom on the older stack | Cause | Fix |
|---|---|---|---|
| 1 | Notebook stopped in §8 (clustering grid): `TypeError: Cannot cast array data from dtype('uint64') to dtype('int64')` | kmedoids < 0.5.4 returns unsigned labels; `np.bincount` under NumPy 1.26 refuses them | labels cast to `int` wherever K-medoids labels are used (`part04`, four places) |
| 2 | Dashboard tables failed with `TypeError: 'str' object cannot be interpreted as an integer`, so the dashboard test failed | in Streamlit 1.45 `st.dataframe` / `st.data_editor` take an integer `width` (`width="stretch"` arrived later); `st.plotly_chart` only tolerated it as a legacy keyword | `app.py` reads the installed signature and passes `width="stretch"` or `use_container_width=True` accordingly (`_full_width`) |
| 3 | `DeprecationWarning` (array-to-scalar conversion) in two notebook cells | matplotlib 3.8 + NumPy 1.26: value labels positioned by category name on a bar chart | labels positioned by bar index (§5 E9 and §18.1 figures); figures unchanged |

Found during the same review (not version-specific): one JSON artifact contained `NaN`, which is not valid JSON for non-Python readers. `lci_core.save_json` now writes `null`, a QA check verifies every JSON artifact is standard JSON, and a unit test covers it.

## Do the results depend on the library versions?

Both columns are **quick-mode** runs (smaller grids, 3-fold CV) of the same code, so they are directly comparable; the submitted full-mode results are in the notebook and HANDOVER.md.

| Result | Older stack | Current stack |
|---|---|---|
| QA checks passed | 37/37 | 37/37 |
| Segmentation selected | K-means (Euclidean) · raw (15 std. features) · K=3 | K-means (Euclidean) · raw (15 std. features) · K=3 |
| Shopping-style view | K-means (Euclidean) K=3 ARI=0.31 | K-means (Euclidean) K=3 ARI=0.31 |
| Churn model selected (1-SE rule) | XGBoost | XGBoost |
| Churn CV PR-AUC | 0.520 | 0.523 |
| Churn test PR-AUC | 0.565 | 0.540 |
| Churn operational threshold | 0.192 | 0.172 |
| Out-of-time: transfers? | no | no |
| Out-of-time test PR-AUC | 0.269 | 0.278 |
| Home routing model / test ROC-AUC | Logistic Regression / 0.631 | Logistic Regression / 0.631 |
| Fashion routing model / test ROC-AUC | Logistic Regression / 0.609 | Logistic Regression / 0.609 |
| Churn calibration: Brier skill | 0.197 | 0.186 |
| CLV 80% range coverage (test) | 0.754 | 0.752 |
| Apriori rules household-driven | 12/19 | 12/19 |
| Data-window ablation (CV PR-AUC with → without) | 0.520 → 0.489 | 0.523 → 0.485 |

8 of 15 results are identical, including every structural finding — segmentation choice, shopping-style view, routing models, the out-of-time verdict, household-driven rules. The differences sit in results that depend on the boosted-tree models (XGBoost 2.1 vs 3.4, scikit-learn 1.6 vs 1.9): churn PR-AUC moves by up to 0.025 (well inside its bootstrap 95% CI of about ±0.06, §13.9), calibration and out-of-time PR-AUC by about 0.01, and the cost-optimal churn threshold by 0.02. None of these changes a conclusion.

Quick mode and full mode can select different near-tied churn models: the submitted full run deploys **Voting (soft)**, both quick runs deploy XGBoost — the full-mode grids and 5-fold CV, not the library versions, cause that difference. In full mode XGBoost's CV PR-AUC (0.525) falls just below the one-standard-error band of the best model (Stacking, 0.538 − 0.008 = 0.529), so the rule takes Voting (soft) (0.536), whose test PR-AUC (0.548) lies inside XGBoost's interval.

## How to reproduce

```bash
python3.11 -m venv compat && source compat/bin/activate
pip install pandas==2.2.3 numpy==1.26.4 scipy==1.13.1 scikit-learn==1.6.1 mlxtend==0.23.4 \
            xgboost==2.1.4 kmedoids==0.5.3.1 matplotlib==3.8.4 seaborn streamlit==1.45.1 plotly==5.24.1 \
            pyarrow==17.0.0 joblib nbconvert ipykernel pytest playwright
python tests/run_all_checks.py      # unit tests + quick notebook run in a temporary copy + dashboard test
```
