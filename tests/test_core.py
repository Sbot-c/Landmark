"""
Unit tests for lci_core.py — the shared logic used by both the notebook and the dashboard.

    python -m pytest tests/test_core.py -q

Tests on the supplied data check reconciliation; tests on synthetic data check that each statistical helper
does what its docstring claims (e.g. the BG/NBD estimator recovers the parameters it was simulated with).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lci_core as L  # noqa: E402


# ------------------------------------------------------------------ data loading and the customer view
@pytest.fixture(scope="module")
def raw():
    cust, tx, viva, paths = L.load_raw()
    return cust, tx, viva


def test_find_csv_accepts_upload_suffix(tmp_path):
    (tmp_path / "customers(1).csv").write_text("customer_id\n")
    assert L.find_csv("customers", tmp_path).name == "customers(1).csv"
    with pytest.raises(FileNotFoundError):
        L.find_csv("transactions", tmp_path)


def test_raw_files_have_expected_shape(raw):
    cust, tx, viva = raw
    assert cust.customer_id.is_unique
    assert tx.transaction_id.is_unique
    assert set(tx.banner) == set(L.CORE_BANNERS)
    assert tx.customer_id.isin(cust.customer_id).all()          # no orphan transactions


def test_customer_view_reconciles_to_source(raw):
    cust, tx, _ = raw
    obs = tx.date.max()
    cv = L.build_customer_view(cust, tx, obs)
    assert len(cv) == len(cust) and cv.index.is_unique
    assert np.isclose(cv.monetary.sum(), tx.amount.sum())
    assert cv.frequency.sum() == len(tx)
    shares = cv[[f"bshare_{b}" for b in L.CORE_BANNERS]].sum(axis=1)
    assert np.allclose(shares[cv.has_tx == 1], 1.0)


def test_customer_view_ignores_transactions_after_reference_date(raw):
    cust, tx, _ = raw
    ref = tx.date.max() - pd.Timedelta(days=180)
    cv = L.build_customer_view(cust, tx, ref)
    assert cv.frequency.sum() == (tx.date <= ref).sum()
    assert (cv.recency_days.dropna() >= 0).all()


# ------------------------------------------------------------------ labels and financial helpers
def _tx(rows):
    return pd.DataFrame(rows, columns=["customer_id", "date", "banner", "amount"]).assign(date=lambda d: pd.to_datetime(d.date))


def test_routing_label_rule_and_ambiguity():
    t = _tx([("a", "2025-01-01", "Home Centre", 70), ("a", "2025-01-02", "Homebox", 30),      # share 0.7 → premium
             ("b", "2025-01-01", "Home Centre", 50), ("b", "2025-01-02", "Homebox", 50),      # share 0.5 → ambiguous
             ("c", "2025-01-01", "Home Centre", 20), ("c", "2025-01-02", "Homebox", 80)])     # share 0.2 → value
    y, share, info = L.routing_label(t, "Home Centre", "Homebox")
    assert y.to_dict() == {"a": 1, "c": 0}
    assert info["ambiguous_excluded"] == 1 and info["shop_both_banners"] == 3


def test_period_coverage_flags_partial_quarters():
    d = pd.DataFrame({"date": pd.to_datetime(["2024-04-15", "2024-06-30", "2024-07-01", "2024-09-30"]), "amount": [1, 1, 1, 1]})
    cov = L.period_coverage(d).set_index("period")
    assert bool(cov.loc["2024Q2", "partial"]) and not bool(cov.loc["2024Q3", "partial"])
    # with an explicit collection window the first quarter is complete
    cov2 = L.period_coverage(d, lo="2024-04-01", hi="2024-09-30").set_index("period")
    assert not cov2.partial.any()


def test_inactivity_label():
    t = _tx([("a", "2025-01-10", "Max", 1), ("b", "2025-03-15", "Max", 1)])
    lab = L.inactivity_label(t, ["a", "b", "c"], pd.Timestamp("2025-01-01"), 30)
    assert lab.to_dict() == {"a": 0, "b": 1, "c": 1}


def test_contribution_scenario_equal_margins_keep_revenue_ranking(raw):
    _, tx, viva = raw
    rank = L.banner_ranking(tx, viva)
    sc = L.contribution_scenario(rank, {b: 30.0 for b in L.ALL_BANNERS})
    assert (sc.rank_change_vs_revenue == 0).all()
    assert set(sc.banner) == set(L.ALL_BANNERS)


def test_epsilon_greedy_is_reproducible_and_bounded():
    a = L.epsilon_greedy_sim([0.05, 0.10, 0.20], 0.1, steps=300, runs=50, seed=1)
    b = L.epsilon_greedy_sim([0.05, 0.10, 0.20], 0.1, steps=300, runs=50, seed=1)
    assert np.allclose(a["avg_reward"], b["avg_reward"])
    assert ((a["pct_optimal"] >= 0) & (a["pct_optimal"] <= 1)).all()
    assert np.all(np.diff(a["cum_regret"]) >= -1e-12)          # regret never decreases


# ------------------------------------------------------------------ validation helpers
def test_bootstrap_indices_preserve_class_counts():
    y = np.array([0] * 70 + [1] * 30)
    idx = L.bootstrap_indices(y, 200, seed=3)
    assert idx.shape == (200, 100)
    assert (y[idx].sum(axis=1) == 30).all()


def test_bootstrap_ci_brackets_point_and_paired_difference():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    good, noise = y + rng.normal(0, 0.6, 400), rng.normal(0, 1, 400)
    idx = L.bootstrap_indices(y, 300, seed=0)
    tab, smp = L.bootstrap_ci(y, {"good": good, "noise": noise}, roc_auc_score, idx)
    assert ((tab.ci_low <= tab.point) & (tab.point <= tab.ci_high)).all()
    d = L.paired_difference(smp["good"], smp["noise"], tab.point[0], tab.point[1])
    assert d["ci_low"] > 0 and d["verdict"] == "A better"


def test_calibration_of_true_probabilities_is_good():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)
    assert L.expected_calibration_error(y, p) < 0.02
    assert L.expected_calibration_error(y, np.clip(p * 0.5, 0, 1)) > 0.15     # systematically under-predicted


def test_clip_to_reference():
    ref = pd.DataFrame({"x": [0, 10], "z": [1, 2]})
    sc = pd.DataFrame({"x": [-5, 5, 20], "z": [1.5, 1.5, 1.5]})
    out, rep = L.clip_to_reference(sc, ref, ["x", "z"])
    assert out.x.tolist() == [0, 5, 10]
    assert np.isclose(rep.loc["x", "share_clipped"], 2 / 3) and rep.loc["z", "share_clipped"] == 0


# ------------------------------------------------------------------ association rules
def _confounded_basket(n=6000, seed=2):
    """A and C each depend on a stratum but are independent within it → raw lift > 1, adjusted ≈ 1."""
    rng = np.random.default_rng(seed)
    s = rng.integers(0, 2, n)
    pa, pc = np.where(s == 1, 0.8, 0.2), np.where(s == 1, 0.7, 0.1)
    b = pd.DataFrame({"A": rng.uniform(size=n) < pa, "C": rng.uniform(size=n) < pc})
    return b, pd.Series(s, index=b.index).astype(str)


def test_stratified_lift_removes_confounding():
    b, s = _confounded_basket()
    raw_lift = (b.A & b.C).mean() / (b.A.mean() * b.C.mean())
    adj = L.stratified_lift(b, ["A"], ["C"], s)["adjusted_lift"]
    assert raw_lift > 1.3 and abs(adj - 1) < 0.06
    assert L.cmh_test(b, ["A"], ["C"], s) > 0.01


def test_cmh_detects_real_association():
    rng = np.random.default_rng(4)
    n = 4000
    s = pd.Series(rng.integers(0, 3, n)).astype(str)
    a = rng.uniform(size=n) < 0.4
    c = np.where(a, rng.uniform(size=n) < 0.6, rng.uniform(size=n) < 0.3)
    b = pd.DataFrame({"A": a, "C": c})
    assert L.stratified_lift(b, ["A"], ["C"], s)["adjusted_lift"] > 1.3
    assert L.cmh_test(b, ["A"], ["C"], s) < 1e-6


# ------------------------------------------------------------------ CLV models
def test_bgnbd_recovers_simulated_parameters():
    true = {"r": 0.6, "alpha": 4.0, "a": 0.9, "b": 3.0}
    T = np.random.default_rng(5).uniform(30, 52, 5000)
    x, t_x, T, fut = L.bgnbd_simulate(true, 5000, T, seed=5, horizon=26)
    fit = L.bgnbd_fit(x, t_x, T)
    assert fit["converged"]
    for k in ("r", "alpha"):
        assert abs(fit[k] / true[k] - 1) < 0.25, (k, fit[k])
    assert abs(fit["mean_dropout_p"] / (true["a"] / (true["a"] + true["b"])) - 1) < 0.25
    pred = L.bgnbd_expected(fit, 26, x, t_x, T)
    assert abs(pred.sum() / fut.sum() - 1) < 0.10                # aggregate hold-out purchases within 10%
    p_alive = L.bgnbd_p_alive(fit, x, t_x, T)
    assert ((p_alive > 0) & (p_alive <= 1)).all()


def test_gamma_gamma_recovers_population_mean_spend():
    rng = np.random.default_rng(6)
    p, q, v = 6.0, 4.0, 15.0
    n = 4000
    nu = rng.gamma(q, 1 / v, n)                                  # customer-level rate
    x = rng.integers(1, 12, n)
    m = np.array([rng.gamma(p, 1 / nu_i, k).mean() for nu_i, k in zip(nu, x)])
    fit = L.gamma_gamma_fit(x, m)
    assert fit["converged"]
    assert abs((fit["p"] * fit["v"] / (fit["q"] - 1)) / (p * v / (q - 1)) - 1) < 0.10
    e = L.gamma_gamma_expected(fit, x, m)
    assert np.corrcoef(e, m)[0, 1] > 0.9


def test_rfm_summary_on_supplied_data(raw):
    _, tx, _ = raw
    cut = tx.date.max() - pd.Timedelta(days=180)
    r = L.rfm_summary(tx, cut)
    assert (r.x >= 0).all() and (r.t_x <= r["T"] + 1e-9).all()
    assert (r.loc[r.x == 0, "m"] == 0).all() and (r.loc[r.x > 0, "m"] > 0).all()


def test_hurdle_regressor_is_non_negative_and_learns():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(800, 3))
    buy = rng.uniform(size=800) < 1 / (1 + np.exp(-2 * X[:, 0]))
    y = np.where(buy, np.exp(5 + 0.5 * X[:, 1] + rng.normal(0, 0.3, 800)), 0.0)
    h = L.HurdleRegressor(GradientBoostingClassifier(random_state=0), GradientBoostingRegressor(random_state=0)).fit(X, y)
    pred = h.predict(X)
    assert (pred >= 0).all()
    assert np.corrcoef(pred, y)[0, 1] > 0.6


def test_feature_label():
    assert L.feature_label("nh_cticket_Fashion") == "Avg ticket · Fashion (outside home)"
    assert L.feature_label("nf_bshare_Home Centre") == "Spend share · Home Centre (outside fashion)"
    assert L.feature_label("recency_days") == "Days since last purchase"


def test_save_json_writes_strict_json(tmp_path):
    import json
    f = tmp_path / "x.json"
    L.save_json({"a": float("nan"), "b": [np.float32(1.5), np.inf], "c": np.array([1.0, np.nan]), "d": np.int64(3)}, f)
    txt = f.read_text()
    assert "NaN" not in txt and "Infinity" not in txt
    assert json.loads(txt) == {"a": None, "b": [1.5, None], "c": [1.0, None], "d": 3}


def test_sort_rules_is_independent_of_input_order():
    d = pd.DataFrame({"antecedent": ["Funcity", "Babyshop", "Splash"], "consequent": ["Babyshop", "Funcity", "Max"],
                      "lift": [1.82, 1.82 + 1e-13, 1.5], "support": [0.3, 0.3, 0.1]})
    a = L.sort_rules(d, ["lift", "support"], [False, False])
    b = L.sort_rules(d.iloc[::-1], ["lift", "support"], [False, False])
    assert a.antecedent.tolist() == b.antecedent.tolist() == ["Babyshop", "Funcity", "Splash"]
    assert list(a.columns) == list(d.columns)
