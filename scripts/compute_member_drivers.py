"""
Per-member churn-risk drivers (occlusion against a typical member) -> artifacts/metrics/member_churn_drivers.csv

For every member the saved churn pipeline is re-scored with one *group* of features at a time replaced by the typical (median / most
common) training-member value. effect = score(member) - score(member with that group made typical):
  > 0  the group pushes this member's risk UP relative to a typical member, < 0 it pulls it DOWN.
Groups are used instead of single features because the features inside a group are highly correlated (e.g. recency and purchase gaps)
and would hide each other. Effects are an approximation of attribution, not an exact additive decomposition.

    python scripts/compute_member_drivers.py
Requires the artifacts produced by the notebook (models/churn_final.joblib, metrics/customer_master.csv) and the raw data in data/.
"""
import sys, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lci_core as L

warnings.filterwarnings("ignore")
cust, tx, viva, _ = L.load_raw()
pre = tx.merge(cust[["customer_id", "signup_date"]], on="customer_id", how="left")
tx["pre_signup"] = (pre["date"] < pre["signup_date"]).astype(int).to_numpy()   # same flag as the notebook's analytical copy
OBS = tx["date"].max()
REF = OBS - pd.Timedelta(days=90)
cv = L.build_customer_view(cust, tx, OBS)
cv_ref = L.build_customer_view(cust, tx, REF)
cv["days_member_ref"] = (OBS - cv.signup_date).dt.days
NUM = ["age", "family_size", "has_kids", "days_member_ref", "recency_days", "recency_gap_ratio", "mean_gap_days", "days_since_first",
       "purchase_rate_30d", "log_spend_rate_30d", "log_avg_ticket", "discount_rate", "n_banners", "n_categories", "banner_hhi",
       "f90", "f180", "log_spend90", "log_spend180", "trend90"] + [f"bshare_{b}" for b in L.CORE_BANNERS]
CAT = ["gender", "emirate"]
train = cv_ref[cv_ref.has_tx == 1]
X, _ = L.clip_to_reference(cv[NUM + CAT], train[NUM + CAT], NUM)
model = joblib.load(ROOT / "artifacts" / "models" / "churn_final.joblib")
base = model.predict_proba(X)[:, 1]

master = pd.read_csv(ROOT / "artifacts" / "metrics" / "customer_master.csv", index_col=0)
chk = float(np.abs(pd.Series(base, index=cv.index).reindex(master.index) - master.churn_risk_next90d).max()) if cv.index.name == "customer_id" else float("nan")
print("max |reproduced − stored risk| =", chk)

GROUPS = {
    "Time since last purchases": ["recency_days", "recency_gap_ratio", "mean_gap_days"],
    "Purchase frequency": ["purchase_rate_30d", "f90", "f180"],
    "Spend level": ["log_spend_rate_30d", "log_spend90", "log_spend180", "log_avg_ticket"],
    "Spend trend": ["trend90"],
    "Membership tenure": ["days_member_ref", "days_since_first"],
    "Discount use": ["discount_rate"],
    "Shopping breadth": ["n_banners", "n_categories", "banner_hhi"],
    "Banner mix": [f"bshare_{b}" for b in L.CORE_BANNERS],
    "Household & demographics": ["age", "family_size", "has_kids", "gender", "emirate"],
}
typical = {c: float(train[c].median()) for c in NUM}
typical.update({c: train[c].mode().iloc[0] for c in CAT})
out = pd.DataFrame({"churn_risk": base}, index=cv.index)
for g, cols in GROUPS.items():
    Xg = X.copy()
    for c in cols:
        Xg[c] = typical[c]
    out[g] = base - model.predict_proba(Xg)[:, 1]
Xt = X.copy()
for c in NUM + CAT:
    Xt[c] = typical[c]
out["typical_member_risk"] = float(model.predict_proba(Xt.iloc[:1])[:, 1][0])
out.index.name = "customer_id"
out.round(5).to_csv(ROOT / "artifacts" / "metrics" / "member_churn_drivers.csv")
print(out.shape, "typical-member risk", round(float(out.typical_member_risk.iloc[0]), 3))
print(out[list(GROUPS)].abs().mean().sort_values(ascending=False).round(3))
