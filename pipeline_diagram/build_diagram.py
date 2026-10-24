"""
Builds the pipeline diagram and the Key Ring (proposed) identity diagram from the notebook's artifacts,
so the picture always shows the pipeline that was actually implemented (K, components, selected models).

    python pipeline_diagram/build_diagram.py      ->  pipeline.dot / .png / .svg  and  key_ring.dot / .png / .svg
Requires the Graphviz `dot` executable.
"""
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
METRICS = HERE.parent / "artifacts" / "metrics"


def load(name):
    p = METRICS / name
    return json.loads(p.read_text()) if p.exists() else {}


cc = load("cluster_choice.json")
ms = load("model_summary.json")
thr = load("threshold_choice.json")
k = cc.get("k", "K")
npc = cc.get("n_pcs", "n")
alg = cc.get("algorithm", "K-means")
other = "K-means (Euclidean)" if alg.startswith("K-medoids") else "K-medoids (Manhattan)"
rep = cc.get("representation", "raw (15 std. features)")
raw_sel = rep.startswith("raw")
rep_short = "raw std. features" if raw_sel else f"{npc} PCs"
seg_edges = ('view -> clu [label="standardised raw\\nfeatures (selected)"]; view -> pca; pca -> clu [style=dashed, label="compared"];'
             if raw_sel else 'view -> pca -> clu [label="PCs (selected)"]; view -> clu [style=dashed, label="raw compared"];')
sel = {t: ms.get(t, {}).get("selected_model", "?") for t in ("churn", "home", "fashion")}
op = thr.get("operational_threshold", "t*")

NAVY, GOLD, TEAL, TERRA, IVORY, BRONZE, SLATE = "#14213D", "#B08D57", "#2A9D8F", "#C0603F", "#F7F5F0", "#A0672E", "#5C6670"

pipeline = f'''digraph Pipeline {{
  graph [rankdir=LR, fontname="Helvetica", fontsize=12, bgcolor="white", nodesep=0.35, ranksep=0.55, pad=0.3,
         label="Landmark Retail Group — Customer Intelligence Engine · implemented pipeline (solid = built, dashed = proposed)", labelloc=t, fontcolor="{NAVY}"];
  node  [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, color="{NAVY}", fillcolor="{IVORY}", penwidth=1.2, margin="0.15,0.08"];
  edge  [color="{SLATE}", penwidth=1.1, arrowsize=0.7, fontname="Helvetica", fontsize=8, fontcolor="{SLATE}"];

  subgraph cluster_data {{ label="Stage 0 · data"; style="rounded"; color="{GOLD}"; fontcolor="{NAVY}";
    raw   [label="customers.csv +\\ntransactions.csv\\n(Shukran IDs)"];
    audit [label="Audit & validation\\nPK/FK · duplicates · IDs\\noutliers in context · MD5"];
    clean [label="Cleaning (no deletion)\\nflags: pre-signup tx,\\npartial periods"];
    view  [label="Unified customer view\\n1 row / member · as-of 2026-10-01\\nRFM · ticket · discount · mix", fillcolor="#EAF2F8"];
  }}
  subgraph cluster_viva {{ label="Viva branch (separate identity system)"; style="rounded,dashed"; color="{BRONZE}"; fontcolor="{BRONZE}";
    vraw [label="viva_transactions.csv\\n(VR… IDs, 35% anonymous)", color="{BRONZE}"];
    vgap [label="Join attempted → 0 matches\\nViva gap quantified", color="{BRONZE}"];
    keyring [label="Key Ring identity bridge\\n(PROPOSED — consent, hashed keys,\\nclerical review)", style="rounded,dashed", color="{BRONZE}", fillcolor="white"];
  }}
  subgraph cluster_seg {{ label="Unsupervised"; style="rounded"; color="{GOLD}"; fontcolor="{NAVY}";
    pca [label="PCA (15 behavioural features\\n→ {npc} PCs ≈ 90% variance)\\n2-D map · candidate representation"];
    clu [label="Clustering K={k}\\n{alg} on {rep_short}\\ncompared with {other}\\n(like-for-like silhouette)"];
    seg [label="Named segments\\n(evidence-based)\\n+ shopping-style view"];
  }}
  subgraph cluster_sup {{ label="Supervised (leakage-controlled)"; style="rounded"; color="{GOLD}"; fontcolor="{NAVY}";
    cut  [label="Temporal & non-target views\\nchurn: as-of − 90 d\\nrouting: non-home / non-fashion"];
    leak [label="Leakage audits\\n(direct + indirect)"];
    lab  [label="Model lab × 3 targets\\n9 singles + bagging, boosting,\\nvoting, stacking · 1-SE rule"];
    thrn [label="Churn threshold by cost\\n(5:1 base → t = {op})\\nvalue-weighted tiers"];
    route[label="Routing: {sel['home']} (home)\\n{sel['fashion']} (fashion)"];
    clv  [label="CLV regression\\nhistorical (circularity audit)\\n+ next-180-day (cutoff −180 d)\\nBG/NBD benchmark · 80% ranges"];
    bv   [label="Bias–variance\\nlearning/validation curves\\ndecomposition"];
    val  [label="Validation layer\\nout-of-time test · ablation\\ncalibration · bootstrap CIs\\nscoring guard", fillcolor="#EAF2F8"];
  }}
  subgraph cluster_act {{ label="Action layer"; style="rounded"; color="{GOLD}"; fontcolor="{NAVY}";
    apr  [label="Apriori on member-level\\nbanner baskets (+ per segment)\\nactivity/household-adjusted lift"];
    bund [label="Bundles: eligible members\\nby segment, risk, value"];
    eps  [label="ε-greedy offer sandbox\\n(SIMULATION)", style="rounded,dashed", fillcolor="white"];
    loy  [label="Shukran analytics · discount\\ndependency · migration"];
    fin  [label="Group financials\\n8-banner ranking · scenarios"];
  }}
  dash [label="Streamlit dashboard\\n(reads artifacts only)", fillcolor="{NAVY}", fontcolor="white"];

  raw -> audit -> clean -> view;
  {seg_edges} clu -> seg;
  view -> cut -> leak -> lab;
  lab -> thrn; lab -> route; view -> clv;
  lab -> bv; clv -> bv; lab -> val; clv -> val; thrn -> val [style=dotted];
  view -> apr; seg -> apr [label="per segment"]; apr -> bund -> eps;
  seg -> bund [style=dotted]; thrn -> bund [style=dotted, label="risk"]; clv -> bund [style=dotted, label="value"];
  seg -> loy; thrn -> loy; route -> loy;
  clean -> fin; vraw -> vgap -> fin;
  vgap -> keyring [style=dashed]; keyring -> view [style=dashed, label="future"];
  fin -> dash; loy -> dash; bund -> dash; eps -> dash; bv -> dash; route -> dash; thrn -> dash; seg -> dash; val -> dash;
}}'''

key_ring = f'''digraph KeyRing {{
  graph [rankdir=LR, fontname="Helvetica", fontsize=12, bgcolor="white", pad=0.3, nodesep=0.3,
         label="Key Ring identity bridge — PROPOSED architecture (not implemented; no links created from the supplied files)", labelloc=t, fontcolor="{NAVY}"];
  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, color="{NAVY}", fillcolor="{IVORY}", margin="0.15,0.08"];
  edge [color="{SLATE}", arrowsize=0.7, fontname="Helvetica", fontsize=8, fontcolor="{SLATE}"];
  shk  [label="Shukran\\n(SHK… IDs)"];
  viv  [label="Viva Rewards\\n(VR… IDs)", color="{BRONZE}"];
  cons [label="1 · Consent capture\\nopt-in to link, withdrawable\\n(UAE PDPL 45/2021)"];
  det  [label="2 · Deterministic match\\nhashed verified phone / email"];
  prob [label="3 · Probabilistic tier\\nFellegi–Sunter scores"];
  rev  [label="Clerical review\\n(between thresholds)", style="rounded,dashed", fillcolor="white"];
  ring [label="4 · Key-ring table\\nring_id · id_type · hash ·\\nrule · confidence · consent_ts", fillcolor="#EAF2F8"];
  aud  [label="5 · Audit & reversal\\nlink / un-link log"];
  down [label="6 · Refresh models\\nCLV · churn · segments · bundles", fillcolor="{NAVY}", fontcolor="white"];
  shk -> cons; viv -> cons; cons -> det -> ring [label="exact"];
  cons -> prob; prob -> ring [label="above upper"]; prob -> rev [label="grey zone"]; rev -> ring [label="approved"];
  ring -> aud; ring -> down;
}}'''

for name, src in (("pipeline", pipeline), ("key_ring", key_ring)):
    dot = HERE / f"{name}.dot"
    dot.write_text(src)
    for fmt in ("png", "svg"):
        args = ["dot", f"-T{fmt}", str(dot), "-o", str(HERE / f"{name}.{fmt}")]
        if fmt == "png":
            args.insert(1, "-Gdpi=150")
        subprocess.run(args, check=True)
    print("rendered", name)
