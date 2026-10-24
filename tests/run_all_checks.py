"""
One command to verify the project end to end:

    python tests/run_all_checks.py            # unit tests + quick notebook run (LCI_FAST=1) + dashboard test
    python tests/run_all_checks.py --full     # same, but the notebook runs in full mode (~45 min on 2 cores)
    python tests/run_all_checks.py --skip-notebook --skip-dashboard

The notebook is executed in a temporary copy of the project, so the delivered artifacts are never overwritten.
The dashboard test runs against the delivered artifacts in this folder.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd, cwd, env=None, timeout=None):
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + p.stderr), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="run the notebook in full mode (slow)")
    ap.add_argument("--skip-notebook", action="store_true")
    ap.add_argument("--skip-dashboard", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep the temporary notebook copy for inspection")
    a = ap.parse_args()
    results = []

    # 1 · unit tests
    rc, out, dt = run([sys.executable, "-m", "pytest", str(ROOT / "tests" / "test_core.py"), "-q"], ROOT)
    results.append(("Unit tests (tests/test_core.py)", rc == 0, out.strip().splitlines()[-1] if out.strip() else "", dt))

    # 2 · notebook in a temporary copy
    if not a.skip_notebook:
        tmp = Path(tempfile.mkdtemp(prefix="lci_check_"))
        dst = tmp / ROOT.name
        shutil.copytree(ROOT, dst, ignore=shutil.ignore_patterns("artifacts", "screenshots", "__pycache__", ".ipynb_checkpoints", "*.html"))
        (dst / "artifacts").mkdir(exist_ok=True)
        env = dict(os.environ)
        env.pop("LCI_FAST", None)
        if not a.full:
            env["LCI_FAST"] = "1"
        nb_dir = dst / "notebooks"
        rc_b, out_b, _ = run([sys.executable, "src/build_notebook.py"], nb_dir, env)
        rc, out, dt = run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "Landmark_Customer_Intelligence.ipynb",
                           "--inplace", "--ExecutePreprocessor.timeout=-1"], nb_dir, env)
        ok = rc == 0 and rc_b == 0
        (ROOT / "tests" / "last_notebook_run.log").write_text(out)
        if not ok:                                   # show the failing cell's traceback (nbconvert does not save a failed notebook)
            tail = [l for l in out.splitlines() if l.strip()][-25:]
            print("\n".join(["notebook execution failed — last lines of the log (full log: tests/last_notebook_run.log):"] + tail))
        rc2, out2, _ = run([sys.executable, str(ROOT / "tests" / "check_notebook.py"), str(nb_dir / "Landmark_Customer_Intelligence.ipynb")], ROOT)
        clean = ok and "ERROR" not in out2 and "stderr" not in out2 and "'nan'" not in out2
        results.append((f"Notebook execution ({'full' if a.full else 'quick'} mode, temporary copy)", ok, out2.strip().splitlines()[0] if ok and out2.strip() else "failed — see tests/last_notebook_run.log", dt))
        results.append(("Notebook output scan (errors, stderr, 'nan' in text)", clean, "clean" if clean else out2.strip()[-400:], 0))
        if a.keep:
            print("temporary copy kept at", dst)
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    # 3 · dashboard end-to-end test on the delivered artifacts
    if not a.skip_dashboard:
        shots = Path(tempfile.mkdtemp(prefix="lci_shots_"))
        rc, out, dt = run([sys.executable, str(ROOT / "tests" / "test_dashboard.py"), str(shots)], ROOT, timeout=1800)
        res = [l for l in out.splitlines() if l.startswith("RESULT")]
        results.append(("Dashboard end-to-end test (all modules, interactions, mobile)", rc == 0, res[-1] if res else out[-300:], dt))
        print("dashboard screenshots:", shots)

    print("\n" + "=" * 100)
    for name, ok, detail, dt in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:70s} {dt:7.0f}s  {detail}")
    print("=" * 100)
    all_ok = all(ok for _, ok, _, _ in results)
    print("OVERALL:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
