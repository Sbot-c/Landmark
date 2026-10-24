"""
End-to-end dashboard test: launches `streamlit run app.py`, opens every module (A–Q) in a headless Chromium,
fails if Streamlit renders an exception, exercises a few interactive controls, and saves screenshots.

    python tests/test_dashboard.py [screenshot_dir]
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import urllib.request
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
PORT = 8599
PAGES = ["", "financials", "banners", "home-centre-vs-homebox", "splash-vs-max", "other-banners", "segments", "churn", "value",
         "shukran", "acquisition", "bundles", "model-lab", "methodology", "recommendation", "member-360", "what-if"]


def wait_idle(page, timeout=90):
    """Wait until Streamlit has finished running the script (no running indicator) and content is present."""
    t0 = time.time()
    page.wait_for_selector('[data-testid="stMain"]', timeout=timeout * 1000)
    try:
        page.wait_for_selector('[data-testid="stMarkdownContainer"]', timeout=20000)
    except Exception:
        pass
    time.sleep(1.0)
    while time.time() - t0 < timeout:
        running = page.locator('[data-testid="stStatusWidget"]').count()
        if running == 0:
            time.sleep(1.2)
            if page.locator('[data-testid="stStatusWidget"]').count() == 0:
                break
        time.sleep(0.5)
    # charts render after the script finishes: wait until no placeholder skeletons remain and the chart count is stable
    last = -1
    while time.time() - t0 < timeout:
        n = page.locator(".js-plotly-plot").count()
        if page.locator('[data-testid="stSkeleton"]').count() == 0 and n == last:
            return
        last = n
        time.sleep(1.0)


def main():
    app_python = os.environ.get("LCI_APP_PYTHON", sys.executable)        # lets the compatibility run serve the app from another environment
    proc = subprocess.Popen([app_python, "-m", "streamlit", "run", "app.py", "--server.port", str(PORT), "--server.headless", "true"],
                            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://localhost:{PORT}/_stcore/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        results = []
        with sync_playwright() as p:
            browser = p.chromium.launch()
            ctx = browser.new_context(viewport={"width": 1440, "height": 4200}, device_scale_factor=1)   # tall viewport: Streamlit scrolls inside its own container
            page = ctx.new_page()
            console_errors = []
            page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
            for slug in PAGES:
                t0 = time.time()
                page.goto(f"http://localhost:{PORT}/{slug}", wait_until="domcontentloaded")
                wait_idle(page)
                exc = page.locator('[data-testid="stException"]')
                n_exc = exc.count()
                msgs = [exc.nth(i).inner_text()[:400] for i in range(n_exc)]
                n_charts = page.locator(".js-plotly-plot").count()
                n_tables = page.locator('[data-testid="stDataFrame"], [data-testid="stTable"]').count()
                page.screenshot(path=str(OUT / f"{slug or 'overview'}.png"), full_page=True)
                results.append({"page": slug, "exceptions": n_exc, "charts": n_charts, "tables": n_tables, "seconds": round(time.time() - t0, 1), "messages": msgs})
                print(f"{slug:28s} exceptions={n_exc} charts={n_charts} tables={n_tables} ({time.time()-t0:.1f}s)", flush=True)
                for m in msgs:
                    print("   !!", m.replace("\n", " | "))
            # --- interactions ---------------------------------------------------------------------------------
            page.goto(f"http://localhost:{PORT}/churn", wait_until="domcontentloaded"); wait_idle(page)
            before = page.locator('[data-testid="stMarkdownContainer"]:has-text("Cost-optimal threshold")').first.inner_text()
            inp = page.get_by_label("Relative cost of a missed churner")
            inp.fill("15"); inp.press("Enter"); wait_idle(page)
            after = page.locator('[data-testid="stMarkdownContainer"]:has-text("Cost-optimal threshold")').first.inner_text()
            print("interaction · churn cost ratio changed card:", before.replace("\n", " ") != after.replace("\n", " "), "|", after.replace("\n", " ")[:120])
            results.append({"page": "interaction:churn-cost", "exceptions": page.locator('[data-testid="stException"]').count()})
            page.screenshot(path=str(OUT / "churn_after_cost_change.png"), full_page=False)
            page.goto(f"http://localhost:{PORT}/model-lab", wait_until="domcontentloaded"); wait_idle(page)
            page.get_by_text("Home Centre vs Homebox", exact=True).first.click(); wait_idle(page)
            results.append({"page": "interaction:lab-home", "exceptions": page.locator('[data-testid="stException"]').count()})
            page.screenshot(path=str(OUT / "model_lab_home.png"), full_page=False)
            page.goto(f"http://localhost:{PORT}/financials", wait_until="domcontentloaded"); wait_idle(page)
            page.get_by_text("Banner", exact=True).first.click(); wait_idle(page)
            results.append({"page": "interaction:financials-by-banner", "exceptions": page.locator('[data-testid="stException"]').count(),
                            "charts": page.locator(".js-plotly-plot").count()})
            page.screenshot(path=str(OUT / "financials_by_banner.png"), full_page=False)
            page.goto(f"http://localhost:{PORT}/bundles", wait_until="domcontentloaded"); wait_idle(page)
            results.append({"page": "interaction:bundles", "exceptions": page.locator('[data-testid="stException"]').count()})
            # second segmentation view
            page.goto(f"http://localhost:{PORT}/segments", wait_until="domcontentloaded"); wait_idle(page)
            page.get_by_text("Shopping-style segments (second view)", exact=True).first.click(); wait_idle(page)
            results.append({"page": "interaction:segments-style-view", "exceptions": page.locator('[data-testid="stException"]').count(),
                            "charts": page.locator(".js-plotly-plot").count()})
            page.screenshot(path=str(OUT / "segments_style_view.png"), full_page=True)
            # robustness tabs
            for slug, tab, shot in [("churn", "Calibration", "churn_calibration_tab.png"), ("churn", "Data-window feature", "churn_window_tab.png"),
                                    ("model-lab", "Test results with 95% CI", "model_lab_ci_tab.png"), ("methodology", "Validation & robustness", "methodology_validation_tab.png")]:
                page.goto(f"http://localhost:{PORT}/{slug}", wait_until="domcontentloaded"); wait_idle(page)
                page.get_by_role("tab", name=tab).first.click(); wait_idle(page)
                results.append({"page": f"interaction:{slug}-{tab}", "exceptions": page.locator('[data-testid="stException"]').count()})
                page.screenshot(path=str(OUT / shot), full_page=True)
            print("interaction · style view, robustness tabs:", [r["exceptions"] for r in results[-5:]])
            # mobile viewport check of the opening screen
            mctx = browser.new_context(viewport={"width": 390, "height": 844}, device_scale_factor=2)
            mp = mctx.new_page(); mp.goto(f"http://localhost:{PORT}/", wait_until="domcontentloaded"); wait_idle(mp)
            mp.screenshot(path=str(OUT / "overview_mobile.png"), full_page=False)
            results.append({"page": "mobile:overview", "exceptions": mp.locator('[data-testid="stException"]').count()})
            browser.close()
        bad = [r for r in results if r.get("exceptions")]
        print("\nCONSOLE ERRORS:", len(console_errors))
        for e in console_errors[:10]:
            print("   console:", e[:200])
        print("RESULT:", "PASS" if not bad else f"FAIL on {[r['page'] for r in bad]}")
        return 0 if not bad else 1
    finally:
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=10)
        except Exception:
            proc.kill(); out = ""
        (OUT / "server_log.txt").write_text(out or "")
        warn = [l for l in (out or "").splitlines() if "Warning" in l or "deprecat" in l.lower() or "Error" in l]
        if warn:
            print("\nSERVER LOG WARNINGS/ERRORS:")
            for l in warn[:25]:
                print("   ", l[:220])


if __name__ == "__main__":
    sys.exit(main())
