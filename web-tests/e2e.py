"""End-to-end check of the deployed (or local) demo in headless Chromium.

    uv run --with playwright --python 3.12 python web-tests/e2e.py https://<deployment-url>

Checks: the disclaimer and privacy notice are visible; every bundled sample gets
a prediction whose probability matches the Python reference; lossless and JPEG
uploads work and match; the Grad-CAM canvas has non-transparent pixels; both
models load; and the page makes no request to any other origin.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
REF = json.loads((HERE / "reference.json").read_text())
TOL_LOSSLESS = 1e-3  # probability, same pixels as Python
TOL_JPEG = 0.03  # probability, browser vs PIL JPEG decoding may differ slightly


def wait_until(page, expr: str, timeout_s: float = 120) -> None:
    """Poll a JS expression with page.evaluate (wait_for_function relies on eval,
    which the demo's Content-Security-Policy blocks)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if page.evaluate(f"() => Boolean({expr})"):
            return
        page.wait_for_timeout(100)
    raise TimeoutError(expr)


def main(url: str) -> int:
    origin = urlparse(url).netloc
    failures: list[str] = []
    foreign: list[str] = []
    worst = {"lossless": 0.0, "jpeg": 0.0}
    out_dir = HERE / "screenshots"
    out_dir.mkdir(exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 1100})
        page.on("request", lambda r: foreign.append(r.url) if urlparse(r.url).netloc not in (origin, "") and not r.url.startswith(("blob:", "data:")) else None)
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        resp = page.goto(url, wait_until="networkidle")
        print("status", resp.status)
        if resp.status != 200:
            failures.append(f"HTTP {resp.status}")
        wait_until(page, "window.__demo && window.__demo.ready")
        backend = page.evaluate("window.__demo.state.backend")
        print("backend", backend, "crossOriginIsolated", page.evaluate("self.crossOriginIsolated"))

        disclaimer = page.locator("#disclaimer")
        if not disclaimer.is_visible() or "Not a medical device" not in disclaimer.inner_text():
            failures.append("disclaimer missing")
        if "never sent anywhere" not in page.locator("#privacy").inner_text():
            failures.append("privacy notice missing")

        def run_and_read(action) -> dict:
            runs = page.evaluate("window.__demo.runs")
            action()
            wait_until(page, f"window.__demo.runs > {runs}", 60)
            r = page.locator("#result")
            cam_alpha = page.evaluate(
                """() => { const c = document.getElementById('cam-canvas');
                const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
                let n = 0; for (let i = 3; i < d.length; i += 4) if (d[i] > 0) n++; return n; }"""
            )
            return {
                "prob": float(r.get_attribute("data-prob")),
                "label": r.get_attribute("data-label"),
                "model": r.get_attribute("data-model"),
                "cam_pixels": cam_alpha,
            }

        for name in REF["models"]:
            page.select_option("#model-select", name)
            wait_until(
                page,
                f"window.__demo.state.models['{name}'] && window.__demo.state.models['{name}'].model"
                " && !document.getElementById('model-select').disabled",
            )
            page.wait_for_timeout(500)
            n_ok = 0
            for i, s in enumerate(REF["samples"]):
                got = run_and_read(lambda i=i: page.locator(f".sample[data-index='{i}']").click())
                exp = REF["models"][name]["samples"][i]
                d = abs(got["prob"] - exp["prob"])
                worst["lossless"] = max(worst["lossless"], d)
                if got["model"] != name or d > TOL_LOSSLESS:
                    failures.append(f"{name} sample {i}: browser {got['prob']:.6f} vs python {exp['prob']:.6f}")
                if got["cam_pixels"] == 0:
                    failures.append(f"{name} sample {i}: empty heatmap")
                n_ok += got["label"] == s["label"]
            print(f"{name}: {len(REF['samples'])} samples predicted, {n_ok} match the true label")
            page.screenshot(path=str(out_dir / f"{name}_sample.png"), full_page=False)

            for j, u in enumerate(REF["uploads"]):
                f = HERE / "fixtures" / u["file"]
                got = run_and_read(lambda f=f: page.set_input_files("#file-input", str(f)))
                exp = REF["models"][name]["uploads"][j]
                d = abs(got["prob"] - exp["prob"])
                kind = "lossless" if u["lossless"] else "jpeg"
                worst[kind] = max(worst[kind], d)
                tol = TOL_LOSSLESS if u["lossless"] else TOL_JPEG
                src = page.locator("#result").get_attribute("data-source")
                if d > tol or src != f"upload:{u['file']}":
                    failures.append(f"{name} upload {u['file']}: browser {got['prob']:.6f} vs python {exp['prob']:.6f}")
                if got["cam_pixels"] == 0:
                    failures.append(f"{name} upload {u['file']}: empty heatmap")
                print(f"{name} upload {u['file']}: P(tumor) browser {got['prob']:.4f} python {exp['prob']:.4f} label {got['label']} (true {u['label']})")
            page.screenshot(path=str(out_dir / f"{name}_upload.png"), full_page=False)

        page.screenshot(path=str(out_dir / "full_page.png"), full_page=True)
        browser.close()

    print("max |browser - python| probability:", worst)
    if foreign:
        failures.append(f"requests to other origins: {sorted(set(foreign))}")
    if errors:
        failures.append(f"page errors: {errors}")
    for f in failures:
        print("FAIL", f)
    print("E2E OK" if not failures else f"{len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "http://localhost:4173"))
