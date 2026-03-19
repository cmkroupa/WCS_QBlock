"""
helpers.py — shared utility functions for QBlock backend.
"""

import re
from pathlib import Path


# ── Scoring ───────────────────────────────────────────────────────────────────

THRESHOLD  = 0.50
SUSPICIOUS = 0.25


def label(score: float) -> str:
    if score >= THRESHOLD:
        return "phishing"
    if score >= SUSPICIOUS:
        return "suspicious"
    return "safe"


# ── HTML fetching (Playwright) ────────────────────────────────────────────────

def fetch_html(url: str) -> tuple[str, str]:
    """Return (html, final_url) — final_url is the URL after all redirects."""
    from playwright.sync_api import sync_playwright, Error as PWError, TimeoutError as PWTimeout

    html      = ""
    final_url = url
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            java_script_enabled=True,
        )
        page = ctx.new_page()
        page.route("**/*.{png,jpg,jpeg,gif,webp,svg,ico,woff,woff2,ttf}",
                   lambda r: r.abort())
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=20_000)
            page.wait_for_timeout(2_000)
            final_url = page.url          # capture URL after all redirects
        except (PWTimeout, PWError) as e:
            print(f"[fetch] Playwright navigation error for {url}: {e}")
        try:
            html = page.content()
        finally:
            browser.close()
    return html[:500_000], final_url


# ── Training file utilities ───────────────────────────────────────────────────

def list_files(base_dir):
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Training directory not found: {base_dir}")
    paths, labels = [], []
    for label_dir in sorted(base.iterdir()):
        if not label_dir.is_dir():
            continue
        lbl = 1 if label_dir.name.lower().startswith("phish") else 0
        for f in sorted(
            list(label_dir.rglob("*.html")) + list(label_dir.rglob("*.htm"))
        ):
            paths.append(str(f))
            labels.append(lbl)
    return paths, labels


def read_file(path):
    try:
        with open(path, "r", encoding="utf-8", errors="strict") as f:
            return f.read()
    except Exception:
        with open(path, "r", encoding="latin-1", errors="ignore") as f:
            return f.read()
