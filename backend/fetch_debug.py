#!/usr/bin/env python3
"""
fetch_debug.py — shows exactly what HTML the scanner receives for a URL.

Usage:
    python3 fetch_debug.py <url> [options]

Options:
    --save          Save the HTML to a file (fetched_<host>.html)
    --text          Also print the visible text extracted by BeautifulSoup
    --features      Also print the 19 HTML features the model sees
    --head N        Only print first N lines of HTML (default: 50, 0 = all)

Examples:
    python3 fetch_debug.py https://claim-trump.live
    python3 fetch_debug.py https://adobe.com --features --head 0 --save
"""

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

# ── Match app.py fetch mode ───────────────────────────────────────────────────
# Set to True to use Playwright (headless Chromium, executes JS).
# Set to False for plain requests.get().
PLAYWRIGHT = False

MAX_BYTES = 500_000


def fetch(url: str) -> tuple[str, str, int]:
    """Returns (final_url, html, status_code). Raises on network error."""
    if PLAYWRIGHT:
        return _fetch_playwright(url)
    return _fetch_requests(url)


def _fetch_requests(url: str) -> tuple[str, str, int]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    resp = requests.get(url, timeout=15, headers=headers, allow_redirects=True)
    return resp.url, resp.text[:MAX_BYTES], resp.status_code


def _fetch_playwright(url: str) -> tuple[str, str, int]:
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            locale="en-US",
        )
        page = ctx.new_page()
        page.route("**/*.{png,jpg,jpeg,gif,webp,svg,ico,woff,woff2,ttf}", lambda r: r.abort())
        status = 200
        def capture_status(resp):
            nonlocal status
            if resp.url == page.url:
                status = resp.status
        page.on("response", capture_status)
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=20_000)
            page.wait_for_timeout(2_000)
        except PWTimeout:
            print("  [Playwright] Page timed out — using partial DOM")
        html = page.content()
        final_url = page.url
        browser.close()
    return final_url, html[:MAX_BYTES], status


def visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return " ".join(soup.get_text(separator=" ").split())


def print_features(html: str, url: str):
    """Import model.py and run HTMLFeatureExtractor on the fetched HTML."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from model import HTMLFeatureExtractor
        extractor = HTMLFeatureExtractor()
        df, _ = extractor.transform([html])
        print("\n── HTML Features (" + str(len(df.columns)) + " total) " + "─" * 40)
        for name, val in df.iloc[0].items():
            print(f"  {name:<35} {val:.4f}")
    except Exception as exc:
        print(f"\n[features] Could not extract — {exc}")
        print("  (Make sure you run this from the backend/ directory with the venv active)")


def main():
    parser = argparse.ArgumentParser(description="Debug HTML fetcher — mirrors app.py exactly")
    parser.add_argument("url", help="URL to fetch")
    parser.add_argument("--save", action="store_true", help="Save HTML to file")
    parser.add_argument("--text", action="store_true", help="Print visible text")
    parser.add_argument("--features", action="store_true", help="Print model feature values")
    parser.add_argument("--head", type=int, default=50, metavar="N",
                        help="Lines of HTML to print (0 = all, default 50)")
    parser.add_argument("--playwright", action="store_true",
                        help="Use Playwright (headless Chromium) instead of requests")
    args = parser.parse_args()

    global PLAYWRIGHT
    if args.playwright:
        PLAYWRIGHT = True

    url = args.url
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    mode = "Playwright (headless Chromium)" if PLAYWRIGHT else "requests.get"
    print(f"\n── Fetching: {url}  [{mode}]")
    try:
        final_url, html, status = fetch(url)
    except Exception as exc:
        print(f"[ERROR] Fetch failed: {exc}")
        sys.exit(1)

    host = urlparse(final_url).netloc.replace(":", "_")
    redirected = final_url != url

    print(f"   Status:      {status}")
    print(f"   Final URL:   {final_url}" + (" ← REDIRECTED" if redirected else ""))
    print(f"   HTML size:   {len(html):,} chars")

    # ── Redirect warning ─────────────────────────────────────────────────────
    if redirected:
        orig_host = urlparse(url).netloc
        final_host = urlparse(final_url).netloc
        if orig_host != final_host:
            print(f"\n  ⚠  Domain changed on redirect: {orig_host} → {final_host}")
            print("     The model scans the FINAL page's HTML, not the original URL's.")

    # ── Cloudflare / bot challenge detection ─────────────────────────────────
    cf_signals = [
        "cf-browser-verification",
        "Checking if the site connection is secure",
        "cf_chl_opt",
        "jschl_vc",
        "Please Wait... | Cloudflare",
        "DDoS protection by Cloudflare",
        "cf-mitigated",
    ]
    hit = [s for s in cf_signals if s in html]
    if hit:
        print(f"\n  ⚠  CLOUDFLARE / BOT CHALLENGE DETECTED")
        print(f"     Signals found: {hit}")
        print("     The model is seeing a challenge page, NOT the actual site content.")
        print("     This is why it scores as SAFE — there's no phishing content in a JS spinner.")

    # ── Parked / taken-down detection ────────────────────────────────────────
    # Use specific phrases only — avoid single words like "parking" that appear
    # legitimately in parking apps, URLs, etc.
    parked_signals = ["This domain is for sale", "domain has expired",
                      "This domain is parked", "GoDaddy", "Namecheap parking"]
    parked_hit = [s for s in parked_signals if s.lower() in html.lower()]
    if parked_hit:
        print(f"\n  ⚠  POSSIBLY PARKED / TAKEN DOWN")
        print(f"     Signals: {parked_hit}")

    # ── HTML preview ─────────────────────────────────────────────────────────
    lines = html.splitlines()
    n = args.head
    print(f"\n── HTML Preview ({n if n else 'all'} / {len(lines)} lines) " + "─" * 30)
    shown = lines if n == 0 else lines[:n]
    print("\n".join(shown))
    if n and len(lines) > n:
        print(f"\n  ... ({len(lines) - n} more lines hidden, use --head 0 to show all)")

    # ── Visible text ─────────────────────────────────────────────────────────
    if args.text:
        txt = visible_text(html)
        print(f"\n── Visible Text ({len(txt)} chars) " + "─" * 40)
        print(txt[:2000] + (" ..." if len(txt) > 2000 else ""))

    # ── Features ─────────────────────────────────────────────────────────────
    if args.features:
        print_features(html, final_url)

    # ── Save ─────────────────────────────────────────────────────────────────
    if args.save:
        fname = f"fetched_{re.sub(r'[^a-zA-Z0-9._-]', '_', host)}.html"
        with open(fname, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\n── Saved to {fname}")


if __name__ == "__main__":
    main()
