"""
heuristics.py – URL structural risk scoring.

Used by app.py (inference) and any future rule-based pre-filtering.
No HTML parsing or ML required — operates on the URL string alone.
"""

import re
from urllib.parse import urlparse

# ── Phishing-signal TLDs ──────────────────────────────────────────────────────
_SUSPECT_TLDS = {
    "live", "xyz", "tk", "ml", "cf", "gq", "ga", "buzz", "click",
    "link", "shop", "online", "site", "top", "club", "work", "vip",
    "win", "loan", "bid", "stream", "download", "review", "racing",
    "accountant", "science", "party", "trade", "webcam", "zip", "mov",
}

_IP_RE   = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_PUNY_RE = re.compile(r"xn--", re.I)


def url_risk_score(url: str) -> tuple[float, list[dict]]:
    """
    Compute a 0–1 risk score from URL structure alone — no HTML needed.
    Returns (score, signals) where signals is a list of triggered indicators
    for display in the frontend.

    These features are robust to distribution shift because the URL itself
    doesn't change whether we're scanning a live page or a saved file.
    """
    signals: list[dict] = []
    risk = 0.0

    try:
        parsed = urlparse(url if "://" in url else "http://" + url)
        host   = (parsed.hostname or "").lower().strip(".")
        path   = parsed.path or ""
        port   = parsed.port
    except Exception:
        return 0.0, []

    if not host:
        return 0.0, []

    parts       = host.split(".")
    tld         = parts[-1] if parts else ""
    domain_name = parts[-2] if len(parts) >= 2 else parts[0]

    # ── IP address as host ────────────────────────────────────────────────
    if _IP_RE.match(host):
        risk += 0.55
        signals.append({"label": "IP address as hostname", "impact": 0.55})

    # ── Suspicious TLD ────────────────────────────────────────────────────
    if tld in _SUSPECT_TLDS:
        risk += 0.30
        signals.append({"label": f"High-risk TLD (.{tld})", "impact": 0.30})

    # ── IDN / punycode homograph ──────────────────────────────────────────
    if _PUNY_RE.search(host):
        risk += 0.40
        signals.append({"label": "Punycode / IDN homograph domain", "impact": 0.40})

    # ── @ in URL (hides real destination) ────────────────────────────────
    if "@" in url:
        risk += 0.45
        signals.append({"label": "@ symbol in URL", "impact": 0.45})

    # ── Excessive subdomains (e.g. secure.update.verify.evil.tk) ─────────
    subdomain_count = max(0, len(parts) - 2)
    if subdomain_count >= 4:
        risk += 0.35
        signals.append({"label": f"{subdomain_count} subdomain levels", "impact": 0.35})
    elif subdomain_count == 3:
        risk += 0.15
        signals.append({"label": "3 subdomain levels", "impact": 0.15})

    # ── Many hyphens in domain label ──────────────────────────────────────
    hyphen_count = domain_name.count("-")
    if hyphen_count >= 4:
        risk += 0.25
        signals.append({"label": f"{hyphen_count} hyphens in domain", "impact": 0.25})
    elif hyphen_count == 3:
        risk += 0.10
        signals.append({"label": "3 hyphens in domain", "impact": 0.10})

    # ── Unusual port ──────────────────────────────────────────────────────
    if port and port not in (80, 443, 8080, 8443):
        risk += 0.20
        signals.append({"label": f"Non-standard port ({port})", "impact": 0.20})

    # ── Very long URL ─────────────────────────────────────────────────────
    url_len = len(url)
    if url_len > 200:
        risk += 0.20
        signals.append({"label": f"Very long URL ({url_len} chars)", "impact": 0.20})
    elif url_len > 120:
        risk += 0.08
        signals.append({"label": f"Long URL ({url_len} chars)", "impact": 0.08})

    # ── URL-encoded chars in the host (obfuscation) ───────────────────────
    if "%" in host:
        risk += 0.35
        signals.append({"label": "URL-encoded characters in hostname", "impact": 0.35})

    # ── Many digits in domain label ──────────────────────────────────────
    digit_count = sum(c.isdigit() for c in domain_name)
    if digit_count >= 5:
        risk += 0.12
        signals.append({"label": f"Many digits in domain ({digit_count})", "impact": 0.12})

    # ── Deep path (e.g. /a/b/c/d/e/login) ────────────────────────────────
    path_depth = path.count("/")
    if path_depth >= 6:
        risk += 0.10
        signals.append({"label": f"Deep URL path (/{path_depth} levels)", "impact": 0.10})

    score = round(min(risk, 1.0), 4)
    signals.sort(key=lambda s: s["impact"], reverse=True)
    return score, signals
