"""
model.py – HTMLFeatureExtractor, url_risk_score, transformer loading, embedding.
Shared between train.py (training) and app.py (inference).
"""

import math
import re
from collections import Counter
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from sklearn.base import BaseEstimator, TransformerMixin

TRANSFORMER_NAME = "xlm-roberta-base"
EMB_BATCH_SIZE   = 32   # 32 fills MPS/CUDA pipelines better than 16; safe on CPU too

# ── Phishing-signal TLDs ──────────────────────────────────────────────────────
_SUSPECT_TLDS = {
    "live", "xyz", "tk", "ml", "cf", "gq", "ga", "buzz", "click",
    "link", "shop", "online", "site", "top", "club", "work", "vip",
    "win", "loan", "bid", "stream", "download", "review", "racing",
    "accountant", "science", "party", "trade", "webcam", "zip", "mov",
}

_IP_RE   = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_PUNY_RE = re.compile(r"xn--", re.I)

# ── Brand / action scrubbing ──────────────────────────────────────────────────
# Phishing pages deliberately mimic well-known brand names and use
# credential-action verbs.  Replacing them with neutral placeholders forces
# RoBERTa to learn the *structural* pattern (form layout, link topology, etc.)
# rather than memorising which brand names co-occur with phishing labels.
#
# Brand list is calibrated to the actual top-targeted brands in phishing
# datasets (APWG Q4-2024, PhishTank, IBM X-Force rankings), grouped by
# category for maintainability.  Sub-brands / service aliases are included
# where they appear as standalone words in page text (e.g., "Outlook",
# "Gmail", "iCloud" are each more common in phishing HTML than their parent
# brand name alone).
_BRANDS = {
    # ── Microsoft ecosystem (largest single phishing target, ~35% of volume) ──
    "microsoft", "outlook", "onedrive", "sharepoint", "office365",
    "hotmail", "skype", "azure", "teams",

    # ── Google ecosystem ───────────────────────────────────────────────────────
    "google", "gmail", "googleplay", "googledrive",

    # ── Apple ecosystem ────────────────────────────────────────────────────────
    "apple", "icloud", "itunes", "appstore",

    # ── Social / messaging (combined second-largest target) ────────────────────
    "facebook", "instagram", "whatsapp", "messenger",
    "twitter", "linkedin", "snapchat", "tiktok", "telegram",

    # ── Payment processors ─────────────────────────────────────────────────────
    "paypal", "cashapp", "venmo", "zelle", "stripe",

    # ── Financial institutions (US + UK heavy-hitters in dataset) ─────────────
    "chase", "wellsfargo", "bankofamerica", "capitalone",
    "americanexpress", "amex", "citibank", "barclays", "hsbc",
    "mastercard", "visa",

    # ── eCommerce ─────────────────────────────────────────────────────────────
    "amazon", "ebay", "walmart",

    # ── Delivery / logistics (massive spike post-2020) ─────────────────────────
    "dhl", "fedex", "ups", "usps", "royalmail",

    # ── Cloud / productivity ───────────────────────────────────────────────────
    "dropbox", "adobe", "docusign", "zoom", "webex",

    # ── Streaming ─────────────────────────────────────────────────────────────
    "netflix", "spotify", "disneyplus",

    # ── Crypto (fastest-growing phishing vertical) ─────────────────────────────
    "coinbase", "binance", "metamask", "ledger", "kraken",

    # ── Government / tax (seasonal peaks in dataset) ──────────────────────────
    "irs", "hmrc",
}

# Build a single compiled regex — longest alternatives first (avoids partial
# matches on substrings, e.g. "bankofamerica" before "bank").
_BRAND_RE = re.compile(
    r"\b(" + "|".join(sorted(_BRANDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)

# ── Action / urgency language specific to phishing pages ─────────────────────
# Designed at the PHRASE level, not just single words, to avoid over-scrubbing
# legitimate pages that incidentally contain generic verbs like "confirm" or
# "update".  Covers three distinct phishing UX patterns observed in the dataset:
#
#   A. Credential-capture CTAs  — "verify your account", "sign in to continue"
#   B. Urgency / threat framing — "account suspended", "unusual activity detected"
#   C. Document / BEC lures     — "shared document", "click here to verify"
#
# Each alternation is as specific as possible; broad single-word matches
# ("verify", "confirm") are intentionally avoided to preserve signal from
# legitimate confirmation emails in the benign class.
_ACTION_RE = re.compile(
    # ── A: Credential-capture CTAs ────────────────────────────────────────────
    r"\bsign[\s\-]?in\b"                           # "sign in", "sign-in"
    r"|\blog[\s\-]?in\b"                           # "log in", "log-in"
    r"|\bsign[\s\-]?up\b"                          # "sign up" (account creation lure)
    r"|\bauthenticat\w+"                           # authenticate / authentication / authenticating
    r"|\bcredential\w*"                            # credential / credentials
    r"|\bverif(?:y|ied|ication)\b"                 # verify / verified / verification
    r"(?:\s+(?:your\s+)?(?:account|identity|email|details?|information|now))?"
    r"|\bconfirm(?:ation)?\b"                      # confirm / confirmation
    r"(?:\s+(?:your\s+)?(?:account|identity|email|details?|information|now))?"
    r"|\bvalidat(?:e|ion|ing)\b"                   # validate / validation
    r"(?:\s+(?:your\s+)?(?:account|identity|information))?"

    # ── B: Urgency / account-threat language ──────────────────────────────────
    r"|\baccount\s+(?:suspended|disabled|locked|limited|restricted|compromised|violated)\b"
    r"|\bunusual\s+(?:activity|sign[\s\-]?in|login|access)\b"
    r"|\bsuspicious\s+(?:activity|login|sign[\s\-]?in|access)\b"
    r"|\baction\s+required\b"
    r"|\bunauthori[sz]ed\s+(?:access|activity|login)\b"
    r"|\byour\s+account\s+(?:has been|will be|is)\s+(?:suspended|closed|disabled|locked)\b"

    # ── Password / account recovery ───────────────────────────────────────────
    r"|\bpassword\s+(?:reset|expir\w+|recovery)\b"
    r"|\brecover\s+(?:your\s+)?(?:account|password)\b"
    r"|\bforgot\s+(?:your\s+)?password\b"

    # ── C: Document / file lures (BEC / spear-phishing) ──────────────────────
    r"|\bshared?\s+(?:document|file|folder|link)\b"
    r"|\bclick\s+(?:here\s+)?to\s+(?:verify|confirm|access|view|download|validate)\b",
    re.IGNORECASE,
)

# ── URL neutralisation ────────────────────────────────────────────────────────
# These path keywords indicate credential-harvesting pages.
_URL_SENSITIVE_RE = re.compile(
    r"login|signin|sign-in|secure|verify|confirm|update|account|password|"
    r"credential|auth|banking|payment|pay|wallet|recover|reset|webscr",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_html(html: str) -> BeautifulSoup:
    """Parse HTML with lxml, falling back to html.parser on error."""
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")


def _clean_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip ends."""
    return re.sub(r"\s+", " ", text).strip()

def _neutralize_url(href: str) -> str:
    """
    Map a raw href / src attribute to a generic semantic token.

    Token vocabulary:
        URL_EMPTY           — blank or missing href
        URL_JS              — javascript: pseudo-URL (inline execution)
        URL_DATA            — data: URI (inline resource / obfuscation)
        URL_MAIL            — mailto: link
        URL_ANCHOR          — fragment-only (#…) link — stays on-page
        URL_INT             — relative / same-origin link
        URL_EXT_SENSITIVE   — absolute URL containing credential-harvest keywords
        URL_EXT             — other absolute external URL
    """
    href = (href or "").strip()
    if not href:
        return "URL_EMPTY"
    if href.startswith("javascript"):
        return "URL_JS"
    if href.startswith("data:"):
        return "URL_DATA"
    if href.startswith("mailto:"):
        return "URL_MAIL"
    if href.startswith("#"):
        return "URL_ANCHOR"
    if not href.startswith(("http://", "https://")):
        # Relative path — same origin
        return "URL_INT"
    # Absolute URL — check for sensitive path keywords
    if _URL_SENSITIVE_RE.search(href):
        return "URL_EXT_SENSITIVE"
    return "URL_EXT"


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


# ─────────────────────────────────────────────────────────────────────────────
# HTMLFeatureExtractor
# ─────────────────────────────────────────────────────────────────────────────

class HTMLFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts two complementary representations from raw HTML:

    Numeric features (returned as a DataFrame, reduced further by VIF filtering):
        Tag counts, Shannon entropy, DOM depth, link topology. Used by Voter A.

    Text for embedding (returned via the `extras` dict):
        visible_texts — brand/action-scrubbed page text for the first RoBERTa channel.
        struct_cores  — URL-tokenised structural fingerprint for the second channel.
        Both are drawn from a <script>/<style>-stripped soup copy.
    """

    def fit(self, X, y=None):
        return self

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _scrub_brands(self, text: str) -> str:
        """
        Replace brand names with [BRAND] and credential-action verbs with
        [ACTION].  Forces the model to learn structural phishing signals
        rather than which brand names are being impersonated.
        """
        text = _BRAND_RE.sub("[BRAND]", text)
        text = _ACTION_RE.sub("[ACTION]", text)
        return text

    def _visible_text(self, soup):
        """
        Non-mutating visible-text extraction for the visible_len feature.
        Skips text nodes whose parent is <script>, <style>, or <noscript>.
        """
        parts = []
        for elem in soup.find_all(string=True):
            if elem.parent and elem.parent.name in {"script", "style", "noscript"}:
                continue
            t = str(elem).strip()
            if t:
                parts.append(t)
        return re.sub(r"\s+", " ", " ".join(parts)).strip()

    def _smart_embed_text(self, soup):
        """
        Curated text for the RoBERTa visible-text embedding.

        Operates on the <script>/<style>-stripped soup copy passed in from
        transform(), so the 512-token budget is used entirely for structural
        signal.  Brand names and credential-action verbs are replaced with
        [BRAND] / [ACTION] placeholders to prevent the model from memorising
        brand–phishing co-occurrences instead of learning page structure.

        Budget allocation (≈4 chars / token):
          Segment A  ~128 tok — page identity: title, h1–h3, meta description
          Segment B  ~256 tok — form intent: labels, placeholders, button text,
                                              link text adjacent to forms,
                                              all input names/values
          Segment C  ~128 tok — page footer: footer element, legal, copyright
        """
        _APPROX_CHARS = {"A": 500, "B": 1000, "C": 500}

        # ── Segment A: page identity ──────────────────────────────────────────
        head_parts = []
        title = soup.find("title")
        if title:
            head_parts.append(title.get_text(" ", strip=True))

        desc = soup.find("meta", attrs={"name": re.compile(r"description", re.I)})
        if desc and desc.get("content"):
            head_parts.append(desc["content"])

        for h in soup.find_all(["h1", "h2", "h3"]):
            t = h.get_text(" ", strip=True)
            if t:
                head_parts.append(t)

        seg_a = self._scrub_brands(_clean_whitespace(" ".join(head_parts))[: _APPROX_CHARS["A"]])

        # ── Segment B: form intent (all forms) ───────────────────────────────
        # Capture all forms — phishing pages often place the credential form
        # below decoy content, so restricting to the first form loses signal.
        form_parts = []
        forms = soup.find_all("form")
        if forms:
            for form in forms:
                parent = form.parent
                if parent and parent.name not in ("body", "html", "[document]", None):
                    context_node = parent
                else:
                    context_node = form
                for el in context_node.find_all(
                    ["label", "input", "button", "select", "textarea", "a", "p", "span"]
                ):
                    t = _clean_whitespace(el.get_text(" ", strip=True))
                    if t and len(t) < 200:
                        form_parts.append(t)
                    for attr in ("placeholder", "value", "name", "aria-label"):
                        v = el.get(attr, "")
                        if v and len(v) < 100:
                            form_parts.append(v)
        else:
            for el in soup.find_all(["a", "button"]):
                t = _clean_whitespace(el.get_text(" ", strip=True))
                if t and len(t) < 150:
                    form_parts.append(t)

        seg_b = self._scrub_brands(_clean_whitespace(" | ".join(form_parts))[: _APPROX_CHARS["B"]])

        # ── Segment C: footer / legal ─────────────────────────────────────────
        footer_parts = []
        footer_node = soup.find("footer") or soup.find(
            attrs={"class": re.compile(r"footer|legal|disclaimer|copyright", re.I)}
        )
        if footer_node:
            footer_parts.append(footer_node.get_text(" ", strip=True))
        else:
            for p in soup.find_all("p")[-5:]:
                t = _clean_whitespace(p.get_text(" ", strip=True))
                if t:
                    footer_parts.append(t)

        seg_c = self._scrub_brands(_clean_whitespace(" ".join(footer_parts))[: _APPROX_CHARS["C"]])

        # ── Assemble with section markers ─────────────────────────────────────
        parts = []
        if seg_a:
            parts.append(f"PAGE TITLE {seg_a}")
        if seg_b:
            parts.append(f"FORM CONTENT {seg_b}")
        if seg_c:
            parts.append(f"FOOTER {seg_c}")

        return _clean_whitespace(" ".join(parts))

    def _structural_core(self, soup) -> str:
        """
        Compact, adversarially-robust structural fingerprint for the second
        RoBERTa embedding channel (replaces the old flat tag-sequence dump).

        Captures:
          TITLE       — page title text (brand-scrubbed)
          META_*      — meta description / keywords (brand-scrubbed)
          FORM        — each form's action token, HTTP method, enctype
          FORM_INPUTS — space-separated input name/type tokens per form
          LINKS       — first 10 anchor hrefs (URL-neutralised) + link text
          IMAGES      — first 10 img srcs (URL-neutralised) + alt text

        All URLs are replaced with semantic tokens via _neutralize_url() so
        the model cannot memorise specific phishing domains — it must instead
        learn the *pattern* (e.g., many URL_EXT_SENSITIVE links inside a form).
        """
        parts = []

        # ── Title ─────────────────────────────────────────────────────────────
        title = soup.find("title")
        if title:
            t = self._scrub_brands(_clean_whitespace(title.get_text(" ", strip=True)))
            if t:
                parts.append(f"TITLE {t}")

        # ── Meta description / keywords ───────────────────────────────────────
        for meta in soup.find_all("meta"):
            name    = (meta.get("name") or meta.get("property") or "").lower()
            content = meta.get("content", "")
            if name in ("description", "keywords") and content:
                scrubbed = self._scrub_brands(_clean_whitespace(content[:200]))
                parts.append(f"META_{name.upper()} {scrubbed}")

        # ── Forms: action / method / enctype + input names ────────────────────
        for form in soup.find_all("form"):
            action  = _neutralize_url(form.get("action", ""))
            method  = (form.get("method") or "get").upper()
            enctype = form.get("enctype", "")
            form_str = f"FORM action={action} method={method}"
            if enctype:
                form_str += f" enctype={enctype}"
            parts.append(form_str)

            # Input names give strong structural signal (e.g., "password email username")
            input_names = []
            for inp in form.find_all("input")[:15]:
                token = inp.get("name") or inp.get("type") or ""
                if token:
                    input_names.append(token.lower()[:30])
            if input_names:
                parts.append(f"FORM_INPUTS {' '.join(input_names)}")

        # ── Links — all sensitive ones + first 10 others ─────────────────────
        # Prioritise URL_EXT_SENSITIVE links regardless of position — phishing
        # pages often bury the credential-harvesting link below decoy content.
        link_parts  = []
        sensitive   = []
        other       = []
        for a in soup.find_all("a", href=True):
            href_tok = _neutralize_url(a.get("href", ""))
            text     = self._scrub_brands(_clean_whitespace(a.get_text(" ", strip=True))[:50])
            entry    = f"{href_tok} {text}".strip()
            if not entry:
                continue
            if href_tok == "URL_EXT_SENSITIVE":
                sensitive.append(entry)
            else:
                other.append(entry)
        link_parts = sensitive + other[:max(0, 10 - len(sensitive))]
        if link_parts:
            parts.append(f"LINKS {' | '.join(link_parts)}")

        # ── First 10 images ───────────────────────────────────────────────────
        img_parts = []
        for img in soup.find_all("img")[:10]:
            src_tok = _neutralize_url(img.get("src", ""))
            alt     = self._scrub_brands(_clean_whitespace(img.get("alt", ""))[:30])
            entry   = f"{src_tok} {alt}".strip()
            if entry:
                img_parts.append(entry)
        if img_parts:
            parts.append(f"IMAGES {' | '.join(img_parts)}")

        return _clean_whitespace(" ".join(parts))

    def _shannon_entropy(self, s):
        if not s:
            return 0.0
        probs = [n / len(s) for n in Counter(s).values()]
        return -sum(p * math.log2(p) for p in probs if p > 0)

    def _max_depth(self, soup):
        """Iterative DOM depth — avoids RecursionError on adversarially deep pages."""
        maxd  = 0
        stack = [(soup, 0)]
        while stack:
            node, cur = stack.pop()
            maxd = max(maxd, cur)
            for c in getattr(node, "contents", []):
                if getattr(c, "name", None):
                    stack.append((c, cur + 1))
        return maxd

    def _link_features(self, soup):
        """
        Count external vs internal links using a base_url-independent definition:
          external — <a> with an absolute URL (http:// or https://)
          internal — <a> with a relative path

        Using absolute-vs-relative rather than same-host-vs-different-host keeps
        the definition identical at train time (file paths, no real host) and
        inference time (real URL), avoiding a systematic train/inference skew.
        count_unique_link_domains and ratio_external_links are intentionally
        omitted — both are derived from these two counts and introduce
        multicollinearity.
        """
        external, internal = 0, 0
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith("#") or href.startswith("javascript"):
                continue
            if href.startswith(("http://", "https://")):
                external += 1
            else:
                internal += 1
        return external, internal

    # ── Public API ────────────────────────────────────────────────────────────

    def transform(self, X, y=None):
        """
        Parameters
        ----------
        X : list[str] — raw HTML strings

        Returns
        -------
        df     : pd.DataFrame — numeric features (one row per page)
        extras : dict with "visible_texts" and "struct_cores" lists
        """
        results = list(tqdm(
            joblib.Parallel(n_jobs=-1, prefer="processes", return_as="generator")(
                joblib.delayed(_process_single_html)(html_text) for html_text in X
            ),
            total=len(X), desc="HTML features", leave=False,
        ))

        rows          = [r[0] for r in results]
        visible_texts = [r[1] for r in results]
        struct_cores  = [r[2] for r in results]

        df = pd.DataFrame(rows)
        return df, {"visible_texts": visible_texts, "struct_cores": struct_cores}


# ─────────────────────────────────────────────────────────────────────────────
# Module-level worker for parallel HTML feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _process_single_html(html_text: str) -> tuple:
    """
    Process one HTML document: return (numeric_doc, visible_text, struct_core).
    Top-level so the multiprocessing spawn backend can pickle it.
    """
    ex = HTMLFeatureExtractor()

    soup       = _parse_html(html_text)
    # Use BS4 decompose — more robust than regex stripping on malformed HTML
    soup_embed = _parse_html(html_text)
    for tag in soup_embed(["script", "style", "noscript"]):
        tag.decompose()

    # ── Numeric features ──────────────────────────────────────────────────────
    doc = {}
    doc["raw_len"]         = len(html_text)
    doc["shannon_entropy"] = ex._shannon_entropy(html_text)

    tags = [t.name for t in soup.find_all()]
    tc   = Counter(tags)
    for t in ["a", "img", "script", "iframe", "form", "input",
              "link", "meta", "button", "svg"]:
        doc[f"count_tag__{t}"] = tc.get(t, 0)

    doc["num_unique_tags"] = len(set(tags))
    doc["max_dom_depth"]   = ex._max_depth(soup)
    doc["visible_len"]     = len(ex._visible_text(soup))

    ext, intern = ex._link_features(soup)
    doc["count_external_links"] = ext
    doc["count_internal_links"] = intern

    # ── High-signal phishing features ─────────────────────────────────────────
    # Password / hidden inputs — direct credential-harvesting indicators
    all_inputs = soup.find_all("input")
    doc["count_password_inputs"] = sum(
        1 for i in all_inputs if (i.get("type") or "").lower() == "password"
    )
    doc["count_hidden_inputs"] = sum(
        1 for i in all_inputs if (i.get("type") or "").lower() == "hidden"
    )

    # POST-method forms — phishing pages almost always POST credentials
    doc["count_forms_post"] = sum(
        1 for f in soup.find_all("form")
        if (f.get("method") or "get").lower() == "post"
    )

    # Meta refresh — common in phishing redirect chains
    doc["count_meta_refresh"] = sum(
        1 for m in soup.find_all("meta")
        if (m.get("http-equiv") or "").lower() == "refresh"
    )

    # javascript: hrefs — obfuscation / credential interception
    doc["count_javascript_hrefs"] = sum(
        1 for a in soup.find_all("a", href=True)
        if a["href"].strip().lower().startswith("javascript")
    )

    # data: URIs in img/embed/iframe — inline resource obfuscation
    doc["count_data_uris"] = sum(
        1 for tag in soup.find_all(["img", "iframe", "embed", "source"])
        if (tag.get("src") or "").strip().lower().startswith("data:")
    )

    visible_text = ex._smart_embed_text(soup_embed)
    struct_core  = ex._structural_core(soup_embed)

    return doc, visible_text, struct_core


# ─────────────────────────────────────────────────────────────────────────────
# PhishBERT — fine-tuned RoBERTa classification head (Voter B)
# ─────────────────────────────────────────────────────────────────────────────

def combine_texts(vis_text: str, struct_core: str) -> str:
    """Concatenate visible text + structural fingerprint for classifier input."""
    parts = [p.strip() for p in (vis_text, struct_core) if p and p.strip()]
    return " ".join(parts)


class _PhishBERTDataset:
    """Minimal PyTorch-compatible dataset for tokenised phishing classification."""

    def __init__(self, input_ids, attention_masks, labels=None):
        self.input_ids       = input_ids
        self.attention_masks = attention_masks
        self.labels          = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


class PhishBERTClassifier:
    """
    XLM-RoBERTa + linear classification head, fine-tuned for phishing detection.

    Architecture
    ────────────
      backbone  : xlm-roberta-base (bottom layers frozen)
      pooling   : [CLS] token from last hidden state (position 0)
      head      : Dropout(p) → Linear(hidden_size, 1)
      loss      : BCEWithLogitsLoss with pos_weight for class imbalance

    Training modes (n_unfreeze_layers)
    ────────────────────────────────────
      0  — head only  (frozen backbone, ~seconds per epoch — used for OOF proxy)
      2  — fine-tune top-2 transformer blocks + head  (default)

    Sklearn-compatible: fit(texts, y), predict_proba(texts) → (N, 2).
    Input texts should be combine_texts(vis_text, struct_core).
    """

    def __init__(
        self,
        model_name=TRANSFORMER_NAME,
        n_unfreeze_layers=2,
        dropout=0.2,
        lr=2e-5,
        head_lr=5e-4,
        weight_decay=0.01,
        batch_size=16,
        max_epochs=8,
        patience=2,
        warmup_ratio=0.06,
        max_length=512,
        device=None,
        random_state=42,
    ):
        self.model_name        = model_name
        self.n_unfreeze_layers = n_unfreeze_layers
        self.dropout           = dropout
        self.lr                = lr
        self.head_lr           = head_lr
        self.weight_decay      = weight_decay
        self.batch_size        = batch_size
        self.max_epochs        = max_epochs
        self.patience          = patience
        self.warmup_ratio      = warmup_ratio
        self.max_length        = max_length
        self.device            = device or get_device()
        self.random_state      = random_state

        self._model     = None
        self._tokenizer = None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build(self):
        """Construct tokeniser and model (backbone + head). Returns (tok, model)."""
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, AutoModel

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        backbone  = AutoModel.from_pretrained(self.model_name)

        # Freeze entire backbone first
        for param in backbone.parameters():
            param.requires_grad = False

        # Selectively unfreeze top N transformer encoder layers
        if self.n_unfreeze_layers > 0:
            for layer in backbone.encoder.layer[-self.n_unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        hidden_size = backbone.config.hidden_size  # 768 for base

        class _Model(nn.Module):
            def __init__(self_, backbone, dropout, hidden_size):
                super().__init__()
                self_.backbone   = backbone
                self_.dropout    = nn.Dropout(dropout)
                self_.head       = nn.Linear(hidden_size, 1)
                nn.init.xavier_uniform_(self_.head.weight)
                nn.init.zeros_(self_.head.bias)

            def forward(self_, input_ids, attention_mask):
                out    = self_.backbone(input_ids=input_ids, attention_mask=attention_mask)
                cls    = out.last_hidden_state[:, 0, :]            # (B, H) — [CLS] token
                return self_.head(self_.dropout(cls)).squeeze(-1)  # (B,) logits

        model = _Model(backbone, self.dropout, hidden_size)
        return tokenizer, model

    def _make_loader(self, texts, labels=None, shuffle=False, override_batch=None):
        """Tokenise texts and return a DataLoader."""
        import torch
        from torch.utils.data import DataLoader

        enc = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        lbl = torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        ds  = _PhishBERTDataset(enc["input_ids"], enc["attention_mask"], lbl)
        bs  = override_batch if override_batch is not None else (
            self.batch_size if shuffle else self.batch_size * 2
        )
        return DataLoader(ds, batch_size=bs, shuffle=shuffle)

    def _raw_logits(self, loader):
        """Run the model in eval mode and return concatenated logits."""
        import torch
        self._model.eval()
        all_logits = []
        with torch.no_grad():
            for batch in loader:
                logits = self._model(
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                )
                all_logits.append(logits.cpu())
        return torch.cat(all_logits)

    def _val_auc(self, loader, y_true):
        from sklearn.metrics import roc_auc_score
        import torch
        probs = torch.sigmoid(self._raw_logits(loader)).numpy()
        return roc_auc_score(y_true, probs)

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, texts, y, val_texts=None, val_y=None):
        import torch
        import torch.nn as nn
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup

        torch.manual_seed(self.random_state)

        n_gpu = get_n_gpu()
        print(
            f"[bert] Building PhishBERT "
            f"(unfreeze_layers={self.n_unfreeze_layers}, device={self.device}"
            + (f", {n_gpu} GPUs" if n_gpu > 1 else "") + ")…"
        )
        self._tokenizer, self._model = self._build()
        self._model = self._model.to(self.device)
        if self.device == "cuda" and n_gpu > 1:
            print(f"[bert] Wrapping in DataParallel ({n_gpu} GPUs)")
            self._model = nn.DataParallel(self._model)

        # Scale batch size across GPUs so each GPU sees self.batch_size samples
        effective_batch = self.batch_size * max(1, n_gpu)

        print(f"[bert] Tokenising {len(texts)} training samples…")
        train_loader = self._make_loader(texts, y, shuffle=True,
                                         override_batch=effective_batch)
        val_loader   = (
            self._make_loader(val_texts, val_y) if val_texts is not None else None
        )

        # Class-imbalance weighting
        n_neg      = int((np.array(y) == 0).sum())
        n_pos      = int((np.array(y) == 1).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(self.device)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Separate learning-rate groups: higher LR for the fresh head
        head_params     = list(self._model.head.parameters())
        head_ids        = {id(p) for p in head_params}
        backbone_params = [p for p in self._model.parameters()
                           if p.requires_grad and id(p) not in head_ids]

        optimizer = AdamW(
            [
                {"params": backbone_params, "lr": self.lr},
                {"params": head_params,     "lr": self.head_lr},
            ],
            weight_decay=self.weight_decay,
        )

        total_steps   = len(train_loader) * self.max_epochs
        warmup_steps  = int(total_steps * self.warmup_ratio)
        scheduler     = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        best_val_auc = 0.0
        best_state   = None
        no_improve   = 0

        for epoch in range(self.max_epochs):
            self._model.train()
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                logits = self._model(
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                )
                loss = criterion(logits, batch["labels"].to(self.device))
                loss.backward()
                _params = (self._model.module if hasattr(self._model, "module")
                           else self._model).parameters()
                nn.utils.clip_grad_norm_(_params, 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            if val_loader is not None:
                val_auc = self._val_auc(val_loader, val_y)
                print(
                    f"[bert]   Epoch {epoch + 1}/{self.max_epochs}  "
                    f"loss={avg_loss:.4f}  val_AUC={val_auc:.4f}"
                )
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    _raw = self._model.module if hasattr(self._model, "module") else self._model
                    best_state   = {k: v.cpu().clone() for k, v in _raw.state_dict().items()}
                    no_improve   = 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        print(f"[bert]   Early stopping at epoch {epoch + 1}")
                        break
            else:
                print(f"[bert]   Epoch {epoch + 1}/{self.max_epochs}  loss={avg_loss:.4f}")

        if best_state is not None:
            self._model.load_state_dict(best_state)
            print(f"[bert] Restored best checkpoint (val_AUC={best_val_auc:.4f})")

        return self

    def predict_proba(self, texts):
        """Return (N, 2) array [P_benign, P_phish] — sklearn convention."""
        import torch
        loader = self._make_loader(texts)
        probs  = torch.sigmoid(self._raw_logits(loader)).numpy()
        return np.column_stack([1 - probs, probs])

    def save(self, path):
        import torch
        # Unwrap DataParallel before saving so the checkpoint is device-agnostic
        raw_model = self._model.module if hasattr(self._model, "module") else self._model
        torch.save(
            {
                "model_state": raw_model.state_dict(),
                "config": {
                    "model_name":        self.model_name,
                    "n_unfreeze_layers": self.n_unfreeze_layers,
                    "dropout":           self.dropout,
                    "max_length":        self.max_length,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path, device=None):
        import torch
        data   = torch.load(path, map_location="cpu", weights_only=False)
        cfg    = data["config"]
        obj    = cls(
            model_name=cfg["model_name"],
            n_unfreeze_layers=cfg["n_unfreeze_layers"],
            dropout=cfg["dropout"],
            max_length=cfg["max_length"],
            device=device,
        )
        obj._tokenizer, obj._model = obj._build()
        obj._model.load_state_dict(data["model_state"])
        obj._model = obj._model.to(obj.device)
        obj._model.eval()
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# Transformer / embedding utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_n_gpu():
    """Return number of usable CUDA GPUs (0 if none)."""
    import torch
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def get_transformer(device=None):
    import torch
    from transformers import AutoTokenizer, AutoModel
    device = device or get_device()
    n_gpu  = get_n_gpu()
    print(f"[model] Loading {TRANSFORMER_NAME} on {device}...")
    tokenizer   = AutoTokenizer.from_pretrained(TRANSFORMER_NAME)
    transformer = AutoModel.from_pretrained(TRANSFORMER_NAME).to(device)
    if device == "cuda" and n_gpu > 1:
        print(f"[model] Wrapping transformer in DataParallel ({n_gpu} GPUs)")
        transformer = torch.nn.DataParallel(transformer)
    transformer.eval()
    return tokenizer, transformer, device


def embed_texts(tokenizer, transformer, device, texts,
                batch_size=EMB_BATCH_SIZE, desc="Embedding"):
    import torch
    # Scale batch size across all available GPUs
    n_gpu          = get_n_gpu()
    effective_bs   = batch_size * max(1, n_gpu)
    all_embs = []
    for i in tqdm(range(0, len(texts), effective_bs), desc=desc, leave=False):
        batch = texts[i:i + effective_bs]
        enc   = tokenizer(batch, padding=True, truncation=True,
                          max_length=512, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        mask      = enc["attention_mask"].to(device)
        with torch.no_grad():
            out  = transformer(input_ids, attention_mask=mask)
            # DataParallel may return a plain tuple; handle both cases
            last = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
            m    = mask.unsqueeze(-1).expand(last.size()).float()
            emb  = (last * m).sum(1) / m.sum(1).clamp(min=1e-9)
            all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)
