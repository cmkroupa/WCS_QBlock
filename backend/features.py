"""
features.py – HTML feature extraction for QBlock.

Exports:
    HTMLFeatureExtractor  — sklearn-compatible transformer; produces both
                            numeric features (for Voter A / XGBoost) and
                            curated text (for Voter B / PhishBERT).
    _process_single_html  — top-level worker used by multiprocessing.
"""

import math
import re
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from sklearn.base import BaseEstimator, TransformerMixin

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
        if len(X) == 1:
            # Skip joblib entirely for single-document inference.
            # joblib.Parallel(prefer="processes") forks new processes, which
            # deadlocks inside Flask worker threads after PyTorch has been loaded.
            results = [_process_single_html(X[0])]
        else:
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

    # Hidden divs — phishing pages frequently hide credential harvesting UI
    doc["count_hidden_divs"] = sum(
        1 for tag in soup.find_all(["div", "span", "section"])
        if re.search(r"display\s*:\s*none|visibility\s*:\s*hidden",
                     (tag.get("style") or ""), re.I)
    )

    visible_text = ex._smart_embed_text(soup_embed)
    struct_core  = ex._structural_core(soup_embed)

    return doc, visible_text, struct_core
