"""
model.py – Shared utilities: HTMLFeatureExtractor, url_risk_score,
           transformer loading, embedding.
Used by both train.py and app.py.
"""

import math
import re
from collections import Counter
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from sklearn.base import BaseEstimator, TransformerMixin

TRANSFORMER_NAME = "xlm-roberta-base"
EMB_BATCH_SIZE = 16

# TLDs heavily abused by phishing campaigns (no brand names — purely structural)
_SUSPECT_TLDS = {
    "live", "xyz", "tk", "ml", "cf", "gq", "ga", "buzz", "click",
    "link", "shop", "online", "site", "top", "club", "work", "vip",
    "win", "loan", "bid", "stream", "download", "review", "racing",
    "accountant", "science", "party", "trade", "webcam", "zip", "mov",
}

_IP_RE    = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_PUNY_RE  = re.compile(r"xn--", re.I)   # internationalised domain (IDN homograph)


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

    # ── Excessive subdomains  (e.g. secure.update.verify.evil.tk) ────────
    subdomain_count = max(0, len(parts) - 2)
    if subdomain_count >= 4:
        risk += 0.35
        signals.append({"label": f"{subdomain_count} subdomain levels", "impact": 0.35})
    elif subdomain_count == 3:
        risk += 0.15
        signals.append({"label": "3 subdomain levels", "impact": 0.15})

    # ── Many hyphens in domain label ─────────────────────────────────────
    hyphen_count = domain_name.count("-")
    if hyphen_count >= 4:
        risk += 0.25
        signals.append({"label": f"{hyphen_count} hyphens in domain", "impact": 0.25})
    elif hyphen_count == 3:
        risk += 0.10
        signals.append({"label": "3 hyphens in domain", "impact": 0.10})

    # ── Unusual port ─────────────────────────────────────────────────────
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
    # Sort signals by impact descending
    signals.sort(key=lambda s: s["impact"], reverse=True)
    return score, signals


class HTMLFeatureExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def _visible_text(self, soup):
        """Full visible text — used only for the visible_len numeric feature."""
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        t = soup.get_text(" ")
        return re.sub(r"\s+", " ", t).strip()

    def _smart_embed_text(self, soup):
        """
        Curated text for the RoBERTa embedding — fits within the 512-token budget
        by prioritising the parts of the page that carry phishing signals, not
        generic body copy.

        Budget allocation (approximate tokens at ~4 chars/token):
          Segment A  ~128 tok  — page identity: title, h1-h3, meta description
          Segment B  ~256 tok  — form intent:   labels, placeholders, button text,
                                                 link text inside / adjacent to forms,
                                                 all input names/values
          Segment C  ~128 tok  — page footer:   footer element, legal, copyright

        If no form exists, Segment B is filled with all link and button text
        (the interactive intent of the page).

        Segments are joined with plain-text section markers so RoBERTa sees
        positional context without needing special tokens.
        """
        _APPROX_CHARS = {
            "A": 500,   # ~128 tokens
            "B": 1000,  # ~256 tokens
            "C": 500,   # ~128 tokens
        }

        def _clean(text):
            return re.sub(r"\s+", " ", text).strip()

        # ── Segment A: page identity ─────────────────────────────────────────
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

        seg_a = _clean(" ".join(head_parts))[: _APPROX_CHARS["A"]]

        # ── Segment B: form intent ────────────────────────────────────────────
        form_parts = []
        form = soup.find("form")
        if form:
            # Expand context: grab the form's parent container too
            context_node = form.parent if form.parent else form
            for el in context_node.find_all(
                ["label", "input", "button", "select", "textarea", "a", "p", "span"]
            ):
                # Text content of the element
                t = _clean(el.get_text(" ", strip=True))
                if t and len(t) < 200:
                    form_parts.append(t)
                # Attribute hints — placeholder / value / name tell a lot
                for attr in ("placeholder", "value", "name", "aria-label"):
                    v = el.get(attr, "")
                    if v and len(v) < 100:
                        form_parts.append(v)
        else:
            # No form — fall back to all interactive element text
            for el in soup.find_all(["a", "button"]):
                t = _clean(el.get_text(" ", strip=True))
                if t and len(t) < 150:
                    form_parts.append(t)

        seg_b = _clean(" | ".join(form_parts))[: _APPROX_CHARS["B"]]

        # ── Segment C: footer / legal ─────────────────────────────────────────
        footer_parts = []
        footer_node = soup.find("footer") or soup.find(
            attrs={"class": re.compile(r"footer|legal|disclaimer|copyright", re.I)}
        )
        if footer_node:
            footer_parts.append(footer_node.get_text(" ", strip=True))
        else:
            # Last <p> tags often carry disclaimers
            for p in soup.find_all("p")[-5:]:
                t = _clean(p.get_text(" ", strip=True))
                if t:
                    footer_parts.append(t)

        seg_c = _clean(" ".join(footer_parts))[: _APPROX_CHARS["C"]]

        # ── Assemble with markers ─────────────────────────────────────────────
        parts = []
        if seg_a:
            parts.append(f"PAGE TITLE {seg_a}")
        if seg_b:
            parts.append(f"FORM CONTENT {seg_b}")
        if seg_c:
            parts.append(f"FOOTER {seg_c}")

        return _clean(" ".join(parts))

    def _tag_sequence(self, soup, max_tags=5000):
        seq = []
        for tag in soup.find_all():
            seq.append(tag.name)
            if len(seq) >= max_tags:
                break
        return " ".join(seq)

    def _shannon_entropy(self, s):
        if not s:
            return 0.0
        probs = [n / len(s) for n in Counter(s).values()]
        return -sum(p * math.log2(p) for p in probs if p > 0)

    def _max_depth(self, soup):
        maxd = 0
        def depth(node, cur):
            nonlocal maxd
            maxd = max(maxd, cur)
            for c in getattr(node, "contents", []):
                if getattr(c, "name", None):
                    depth(c, cur + 1)
        depth(soup, 0)
        return maxd

    def _link_features(self, soup, base_url=""):
        """Count external vs internal links and unique linked domains."""
        from urllib.parse import urlparse
        try:
            base_host = urlparse(base_url).hostname or ""
        except Exception:
            base_host = ""

        external, internal, domains = 0, 0, set()
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith("#") or href.startswith("javascript"):
                continue
            try:
                parsed = urlparse(href)
                host = parsed.hostname or ""
                if host and host != base_host:
                    external += 1
                    # strip www. and record eTLD+1 (last two parts)
                    parts = host.lstrip("www.").split(".")
                    domains.add(".".join(parts[-2:]) if len(parts) >= 2 else host)
                elif href.startswith("http"):
                    internal += 1
            except Exception:
                pass
        return external, internal, len(domains)

    def transform(self, X, y=None, urls=None):
        rows, visible_texts, tag_sequences = [], [], []
        if urls is None:
            urls = [""] * len(X)
        for html_text, url in tqdm(zip(X, urls), desc="HTML features", leave=False, total=len(X)):
            try:
                soup = BeautifulSoup(html_text, "lxml")
            except Exception:
                soup = BeautifulSoup(html_text, "html.parser")

            doc = {}
            doc["raw_len"] = len(html_text)
            doc["shannon_entropy"] = self._shannon_entropy(html_text)

            tags = [t.name for t in soup.find_all()]
            tc = Counter(tags)
            for t in ["a", "img", "script", "iframe", "form", "input", "link", "meta", "button", "svg"]:
                doc[f"count_tag__{t}"] = tc.get(t, 0)

            doc["num_unique_tags"] = len(set(tags))
            doc["max_dom_depth"] = self._max_depth(soup)

            # visible_len is a numeric feature — use full text so it stays meaningful
            visible_full = self._visible_text(soup)
            doc["visible_len"] = len(visible_full)

            # embedding input — curated 512-token budget prioritising form/interactive signals
            visible_texts.append(self._smart_embed_text(soup))

            ext, intern, unique_domains = self._link_features(soup, url)
            doc["count_external_links"]    = ext
            doc["count_internal_links"]    = intern
            doc["count_unique_link_domains"] = unique_domains
            # ratio: 0 external links on a page with links = strong phishing signal
            total_links = ext + intern
            doc["ratio_external_links"] = ext / total_links if total_links > 0 else 0.0

            seq = self._tag_sequence(soup)
            tag_sequences.append(seq)

            rows.append(doc)

        df = pd.DataFrame(rows)
        return df, {"visible_texts": visible_texts, "tag_sequences": tag_sequences}


def get_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_transformer(device=None):
    import torch
    from transformers import AutoTokenizer, AutoModel
    device = device or get_device()
    print(f"[model] Loading {TRANSFORMER_NAME} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_NAME)
    transformer = AutoModel.from_pretrained(TRANSFORMER_NAME).to(device)
    transformer.eval()
    return tokenizer, transformer, device


def embed_texts(tokenizer, transformer, device, texts, batch_size=EMB_BATCH_SIZE, desc="Embedding"):
    import torch
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc, leave=False):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = transformer(input_ids, attention_mask=mask)
            last = out.last_hidden_state
            m = mask.unsqueeze(-1).expand(last.size()).float()
            emb = (last * m).sum(1) / m.sum(1).clamp(min=1e-9)
            all_embs.append(emb.cpu().numpy())
    return np.vstack(all_embs)
