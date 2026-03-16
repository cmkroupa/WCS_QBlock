"""
app.py – QBlock phishing detection API.

Requires artifacts/model.joblib (produced by train.py).
Requires Playwright:  pip install playwright && python3 -m playwright install chromium

Endpoints:
    POST /api/scan   { "url": "..." }  →  phishing verdict + branch scores
    GET  /health                       →  server status

Start:
    python3 app.py
"""

import os
from pathlib import Path

# Python 3.14 + macOS ARM64: fork() after loading torch causes SIGSEGV at shutdown.
# This env var tells the OS not to enforce fork safety checks, preventing the crash.
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import joblib
import numpy as np
import shap
from flask import Flask, jsonify, request
from flask_cors import CORS

# ── Feature switches ──────────────────────────────────────────────────────────
# USE_URL_RISK      True  → blend URL structural signals into final score
# USE_HTML_OVERRIDE True  → hard rules override score for credential forms,
#                           iframes, obfuscated content, etc.
USE_URL_RISK      = False
USE_HTML_OVERRIDE = False

from model import HTMLFeatureExtractor, embed_texts, get_transformer, url_risk_score

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
MODEL_PATH    = ARTIFACTS_DIR / "model.joblib"

app = Flask(__name__)
CORS(app)

# ── Lazy-loaded globals ───────────────────────────────────────────────────────
_bundle           = None
_tokenizer        = None
_transformer      = None
_device           = None
_explainer_numeric = None   # shap.TreeExplainer for Voter A
_explainer_meta    = None   # shap.TreeExplainer for Meta blender


def load_model():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No model found at {MODEL_PATH}. "
                "Train one first:  python3 train.py data/training"
            )
        print(f"[app] Loading model from {MODEL_PATH}...")
        _bundle = joblib.load(MODEL_PATH)
        print("[app] Model ready.")
    return _bundle


def load_transformer():
    global _tokenizer, _transformer, _device
    if _tokenizer is None:
        _tokenizer, _transformer, _device = get_transformer()
    return _tokenizer, _transformer, _device


def get_explainers():
    """Return (explainer_numeric, explainer_meta), built once and cached."""
    global _explainer_numeric, _explainer_meta
    if _explainer_numeric is None:
        bundle = load_model()
        # TreeExplainer is exact and fast for XGBoost — no background samples needed.
        _explainer_numeric = shap.TreeExplainer(bundle["xgb_numeric"])
        _explainer_meta    = shap.TreeExplainer(bundle["xgb_meta"])
        print("[app] SHAP explainers ready.")
    return _explainer_numeric, _explainer_meta


def compute_shap(Xn, numeric_cols, meta_input):
    """
    Returns a dict ready to send to the frontend:
      numeric_top  — top 8 HTML features by |SHAP|, with name/raw-value/shap-impact
      meta         — how much each voter (A numeric, B bert) drove the final score
    """
    exp_num, exp_meta = get_explainers()

    # Voter A SHAP — shape (1, n_features), positive = pushes toward phishing
    sv_num  = exp_num.shap_values(Xn)[0]           # 1-D array, one value per feature
    sv_meta = exp_meta.shap_values(meta_input)[0]   # 2-D → [shap_P_xgb, shap_P_bert]

    # Build ranked list of numeric features
    pairs = sorted(
        zip(numeric_cols, Xn[0].tolist(), sv_num.tolist()),
        key=lambda x: abs(x[2]),
        reverse=True,
    )
    top8 = [
        {"feature": name, "raw_value": round(float(raw), 4), "impact": round(float(sv), 4)}
        for name, raw, sv in pairs[:8]
    ]

    # Meta contributions — which voter pushed the final score more
    meta_shap_names = ["XGBoost (HTML features)", "XGBoost (BERT embeddings)"]
    meta_contribs = [
        {"voter": name, "impact": round(float(sv), 4)}
        for name, sv in zip(meta_shap_names, sv_meta.tolist())
    ]

    return {"numeric_top": top8, "meta_contributions": meta_contribs}


def fetch_html(url: str) -> str:
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
            # Don't load images or fonts — faster, same HTML content
            java_script_enabled=True,
        )
        page = ctx.new_page()

        # Block images/fonts to speed up load — HTML content is unchanged
        page.route("**/*.{png,jpg,jpeg,gif,webp,svg,ico,woff,woff2,ttf}", lambda r: r.abort())

        try:
            page.goto(url, wait_until="domcontentloaded", timeout=20_000)
            # Wait a moment for JS to populate the DOM (phishing kits often inject content)
            page.wait_for_timeout(2_000)
        except PWTimeout:
            # Page timed out — grab whatever DOM loaded so far
            print(f"[fetch] Playwright timeout for {url}, using partial DOM")

        html = page.content()
        browser.close()
    return html[:500_000]


THRESHOLD   = 0.50   # final verdict threshold
SUSPICIOUS  = 0.25   # shown as suspicious but not phishing


def label(score: float) -> str:
    if score >= THRESHOLD:
        return "phishing"
    if score >= SUSPICIOUS:
        return "suspicious"
    return "safe"


def html_override(features: dict) -> tuple[bool, str]:
    """
    Detect specific phishing patterns directly from HTML features.
    Returns (triggered: bool, reason: str).

    These rules target known attack techniques, not a statistical threshold,
    so legitimate complex pages (QR generators, dashboards) won't be caught.
    """
    form    = features.get("count_tag__form",        0)
    input_  = features.get("count_tag__input",       0)
    iframe  = features.get("count_tag__iframe",      0)
    script  = features.get("count_tag__script",      0)
    unique  = features.get("num_unique_tags",         0)
    vis     = features.get("visible_len",             0)
    entr    = features.get("shannon_entropy",         0)
    ext     = features.get("count_external_links",   0)
    anchors = features.get("count_tag__a",            0)
    ratio   = features.get("ratio_external_links",   0.0)

    # Rule 1 — Credential harvesting kit:
    # Form + inputs + almost no text + zero external links
    # Legit payment pages (Stripe, parking, etc.) always link out to privacy/terms/processor
    if form >= 1 and input_ >= 3 and vis < 300 and ext == 0:
        return True, "credential_harvest"

    # Rule 2 — Hidden iframe injection:
    # Multiple iframes = classic phishing/malware delivery
    if iframe >= 2:
        return True, "iframe_injection"

    # Rule 3 — Obfuscated shell:
    # High entropy (base64/encoded payloads) with almost no visible text
    if entr > 5.8 and vis < 150:
        return True, "obfuscated_content"

    # Rule 4 — Script-only skeleton:
    # Lots of scripts but very few unique HTML elements = automated phishing kit
    if script > 15 and unique < 6:
        return True, "script_skeleton"

    # Rule 5 — Empty form farm:
    # Multiple forms with essentially no content
    if form >= 3 and vis < 100:
        return True, "empty_form_farm"

    # Rule 6 — Link isolation:
    # Page has links but zero go outside → classic phishing trap (keeps victim on fake site)
    if anchors >= 5 and ext == 0:
        return True, "link_isolation"

    # Rule 7 — No external links + credential form:
    # Form with inputs and every link stays on-page
    if form >= 1 and input_ >= 2 and ratio < 0.05 and anchors >= 3:
        return True, "isolated_credential_form"

    return False, ""


def run_inference(url: str) -> dict:
    # 0. URL structural risk — computed before any HTML fetch, never suffers from
    #    bot challenges or distribution shift between saved training files and live pages
    url_score, url_signals = url_risk_score(url)

    # 1. Fetch HTML
    html_text = fetch_html(url)

    # 2. Extract HTML numeric features (pass url so link features work correctly)
    extractor = HTMLFeatureExtractor()
    df_num, extras = extractor.transform([html_text], urls=[url])
    vis  = extras["visible_texts"]
    tags = extras["tag_sequences"]

    # 3. Embed with XLM-RoBERTa (visible text + tag sequence)
    tok, trans, dev = load_transformer()
    emb_vis  = embed_texts(tok, trans, dev, vis,  desc="vis")
    emb_tag  = embed_texts(tok, trans, dev, tags, desc="tag")
    emb_bert = np.hstack([emb_vis, emb_tag]).astype(np.float32)  # (1, 1536)

    # 4. Load model bundle
    bundle       = load_model()
    numeric_cols = bundle["numeric_columns"]

    Xn = df_num[numeric_cols].fillna(0).values.astype(np.float32)

    # 5. Stacking inference (new model format)
    if "xgb_meta" in bundle:
        score_xgb  = float(bundle["xgb_numeric"].predict_proba(Xn)[0, 1])
        score_bert = float(bundle["xgb_bert"].predict_proba(emb_bert)[0, 1])
        meta_input = np.array([[score_xgb, score_bert]], dtype=np.float32)
        score_final = float(bundle["xgb_meta"].predict_proba(meta_input)[0, 1])
    else:
        # Legacy model format (single XGBoost on [numeric | vis_emb | tag_emb])
        X_full      = np.hstack([Xn, emb_bert])
        score_final = float(bundle["model"].predict_proba(X_full)[0, 1])
        # Approximate individual scores by zeroing out the other half
        n_num       = bundle.get("n_numeric", len(numeric_cols))
        X_a         = X_full.copy(); X_a[:, n_num:] = 0.0
        X_b         = X_full.copy(); X_b[:, :n_num] = 0.0
        score_xgb   = float(bundle["model"].predict_proba(X_a)[0, 1])
        score_bert  = float(bundle["model"].predict_proba(X_b)[0, 1])

    # HTML pattern override — specific attack signatures beat the blend
    features      = df_num.iloc[0].to_dict()
    override_hit, override_reason = html_override(features)
    if USE_HTML_OVERRIDE and override_hit:
        score_final = max(score_final, 0.85)   # hard floor, not a cap

    # URL risk blend
    if USE_URL_RISK:
        if url_score >= 0.60:
            score_final = max(score_final, 0.75)
        elif url_score >= 0.30:
            score_final = score_final * 0.60 + url_score * 0.40

    score_final = round(score_final, 4)

    # SHAP reasoning — only for stacking model (legacy path has no named numeric cols)
    shap_data = None
    if "xgb_meta" in bundle:
        try:
            shap_data = compute_shap(Xn, numeric_cols, meta_input)
        except Exception as shap_err:
            print(f"[app] SHAP skipped: {shap_err}")

    return {
        "url": url,
        "blend": {
            "final_prediction": label(score_final),
            "final_score":      score_final,
        },
        "split_a": {
            "score":      round(score_xgb, 4),
            "prediction": label(score_xgb),
        },
        "split_b": {
            "score":      round(score_bert, 4),
            "prediction": label(score_bert),
        },
        "url_risk": {
            "score":   url_score,
            "signals": url_signals,
        },
        "override": override_reason or None,
        "shap":     shap_data,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":      "ok",
        "model_ready": MODEL_PATH.exists(),
        "model_path":  str(MODEL_PATH),
    })


@app.route("/api/scan", methods=["POST"])
def scan():
    data = request.get_json(force=True) or {}
    url  = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "Missing 'url' field"}), 400
    try:
        return jsonify(run_inference(url))
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("[app] QBlock backend starting on http://localhost:5001")
    try:
        load_model()
    except FileNotFoundError as e:
        print(f"[app] WARNING: {e}")
    app.run(host="0.0.0.0", port=5001, debug=False)
