"""
app.py – QBlock phishing detection API.

Requires artifacts/model.joblib (produced by train.py).
Requires Playwright:  pip install playwright && python3 -m playwright install chromium

Endpoints:
    POST /api/scan   { "url": "..." }  →  phishing verdict + branch scores
    GET  /health                       →  server status

Start:
    python3 app.py

Inference pipeline (mirrors train.py preprocessing exactly):
    raw HTML
      → HTMLFeatureExtractor   → numeric features
                                    → StandardScaler           [bundle["scaler"]]
                                    → Voter A XGBoost          → P_xgb
      → combine_texts(vis, struct)   → PhishBERTClassifier     → P_bert
                                       (fine-tuned RoBERTa + linear head)
      → [P_xgb, P_bert]             → Meta LogisticRegression + Platt calibration
                                    → final score
"""

import os
from pathlib import Path

# Python 3.14 + macOS ARM64: fork() after loading torch causes SIGSEGV at shutdown.
os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import joblib
import numpy as np
import shap
from bs4 import BeautifulSoup
from flask import Flask, jsonify, request
from flask_cors import CORS

from helpers import fetch_html, label
from model import HTMLFeatureExtractor, url_risk_score, combine_texts, PhishBERTClassifier

ARTIFACTS_DIR  = Path(__file__).parent / "artifacts"
MODEL_PATH     = ARTIFACTS_DIR / "model.joblib"
BERT_CLF_PATH  = ARTIFACTS_DIR / "bert_classifier.pt"

app = Flask(__name__)
CORS(app)

# ── Lazy-loaded globals ───────────────────────────────────────────────────────
_bundle            = None
_bert_clf          = None   # PhishBERTClassifier (Voter B)
_explainer_numeric = None   # shap.TreeExplainer for Voter A (XGBoost)


def load_model():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No model found at {MODEL_PATH}. "
                "Train one first:  python3 train.py data/training"
            )
        print(f"[app] Loading model from {MODEL_PATH}…")
        _bundle = joblib.load(MODEL_PATH)
        print("[app] Model ready.")
    return _bundle


def load_bert_clf():
    global _bert_clf
    if _bert_clf is None:
        path = Path(load_model().get("bert_clf_path", str(BERT_CLF_PATH)))
        if not path.exists():
            raise FileNotFoundError(
                f"PhishBERT model not found at {path}. Re-run train.py."
            )
        print(f"[app] Loading PhishBERT classifier from {path}…")
        _bert_clf = PhishBERTClassifier.load(path)
        print("[app] PhishBERT ready.")
    return _bert_clf


def get_explainers():
    """Return explainer_numeric (TreeExplainer on Voter A), built once and cached."""
    global _explainer_numeric
    if _explainer_numeric is None:
        bundle = load_model()
        _explainer_numeric = shap.TreeExplainer(bundle["xgb_numeric"])
        print("[app] SHAP explainer ready.")
    return _explainer_numeric


def compute_shap(Xn_raw, Xn_scaled, numeric_cols, meta_input):
    """
    Return a dict ready to send to the frontend:
      numeric_top  — top 8 HTML features by |SHAP|, with feature name,
                     original (unscaled) raw value, and SHAP impact.
      meta         — per voter: learned LR weight, input probability, and
                     per-sample contribution (weight × P, in log-odds units).

    Parameters
    ----------
    Xn_raw    : np.ndarray (1, n_features)  — unscaled numeric features (for display)
    Xn_scaled : np.ndarray (1, n_features)  — scaled numeric features   (for SHAP)
    numeric_cols : list[str]
    meta_input   : np.ndarray (1, 2)        — [P_xgb, P_lr]
    """
    exp_num = get_explainers()

    # Voter A SHAP — computed on scaled features (same space the model trained on).
    # shap_values() output differs by version:
    #   SHAP <0.40  → list [neg_class_vals, pos_class_vals], each (n_samples, n_features)
    #   SHAP ≥0.40  → single ndarray (n_samples, n_features) for the positive class
    _sv_raw = exp_num.shap_values(Xn_scaled)
    if isinstance(_sv_raw, list):
        # Take positive-class values for the single sample
        sv_num = np.array(_sv_raw[1])[0]
    else:
        sv_num = np.array(_sv_raw)[0]

    # Meta contributions — expose the LR coefficients directly so the frontend
    # can show the model's learned trust in each voter, plus the per-sample
    # contribution (coef × P) in log-odds units.
    meta_lr    = load_model()["meta_lr"]
    coefs      = meta_lr.coef_[0]                        # [w_xgb, w_lr]
    probs      = meta_input[0]                           # [P_xgb, P_lr]
    contribs   = (coefs * probs).tolist()                # per-sample log-odds push

    pairs = sorted(
        zip(numeric_cols, Xn_raw[0].tolist(), sv_num.tolist()),
        key=lambda x: abs(x[2]),
        reverse=True,
    )
    shap_features = [
        {
            "feature":   name,
            "raw_value": round(float(raw), 4),
            "impact":    round(float(sv),  4),
        }
        for name, raw, sv in pairs          # all features, sorted by |SHAP|
    ]

    meta_shap_names = [
        "XGBoost (HTML numeric features)",
        "PhishBERT (fine-tuned RoBERTa)",
    ]
    meta_contribs = [
        {
            "voter":       name,
            "weight":      round(float(coef), 4),   # learned LR coefficient
            "probability": round(float(prob), 4),   # this sample's voter score
            "impact":      round(float(contrib), 4),# weight × probability (log-odds)
        }
        for name, coef, prob, contrib in zip(
            meta_shap_names, coefs.tolist(), probs.tolist(), contribs
        )
    ]

    return {"numeric_top": shap_features, "meta_contributions": meta_contribs}


def _core_inference(url: str) -> dict:
    """
    Run the full inference pipeline and return all intermediate values.
    Called by both run_inference() and run_pipeline_inference() to avoid duplication.
    """
    # ── 1. Fetch HTML (follows redirects) ─────────────────────────────────────
    html_text, final_url = fetch_html(url)

    # ── 2. URL structural risk — on final URL after redirects ─────────────────
    url_score, url_signals = url_risk_score(final_url)

    # ── 3. Extract HTML numeric features ──────────────────────────────────────
    df_num, extras = HTMLFeatureExtractor().transform([html_text])
    vis          = extras["visible_texts"]
    struct_cores = extras["struct_cores"]

    # ── 4. Load model bundle + PhishBERT ──────────────────────────────────────
    bundle       = load_model()
    bert_clf     = load_bert_clf()
    numeric_cols = bundle["numeric_columns"]
    Xn_raw       = df_num[numeric_cols].fillna(0).values.astype(np.float32)

    # ── 5. Stacking inference ──────────────────────────────────────────────────
    Xn_scaled  = bundle["scaler"].transform(Xn_raw).astype(np.float32)
    score_xgb  = float(bundle["xgb_numeric"].predict_proba(Xn_scaled)[0, 1])

    combined_text = combine_texts(vis[0] if vis else "", struct_cores[0] if struct_cores else "")
    score_bert    = float(bert_clf.predict_proba([combined_text])[0, 1])
    meta_input = np.array([[score_xgb, score_bert]], dtype=np.float32)

    raw_meta_p  = bundle["meta_lr"].predict_proba(meta_input)[:, 1].reshape(-1, 1)
    score_final = round(float(bundle["meta_calibrator"].predict_proba(raw_meta_p)[0, 1]), 4)

    # ── 7. SHAP ────────────────────────────────────────────────────────────────
    shap_data = None
    if "xgb_numeric" in bundle:
        try:
            shap_data = compute_shap(Xn_raw, Xn_scaled, numeric_cols, meta_input)
        except Exception as e:
            print(f"[app] SHAP skipped: {e}")

    return {
        "html_text":      html_text,
        "final_url":      final_url,
        "url_score":      url_score,
        "url_signals":    url_signals,
        "df_num":         df_num,
        "vis":            vis,
        "struct_cores":   struct_cores,
        "combined_text":  combined_text,
        "bundle":         bundle,
        "numeric_cols":   numeric_cols,
        "Xn_raw":         Xn_raw,
        "Xn_scaled":      Xn_scaled,
        "score_xgb":      score_xgb,
        "score_bert":     score_bert,
        "meta_input":     meta_input,
        "score_final":    score_final,
        "shap_data":      shap_data,
    }


def run_inference(url: str) -> dict:
    c = _core_inference(url)
    return {
        "url": c["final_url"],
        "blend": {
            "final_prediction": label(c["score_final"]),
            "final_score":      c["score_final"],
        },
        "split_a": {
            "score":      round(c["score_xgb"],  4),
            "prediction": label(c["score_xgb"]),
        },
        "split_b": {
            "score":      round(c["score_bert"], 4),
            "prediction": label(c["score_bert"]),
        },
        "url_risk": {
            "score":   c["url_score"],
            "signals": c["url_signals"],
        },
        "shap": c["shap_data"],
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
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        return jsonify(run_inference(url))
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def run_pipeline_inference(url: str) -> dict:
    """Full inference pipeline returning every intermediate value for visualisation."""
    original_url = url
    c = _core_inference(url)

    try:
        soup_title = BeautifulSoup(c["html_text"][:20_000], "html.parser")
        title_tag  = soup_title.find("title")
        page_title = title_tag.get_text(strip=True)[:120] if title_tag else ""
    except Exception:
        page_title = ""

    redirected   = c["final_url"].rstrip("/") != original_url.rstrip("/")
    all_features = {k: round(float(v), 4) for k, v in c["df_num"].iloc[0].to_dict().items()}

    return {
        "url": c["final_url"],
        "stages": {
            "url_analysis": {
                "original_url": original_url,
                "final_url":    c["final_url"],
                "redirected":   redirected,
                "score":        c["url_score"],
                "signals":      c["url_signals"],
            },
            "fetch": {
                "html_bytes": len(c["html_text"]),
                "title":      page_title,
            },
            "html_features": {
                "features": all_features,
            },
            "text_extraction": {
                "visible_text":        (c["vis"][0]          if c["vis"]          else "")[:1500],
                "visible_text_len":    len(c["vis"][0])      if c["vis"]          else 0,
                "structural_core":     (c["struct_cores"][0] if c["struct_cores"] else "")[:1500],
                "structural_core_len": len(c["struct_cores"][0]) if c["struct_cores"] else 0,
            },
            "embedding": {
                "voter_b_model": "PhishBERTClassifier (xlm-roberta-base + head)",
                "bert_input_len": len(c["combined_text"]),
            },
            "voter_a": {
                "score":      round(c["score_xgb"], 4),
                "prediction": label(c["score_xgb"]),
                "shap_top":   c["shap_data"]["numeric_top"] if c["shap_data"] else [],
            },
            "voter_b": {
                "score":      round(c["score_bert"], 4),
                "prediction": label(c["score_bert"]),
            },
            "meta": {
                "inputs":             [round(c["score_xgb"], 4), round(c["score_bert"], 4)],
                "lr_weights":         [round(float(coef), 4) for coef in c["bundle"]["meta_lr"].coef_[0]],
                "lr_intercept":       round(float(c["bundle"]["meta_lr"].intercept_[0]), 4),
                "calibrated_score":   c["score_final"],
                "prediction":         label(c["score_final"]),
                "shap_contributions": c["shap_data"]["meta_contributions"] if c["shap_data"] else [],
            },
        },
    }


@app.route("/api/pipeline", methods=["POST"])
def pipeline():
    data = request.get_json(force=True) or {}
    url  = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "Missing 'url' field"}), 400
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        return jsonify(run_pipeline_inference(url))
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
