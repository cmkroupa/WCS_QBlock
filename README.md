# QBlock

QR code phishing scanner. Decodes a QR code, fetches the destination, and classifies it as safe, suspicious, or phishing — before the browser opens it.

---

## Requirements

- Python 3.11+
- Node.js 18+
- ~8 GB RAM for the full training run (RoBERTa embeddings)
- Training data in `backend/data/training/Phish/` and `backend/data/training/NotPhish/` — not included in this repo

---

## 1. Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m playwright install chromium
```

---

## 2. Training data

The dataset (10,296 HTML files, 2.7 GB) is not included in this repo. Download it from Google Drive:

**[Download qblock_training_data.zip](https://drive.google.com/file/d/1YJ0FjIGg3hiB46Gg8aZ7It6ZkkP5zlgo/view?usp=sharing)**

Extract into the backend directory:

```bash
unzip qblock_training_data.zip -d backend/
```

This places files at `backend/data/training/Phish/` and `backend/data/training/NotPhish/`.

---

## 3. Train the model

Training data goes in:
```
backend/data/training/
    Phish/       ← phishing HTML files
    NotPhish/    ← legitimate HTML files
```

**Full run** — embeds all pages with XLM-RoBERTa, then trains the classifiers. Takes 30–90 min. Run this first.

```bash
cd backend
python3 train.py data/training
```

**Fast run** — skips re-embedding, retrains XGBoost from the existing embedding cache. Use this after the first full run if you tweak hyperparameters.

```bash
python3 train.py data/training --fast
```

Outputs written to `backend/artifacts/`:

| File | Description |
|---|---|
| `model.joblib` | Trained model bundle — all 3 XGBoosts |
| `emb_cache.npz` | Cached RoBERTa embeddings (gitignored, stays local) |
| `training_summary.json` | AUC, accuracy, split sizes |

> `emb_cache.npz` is gitignored (49 MB). On a fresh clone you must do a full run first, then `--fast` is available.

---

## 4. Start the backend

```bash
cd backend
source .venv/bin/activate
python3 app.py
```

Runs on `http://localhost:5001`.

### Switches (top of `app.py`)

| Constant | Default | Effect |
|---|---|---|
| `USE_URL_RISK` | `False` | Blend URL structural signals into the final score |
| `USE_HTML_OVERRIDE` | `False` | Hard rules that can override the model (credential forms, iframes, etc.) |

Flip to `True` and restart to enable.

---

## 5. Frontend

```bash
cd frontend
npm install
npm run dev
```

Runs on `http://localhost:3000`.

---

## Architecture

```
QR code decoded in browser
         ↓
    URL extracted
         ↓
    Page fetched  ──── Playwright (headless Chromium, executes JS)
         ↓
    HTML parsed
         ↓
    ┌────────────────────┬────────────────────┐
    │  Voter A           │  Voter B           │
    │  XGBoost           │  XGBoost           │
    │  19 HTML features  │  RoBERTa embeds    │
    └─────────┬──────────┴──────────┬─────────┘
              └──────────┬──────────┘
                         ↓
                  Meta XGBoost
                         ↓
             Safe / Suspicious / Phishing
```

**Voter A** — extracts 19 structural signals from the HTML: form counts, input fields, link ratios, iframe presence, visible text length, Shannon entropy, unique tag count, external link domains.

**Voter B** — runs the page through XLM-RoBERTa (`xlm-roberta-base`) using a curated 512-token budget. Prioritises page title and headings, then form context (labels, placeholders, button text), then footer. Pushes intent signals to the front rather than generic body copy.

**Meta model** — takes `[P_xgb, P_bert]` as input and blends them into a final score. Thresholds: `< 0.25` = Safe · `0.25–0.50` = Suspicious · `≥ 0.50` = Phishing.

---

## API

```
POST /api/scan
Content-Type: application/json

{ "url": "https://example.com" }
```

Response shape:
```json
{
  "url": "https://example.com",
  "blend": { "final_prediction": "safe", "final_score": 0.12 },
  "split_a": { "prediction": "safe", "score": 0.09 },
  "split_b": { "prediction": "safe", "score": 0.14 },
  "shap": {
    "numeric_top": [
      { "feature": "count_tag__a", "raw_value": 12, "impact": -0.34 }
    ],
    "meta_contributions": [
      { "voter": "xgb_score", "impact": -0.21 },
      { "voter": "bert_score", "impact": -0.18 }
    ]
  }
}
```

---

## Reference

The model architecture is based on the Kaggle notebook [Phishing Detection using RoBERTa + XGB](https://www.kaggle.com/code/campkittydog/phishing-detection-using-roberta-xgb) (`phishing-detection-using-roberta-xgb.ipynb`), which documents the original research and embedding experiments. That notebook is not included in this repo — `train.py` is the production implementation derived from it.

---

## Debug tool

Check what HTML the scanner actually receives for any URL:

```bash
cd backend
python3 fetch_debug.py https://example.com
python3 fetch_debug.py https://example.com --playwright    # use headless browser
python3 fetch_debug.py https://example.com --features      # show extracted feature values
python3 fetch_debug.py https://example.com --text          # show visible text only
```
