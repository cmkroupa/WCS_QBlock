# QBlock

QR code phishing scanner. Decodes a QR code, fetches the final destination URL, and classifies it as **safe**, or **phishing**.

---

## Requirements

- Python 3.11+
- Node.js 18+
- ~8 GB RAM for the full training run (RoBERTa embeddings)
- Training data in `backend/data/training/Phish/` and `backend/data/training/NotPhish/` — not included in this repo

---

## 1. Backend setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m playwright install chromium
```

---

## 2. Training data

The dataset (10,296 HTML files, ~2.7 GB) is not included in this repo. Download it from Google Drive:

**[Download qblock_training_data.zip](https://drive.google.com/file/d/1YJ0FjIGg3hiB46Gg8aZ7It6ZkkP5zlgo/view?usp=sharing)**

Extract into the backend directory:

```bash
unzip qblock_training_data.zip -d backend/
```

This places files at `backend/data/training/Phish/` and `backend/data/training/NotPhish/`.

---

## 3. Train the model

**Full run** — embeds all pages with XLM-RoBERTa, then trains all classifiers. Takes 30–90 min depending on hardware. Required on first run.

```bash
cd backend
source .venv/bin/activate
python3 train.py data/training
```

**Fast run** — skips re-embedding and loads the cached embeddings. Use this after the first full run when iterating on hyperparameters.

```bash
python3 train.py data/training --fast
```

Outputs written to `backend/artifacts/`:

| File | Description |
|---|---|
| `model.joblib` | Full model bundle (voters, meta-learner, Platt calibrator, scaler, PCA) — gitignored |
| `emb_cache.npz` | Cached XLM-RoBERTa embeddings — gitignored |
| `minhash_cache.pkl` | Near-duplicate detection cache — gitignored |
| `training_summary.json` | AUC, precision, recall, F1, confusion matrix for all three models |

> All three cache/model files are gitignored. On a fresh clone you must do a full run first before `--fast` is available.

---

## 4. Start the backend

```bash
cd backend
source .venv/bin/activate
python3 app.py
```

Runs on `http://localhost:5001`.

---

## 5. Frontend

```bash
cd frontend
npm install
npm run dev
```

Runs on `http://localhost:3000`.

## API

```
POST /api/scan
Content-Type: application/json

{ "url": "https://example.com" }
```

## Training results

Latest training run on 14,651 files (7,326 phishing / 7,325 benign), 20% held-out test set:

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Voter A — XGBoost (numeric) | 0.9766 | 0.9157 | 0.8901 | 0.9027 |
| Voter B — LR-BERT (embeddings) | 0.9707 | 0.8497 | 0.9162 | 0.8817 |
| **Meta blend (calibrated)** | **0.9842** | **0.9055** | **0.9367** | **0.9208** |

Full per-model breakdown including confusion matrices is in `backend/artifacts/training_summary.json`.
