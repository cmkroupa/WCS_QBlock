# Codex QBlock — QR URL Phishing Blocker

A real-time QR code safety gateway. Scans a QR code via camera, extracts the URL, runs a dual-branch ML pipeline, and returns a phishing verdict before the user ever trusts the destination.

## Architecture

- **Frontend:** Next.js live camera scanner (`/frontend`)
- **Backend:** Python Flask API (`/backend`)

**Pipeline:**
```
QR scan → URL → split()
  A) 35-feature extraction matrix → Logistic Regression → blend()
  B) DistilBERT (URL + live webpage content) → blend()
blend() → XGBoost calibrated output → Safe / Phishing verdict
```

---

## Large Files (Google Drive)

The following files exceed GitHub's size limit and are **not included in this repo**. Download `codex_qblock_large_files.tar.gz` from Google Drive and extract it from the repo root:

https://drive.google.com/file/d/1y7U6NKN5blFS0trz3kcNX9fg3R8qRC0i/view?usp=sharing

```bash
tar -xzf codex_qblock_large_files.tar.gz -C backend/artifacts/
```

This places all three files at the correct paths:

| File | Path in project |
|------|----------------|
| `tranco_full.csv` | `backend/artifacts/tranco_full.csv` |
| `feature_matrix.csv` | `backend/artifacts/feature_matrix.csv` |
| `feature_matrix.parquet` | `backend/artifacts/feature_matrix.parquet` |

> If you skip the feature matrix files, the backend will auto-build them on first run by downloading the dataset from HuggingFace (this takes several minutes). `tranco_full.csv` is required for Tranco rank features — without it those features default to -1.

---

## Setup

### 1. Backend (Python 3.11)

```bash
cd backend
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Backend runs at `http://localhost:5001`.

### 2. Frontend (Next.js)

```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```

Frontend runs at `http://localhost:3000`.

---

## Build Feature Matrix Manually

The backend auto-builds the feature matrix if not found. To trigger it manually:

```bash
curl -X POST http://localhost:5001/api/build-feature-matrix \
     -H 'Content-Type: application/json' -d '{}'
```

Artifacts saved to `backend/artifacts/`:
- `feature_matrix.csv` — full feature matrix
- `feature_matrix.parquet` — parquet copy
- `feature_matrix_meta.json` — build metadata
