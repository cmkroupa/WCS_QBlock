"""
train.py – Trains the phishing detection stacking ensemble from HTML files.

Based on: phishing-detection-using-roberta-xgb.ipynb

Usage:
    python3 train.py data/training          # full run (embeds with RoBERTa, trains XGBoosts)
    python3 train.py data/training --fast   # skip embedding, retrain XGBoosts from cache

Expected directory structure:
    data/training/
        Phish/        ← phishing HTML files
        NotPhish/     ← legitimate HTML files

Architecture (stacking ensemble):
    Voter A — XGBoost on HTML numeric features (15-dim)       → P_xgb
    Voter B — XGBoost on XLM-RoBERTa embeddings (1536-dim)   → P_bert
    Meta    — XGBoost blends [P_xgb, P_bert]                 → final score

Output:
    artifacts/model.joblib          ← trained model bundle (all 3 XGBoosts)
    artifacts/emb_cache.npz         ← cached embeddings (reuse with --fast)
    artifacts/training_summary.json
"""

import json
import multiprocessing
import sys
from pathlib import Path

# Python 3.14 + macOS ARM64: fork() after loading torch/tokenizers causes SIGSEGV.
# "spawn" starts fresh child processes instead of forking, avoiding the crash.
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm.auto import tqdm

from model import HTMLFeatureExtractor, embed_texts, get_transformer

RANDOM_SEED   = 42
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
EMB_CACHE     = ARTIFACTS_DIR / "emb_cache.npz"    # embeddings only — re-use when features change
CACHE_PATH    = EMB_CACHE  # alias kept for backwards compat

# n_estimators is intentionally NOT in the grids below.
# Early stopping (patience=30) determines the actual tree count — the model
# trains until the validation logloss stops improving, then rewinds to the best round.
# Setting a high ceiling (1000) means we never cut training short artificially.
N_ESTIMATORS_MAX    = 1000   # ceiling — early stopping will fire well before this
EARLY_STOPPING_RNDS = 30     # stop after 30 rounds with no val-logloss improvement

# Numeric XGBoost — 19 features, lighter regularization ok
PARAM_GRID_NUMERIC = {
    "max_depth":        [3, 4, 5],
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma":            [0, 0.1, 0.3],
    "reg_alpha":        [0, 0.1, 1.0],
    "reg_lambda":       [1.0, 2.0, 5.0],
}

# BERT XGBoost — 1536 features, high-dimensional → heavy regularization to prevent overfit
PARAM_GRID_BERT = {
    "max_depth":        [2, 3, 4],          # shallower trees for high-dim input
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.6, 0.7, 0.8],    # more aggressive subsampling
    "colsample_bytree": [0.3, 0.5, 0.7],    # only sample a fraction of 1536 features per tree
    "min_child_weight": [3, 5, 10],
    "gamma":            [0.1, 0.3, 0.5],
    "reg_alpha":        [0.1, 1.0, 5.0],
    "reg_lambda":       [1.0, 5.0, 10.0],
}


def list_files(base_dir):
    base = Path(base_dir)
    if not base.exists():
        raise FileNotFoundError(f"Training directory not found: {base_dir}")
    paths, labels = [], []
    for label_dir in sorted(base.iterdir()):
        if not label_dir.is_dir():
            continue
        label = 1 if label_dir.name.lower().startswith("phish") else 0
        for f in sorted(list(label_dir.rglob("*.html")) + list(label_dir.rglob("*.htm"))):
            paths.append(str(f))
            labels.append(label)
    return paths, labels


def read_file(path):
    try:
        with open(path, "r", encoding="utf-8", errors="strict") as f:
            return f.read()
    except Exception:
        with open(path, "r", encoding="latin-1", errors="ignore") as f:
            return f.read()


def tune_xgb(X_tr, y_tr, X_val, y_val, param_grid, label="", scale_pos_weight=1.0):
    """
    Find the best XGBoost hyperparameters using 5-fold stratified CV × 20 random
    iterations, then refit the best combo on X_tr with early stopping on the
    explicit X_val set.

    X_val / y_val are the dedicated validation split passed in from main() —
    they are never seen during the hyperparameter search, only used for the
    final refit early-stopping evaluation and the overfit-gap report.

    scale_pos_weight = count(negatives) / count(positives) in y_tr — tells
    XGBoost to penalise false negatives more heavily than false positives,
    correcting for the class imbalance in the dataset.
    """
    best_params = None
    best_score  = -1.0

    cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    param_keys = list(param_grid.keys())
    sampled = [
        {k: rng.choice(param_grid[k]).item() for k in param_keys}
        for _ in range(20)
    ]

    print(f"[train] {label}: searching 20 param combos × 5 folds  "
          f"(scale_pos_weight={scale_pos_weight:.2f})…")
    for combo in tqdm(sampled, desc=f"{label} search"):
        fold_aucs = []
        for fi_tr, fi_val in cv.split(X_tr, y_tr):
            Xf_tr, Xf_val = X_tr[fi_tr], X_tr[fi_val]
            yf_tr, yf_val = y_tr[fi_tr], y_tr[fi_val]

            m = xgb.XGBClassifier(
                **combo,
                n_estimators=N_ESTIMATORS_MAX,
                early_stopping_rounds=EARLY_STOPPING_RNDS,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_SEED,
            )
            m.fit(Xf_tr, yf_tr, eval_set=[(Xf_val, yf_val)], verbose=False)
            fold_aucs.append(roc_auc_score(yf_val, m.predict_proba(Xf_val)[:, 1]))

        mean_auc = float(np.mean(fold_aucs))
        if mean_auc > best_score:
            best_score  = mean_auc
            best_params = combo

    print(f"[train] {label} best CV AUC={best_score:.4f}  params: {best_params}")

    # ── Refit on full X_tr, early-stop on the dedicated val set ──────────────
    best_model = xgb.XGBClassifier(
        **best_params,
        n_estimators=N_ESTIMATORS_MAX,
        early_stopping_rounds=EARLY_STOPPING_RNDS,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
    )
    best_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    stopped_at = best_model.best_iteration + 1
    train_auc  = roc_auc_score(y_tr,  best_model.predict_proba(X_tr)[:, 1])
    val_auc    = roc_auc_score(y_val, best_model.predict_proba(X_val)[:, 1])
    gap        = train_auc - val_auc
    print(f"[train] {label} stopped at tree {stopped_at}  "
          f"train AUC={train_auc:.4f}  val AUC={val_auc:.4f}  gap={gap:.4f}"
          + (" ⚠ overfit risk" if gap > 0.03 else " ✓"))

    return best_model, best_params


def main(training_root, fast=False):
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # ── 1. Scan + read files (always needed) ─────────────────────────────────
    print(f"[train] Scanning {training_root}...")
    paths, labels = list_files(training_root)
    y = np.array(labels)
    print(f"[train] {len(paths)} files — {y.sum()} phishing, {(y==0).sum()} benign")

    print("[train] Reading HTML files...")
    raw = [read_file(p) for p in tqdm(paths, desc="Reading")]

    # ── 1b. Extract HTML numeric features (always re-extracted) ──────────────
    # Fast — adding new HTML features only costs this step, not re-embedding
    print("[train] Extracting HTML features...")
    extractor    = HTMLFeatureExtractor()
    df_num, extras = extractor.transform(raw, urls=paths)
    vis          = extras["visible_texts"]
    tags         = extras["tag_sequences"]
    numeric_cols = list(df_num.columns)
    Xn           = df_num[numeric_cols].fillna(0).values.astype(np.float32)
    print(f"[train] {len(numeric_cols)} numeric features extracted")

    # ── 1c. Load or compute RoBERTa embeddings ────────────────────────────────
    if fast and EMB_CACHE.exists():
        print("[train] Loading cached embeddings (--fast, skipping RoBERTa)...")
        cache    = np.load(EMB_CACHE, allow_pickle=True)
        emb_bert = cache["emb_bert"]
        assert len(emb_bert) == len(y), (
            f"Embedding cache has {len(emb_bert)} rows but dataset has {len(y)}. "
            "Delete artifacts/emb_cache.npz and re-run without --fast."
        )
        print(f"[train] Embeddings loaded — {emb_bert.shape}")
    else:
        tokenizer, transformer, device = get_transformer()

        print("[train] Embedding visible text...")
        emb_vis = embed_texts(tokenizer, transformer, device, vis,  desc="Visible text")

        print("[train] Embedding tag sequences...")
        emb_tag = embed_texts(tokenizer, transformer, device, tags, desc="Tag sequences")

        emb_bert = np.hstack([emb_vis, emb_tag]).astype(np.float32)  # (N, 1536)

        np.savez_compressed(EMB_CACHE, emb_bert=emb_bert)
        print(f"[train] Embeddings cached → {EMB_CACHE}")

    print(f"[train] Feature shapes — numeric: {Xn.shape}, bert: {emb_bert.shape}")

    # ── 2. Three-way split: 40% train / 30% val / 30% test ──────────────────
    # Val is a dedicated held-out set used only for early stopping and the
    # overfit-gap report — never seen during the hyperparameter search.
    # Test is touched exactly once at the very end to report generalisation.
    idx = np.arange(len(y))

    # First cut: 40% train, 60% temp
    tr_idx, temp_idx = train_test_split(idx, test_size=0.60, stratify=y,
                                        random_state=RANDOM_SEED)
    # Second cut: split temp 50/50 → 30% val, 30% test
    val_idx, te_idx = train_test_split(temp_idx, test_size=0.50, stratify=y[temp_idx],
                                       random_state=RANDOM_SEED)

    Xn_tr,  Xn_val,  Xn_te  = Xn[tr_idx],       Xn[val_idx],       Xn[te_idx]
    Xb_tr,  Xb_val,  Xb_te  = emb_bert[tr_idx],  emb_bert[val_idx], emb_bert[te_idx]
    y_tr,   y_val,   y_te   = y[tr_idx],          y[val_idx],        y[te_idx]

    print(f"[train] Split — train={len(y_tr)} ({y_tr.sum()} phish), "
          f"val={len(y_val)} ({y_val.sum()} phish), "
          f"test={len(y_te)} ({y_te.sum()} phish)")

    # ── 2b. Class weight ─────────────────────────────────────────────────────
    # scale_pos_weight = negatives / positives in the training set.
    # Makes the model penalise a missed phishing site (false negative) roughly
    # as much as a false alarm, correcting for the 60/40 class imbalance.
    neg_count = int((y_tr == 0).sum())
    pos_count = int((y_tr == 1).sum())
    spw = neg_count / pos_count
    print(f"[train] Class weight — scale_pos_weight={spw:.3f} "
          f"({neg_count} safe / {pos_count} phish)")

    # ── 3. Tune base models ──────────────────────────────────────────────────
    print("[train] Tuning Voter A — XGBoost on numeric features...")
    xgb_a, params_a = tune_xgb(Xn_tr, y_tr, Xn_val, y_val,
                                PARAM_GRID_NUMERIC, "Voter A (numeric)",
                                scale_pos_weight=spw)

    print("[train] Tuning Voter B — XGBoost on BERT embeddings...")
    xgb_b, params_b = tune_xgb(Xb_tr, y_tr, Xb_val, y_val,
                                PARAM_GRID_BERT, "Voter B (BERT)",
                                scale_pos_weight=spw)

    # ── 4. OOF stacking — build meta-features without leakage ───────────────
    print("[train] Generating out-of-fold predictions for meta-learner...")
    kf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    oof_meta = np.zeros((len(y_tr), 2), dtype=np.float32)

    for fold, (fi_tr, fi_val) in enumerate(kf.split(Xn_tr, y_tr)):
        fold_spw = float((y_tr[fi_tr] == 0).sum()) / max(float((y_tr[fi_tr] == 1).sum()), 1)
        tmp_a = xgb.XGBClassifier(
            **params_a,
            n_estimators=N_ESTIMATORS_MAX,
            early_stopping_rounds=EARLY_STOPPING_RNDS,
            eval_metric="logloss",
            scale_pos_weight=fold_spw,
            random_state=RANDOM_SEED,
        )
        tmp_b = xgb.XGBClassifier(
            **params_b,
            n_estimators=N_ESTIMATORS_MAX,
            early_stopping_rounds=EARLY_STOPPING_RNDS,
            eval_metric="logloss",
            scale_pos_weight=fold_spw,
            random_state=RANDOM_SEED,
        )

        tmp_a.fit(Xn_tr[fi_tr], y_tr[fi_tr],
                  eval_set=[(Xn_tr[fi_val], y_tr[fi_val])], verbose=False)
        tmp_b.fit(Xb_tr[fi_tr], y_tr[fi_tr],
                  eval_set=[(Xb_tr[fi_val], y_tr[fi_val])], verbose=False)

        oof_meta[fi_val, 0] = tmp_a.predict_proba(Xn_tr[fi_val])[:, 1]
        oof_meta[fi_val, 1] = tmp_b.predict_proba(Xb_tr[fi_val])[:, 1]
        print(f"[train]   fold {fold+1}/5 — "
              f"Voter A stopped @{tmp_a.best_iteration+1} trees, "
              f"Voter B stopped @{tmp_b.best_iteration+1} trees")

    # ── 5. Train meta XGBoost on OOF predictions ────────────────────────────
    # Meta input = [P_xgb, P_bert] from the OOF predictions.
    # Use the dedicated val set OOF predictions for early stopping.
    print("[train] Training Meta XGBoost on [P_xgb, P_bert]...")

    # Generate val set predictions from the final base models (not OOF — those
    # were produced by fold models; use the tuned finals for the val set)
    val_meta = np.column_stack([
        xgb_a.predict_proba(Xn_val)[:, 1],
        xgb_b.predict_proba(Xb_val)[:, 1],
    ]).astype(np.float32)

    meta_spw = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1)
    xgb_meta = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS_MAX,
        max_depth=2,
        learning_rate=0.01,
        subsample=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        eval_metric="logloss",
        early_stopping_rounds=EARLY_STOPPING_RNDS,
        scale_pos_weight=meta_spw,
        random_state=RANDOM_SEED,
    )
    xgb_meta.fit(oof_meta, y_tr, eval_set=[(val_meta, y_val)], verbose=False)
    meta_train_auc = roc_auc_score(y_tr,  xgb_meta.predict_proba(oof_meta)[:, 1])
    meta_val_auc   = roc_auc_score(y_val, xgb_meta.predict_proba(val_meta)[:, 1])
    print(f"[train] Meta train AUC={meta_train_auc:.4f}  val AUC={meta_val_auc:.4f}"
          + (" ⚠ overfit risk" if meta_train_auc - meta_val_auc > 0.03 else " ✓"))

    # ── 6. Evaluate on held-out test set (touched once, final number) ────────
    te_a    = xgb_a.predict_proba(Xn_te)[:, 1]
    te_b    = xgb_b.predict_proba(Xb_te)[:, 1]
    meta_te = np.column_stack([te_a, te_b]).astype(np.float32)

    y_proba_final = xgb_meta.predict_proba(meta_te)[:, 1]
    auc      = roc_auc_score(y_te, y_proba_final)
    auc_xgb  = roc_auc_score(y_te, te_a)
    auc_bert = roc_auc_score(y_te, te_b)

    gap_final = meta_train_auc - auc
    print(f"[train] Test AUC — Voter A (numeric): {auc_xgb:.4f}  "
          f"Voter B (BERT): {auc_bert:.4f}  Meta blend: {auc:.4f}")
    print(f"[train] Generalisation gap (meta train→test): {gap_final:.4f}"
          + (" ⚠ possible overfit" if gap_final > 0.03 else " ✓ generalising well"))

    # ── 7. Save artifacts ────────────────────────────────────────────────────
    model_path = ARTIFACTS_DIR / "model.joblib"
    bundle = {
        "xgb_numeric":      xgb_a,
        "xgb_bert":         xgb_b,
        "xgb_meta":         xgb_meta,
        "numeric_columns":  numeric_cols,
        "transformer_name": "xlm-roberta-base",
    }
    joblib.dump(bundle, model_path)
    print(f"[train] Model saved → {model_path}")

    summary = {
        "n_files":         int(len(y)),
        "n_phishing":      int(y.sum()),
        "n_benign":        int((y == 0).sum()),
        "auc_xgb_numeric": round(float(auc_xgb), 4),
        "auc_bert":        round(float(auc_bert), 4),
        "auc_meta":        round(float(auc), 4),
        "params_numeric":  params_a,
        "params_bert":     params_b,
        "numeric_columns": numeric_cols,
    }
    with open(ARTIFACTS_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[train] Done — Meta AUC={auc:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 train.py data/training [--fast]")
        sys.exit(1)
    fast_mode = "--fast" in sys.argv
    main(sys.argv[1], fast=fast_mode)
