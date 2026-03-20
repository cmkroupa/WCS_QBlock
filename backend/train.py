"""
train.py — QBlock phishing detector trainer (v2).

Architecture
────────────
  Voter A : XGBoost on scaled numeric HTML features
  Voter B : Fine-tuned RoBERTa on visible text + structural HTML core
  Meta    : LogisticRegression on [P_xgb, P_roberta] → Platt-calibrated score

Speed-ups
─────────
  • HTML reading              — joblib threading   (all cores, I/O bound)
  • HTML feature extraction   — joblib processes   (all cores, CPU bound)
  • MinHash deduplication     — persistent cache   (only new files recomputed)
  • Feature / text cache      — npz on disk        (--fast skips entirely)
  • PhishBERT checkpoint      — persistent .pt     (--fast reloads if split unchanged)
  • XGBoost                   — fixed proven hyperparams + early stopping (no grid search)

Usage
─────
  python3 train.py data/training          # full run
  python3 train.py data/training --fast   # skip feature extraction + BERT fine-tune if cached
"""

import hashlib
import json
import multiprocessing
import os
import pickle
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup
from datasketch import MinHash, MinHashLSH

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

import joblib
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from helpers import list_files, read_file
from features import HTMLFeatureExtractor
from phishbert import PhishBERTClassifier

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
FEAT_CACHE    = ARTIFACTS_DIR / "feat_cache.npz"
BERT_CLF_PATH = ARTIFACTS_DIR / "bert_classifier.pt"
BERT_CLF_META = ARTIFACTS_DIR / "bert_clf_meta.npz"
VOTER_A_PATH  = ARTIFACTS_DIR / "voter_a.joblib"
VOTER_A_META  = ARTIFACTS_DIR / "voter_a_meta.npz"
META_LR_PATH  = ARTIFACTS_DIR / "meta_lr.joblib"
META_LR_META  = ARTIFACTS_DIR / "meta_lr_meta.npz"
MINHASH_CACHE = ARTIFACTS_DIR / "minhash_cache.pkl"

_MINHASH_NUM_PERM    = 128
PREPROCESSING_VERSION = "v5"
DECISION_THRESHOLD    = 0.50

# ── Fixed XGBoost hyperparameters ─────────────────────────────────────────────
# Empirically strong defaults for tabular phishing detection.
# No random search needed — avoids 20×5-fold overhead on every run.
XGB_PARAMS = {
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "gamma":            0.3,
    "reg_alpha":        1.0,
    "reg_lambda":       5.0,
}
XGB_N_ESTIMATORS   = 500
XGB_EARLY_STOPPING = 50


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _xgb_gpu_kwargs() -> dict:
    """Return XGBoost GPU kwargs when CUDA is available, empty dict otherwise.
    Note: XGBoost does not support MPS — CPU is the correct fallback on Apple Silicon."""
    try:
        import torch
        if torch.cuda.is_available():
            return {"device": "cuda", "tree_method": "hist"}
    except Exception:
        pass
    return {}


def _model_stats(name: str, y_true, y_proba, threshold: float = DECISION_THRESHOLD) -> dict:
    """Print and return key metrics for one model on a held-out set."""
    auc  = roc_auc_score(y_true, y_proba)
    ap   = average_precision_score(y_true, y_proba)
    pred = (y_proba >= threshold).astype(int)
    prec = precision_score(y_true, pred, zero_division=0)
    rec  = recall_score(y_true, pred, zero_division=0)
    f1   = f1_score(y_true, pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    print(f"  {name}")
    print(f"    ROC-AUC={auc:.4f}  PR-AUC={ap:.4f}")
    print(f"    Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}  (@ threshold={threshold})")
    print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    return {
        "roc_auc":   round(float(auc),  4),
        "pr_auc":    round(float(ap),   4),
        "precision": round(float(prec), 4),
        "recall":    round(float(rec),  4),
        "f1":        round(float(f1),   4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MinHash deduplication  (near-duplicate grouping with persistent cache)
# ─────────────────────────────────────────────────────────────────────────────

class _UnionFind:
    def __init__(self, n):
        self._p = list(range(n))

    def find(self, x):
        while self._p[x] != x:
            self._p[x] = self._p[self._p[x]]   # path halving
            x = self._p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._p[rb] = ra


def _compute_minhash(html: str) -> np.ndarray:
    try:
        soup = BeautifulSoup(html, "lxml")
        body = soup.find("body")
        text = body.get_text() if body else soup.get_text()
    except Exception:
        text = html
    normalised = re.sub(r"\s+", "", text).lower()
    m = MinHash(num_perm=_MINHASH_NUM_PERM)
    for i in range(len(normalised) - 4):
        m.update(normalised[i:i + 5].encode("utf-8"))
    return m.hashvalues.copy()


def assign_groups(paths: list, labels: list, raw_html: list, sim_threshold: float = 0.98):
    """Group near-duplicate pages so they always land in the same split."""
    n = len(paths)
    print(f"[dedup] Assigning groups to {n} files…")
    uf = _UnionFind(n)

    # Pass 1: exact MD5 duplicates
    md5_to_first: dict[str, int] = {}
    md5s: list[str] = []
    for i, html in enumerate(raw_html):
        h = hashlib.md5(html.encode("utf-8", errors="ignore")).hexdigest()
        md5s.append(h)
        if h in md5_to_first:
            uf.union(md5_to_first[h], i)
        else:
            md5_to_first[h] = i

    # Pass 2: MinHash LSH near-duplicates (with persistent cache)
    mh_cache: dict[str, np.ndarray] = {}
    if MINHASH_CACHE.exists():
        try:
            with open(MINHASH_CACHE, "rb") as fh:
                mh_cache = pickle.load(fh)
            print(f"[dedup]   MinHash cache: {len(mh_cache)} entries loaded")
        except Exception as exc:
            print(f"[dedup]   MinHash cache unreadable ({exc}), rebuilding…")
            mh_cache = {}

    miss_idx = [i for i in range(n) if md5s[i] not in mh_cache]
    if miss_idx:
        print(
            f"[dedup]   Computing MinHash for {len(miss_idx)} files "
            f"({n - len(miss_idx)} served from cache)…"
        )
        new_hv = joblib.Parallel(n_jobs=-1, prefer="processes")(
            joblib.delayed(_compute_minhash)(raw_html[i]) for i in miss_idx
        )
        for i, hv in zip(miss_idx, new_hv):
            mh_cache[md5s[i]] = hv
        ARTIFACTS_DIR.mkdir(exist_ok=True)
        with open(MINHASH_CACHE, "wb") as fh:
            pickle.dump(mh_cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[dedup]   MinHash cache saved ({len(mh_cache)} entries total)")

    def _mh(md5: str) -> MinHash:
        m = MinHash(num_perm=_MINHASH_NUM_PERM)
        m.hashvalues = mh_cache[md5].copy()
        return m

    # Single cross-label LSH pass: a phishing clone of a benign page must land
    # in the same split regardless of its label, so we query/insert across the
    # full dataset rather than within each class separately.
    lsh = MinHashLSH(threshold=sim_threshold, num_perm=_MINHASH_NUM_PERM)
    for i in range(n):
        mh = _mh(md5s[i])
        for match_key in lsh.query(mh):
            uf.union(i, int(match_key))
        lsh.insert(str(i), mh)

    roots        = [uf.find(i) for i in range(n)]
    unique_roots = sorted(set(roots))
    root_to_gid  = {r: idx for idx, r in enumerate(unique_roots)}
    group_ids    = np.array([root_to_gid[r] for r in roots], dtype=np.int32)
    print(
        f"[dedup] {n} files → {len(unique_roots)} groups "
        f"(threshold={sim_threshold:.0%}, all files kept)"
    )
    return paths, labels, raw_html, group_ids


# ─────────────────────────────────────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main(training_root: str, fast: bool = False):
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    # ── 1. Discover files ─────────────────────────────────────────────────────
    print(f"[train] Scanning {training_root}…")
    paths, labels = list_files(training_root)
    print(
        f"[train] {len(paths)} files found — "
        f"{sum(labels)} phishing, {sum(1 for l in labels if l == 0)} benign"
    )

    # ── 2. Feature cache ──────────────────────────────────────────────────────
    feat_cache_valid = False
    if fast and FEAT_CACHE.exists():
        cache = np.load(FEAT_CACHE, allow_pickle=True)
        cached_version = str(cache.get("version", np.array("unknown")))
        if cached_version != PREPROCESSING_VERSION:
            print(
                f"[train] Cache version mismatch ('{cached_version}' vs "
                f"'{PREPROCESSING_VERSION}'). Delete {FEAT_CACHE} and re-run without --fast."
            )
            sys.exit(1)
        required = ("Xn_all", "numeric_cols", "group_ids", "vis", "struct_cores")
        if all(k in cache for k in required) and len(cache["Xn_all"]) == len(paths):
            print("[train] Loading feature cache (--fast, skipping HTML extraction)…")
            Xn_all       = cache["Xn_all"]
            numeric_cols = [str(c) for c in cache["numeric_cols"]]
            group_ids    = cache["group_ids"]
            vis          = list(cache["vis"])
            struct_cores = list(cache["struct_cores"])
            y            = np.array(labels)
            feat_cache_valid = True
            print(
                f"[train] Cache loaded — {len(y)} samples, "
                f"{len(numeric_cols)} numeric features"
            )
        else:
            print("[train] Feature cache stale — rebuilding…")

    if not feat_cache_valid:
        # Read files in parallel — threading is fastest for I/O
        print("[train] Reading HTML files…")
        raw = joblib.Parallel(n_jobs=-1, backend="threading")(
            joblib.delayed(read_file)(p) for p in tqdm(paths, desc="Reading")
        )

        paths, labels, raw, group_ids = assign_groups(paths, labels, raw, sim_threshold=0.98)
        y = np.array(labels)
        print(f"[train] Dataset: {len(y)} files — {y.sum()} phishing, {(y == 0).sum()} benign")

        # Extract numeric features + text strings — processes for CPU-bound parsing
        print("[train] Extracting HTML features…")
        extractor = HTMLFeatureExtractor()
        df_num, extras = extractor.transform(raw)
        vis          = extras["visible_texts"]
        struct_cores = extras["struct_cores"]
        del raw   # free 6 GB+ of raw HTML strings before model work starts
        numeric_cols = list(df_num.columns)
        Xn_all       = df_num[numeric_cols].fillna(0).values.astype(np.float32)
        print(f"[train] {len(numeric_cols)} numeric features extracted")

        np.savez_compressed(
            FEAT_CACHE,
            Xn_all=Xn_all,
            numeric_cols=np.array(numeric_cols),
            group_ids=group_ids,
            vis=np.array(vis, dtype=object),
            struct_cores=np.array(struct_cores, dtype=object),
            version=np.array(PREPROCESSING_VERSION),
        )
        print(f"[train] Feature cache saved → {FEAT_CACHE}  [v={PREPROCESSING_VERSION}]")

    # ── 3. Group-aware train / val / test split  (60 / 20 / 20) ──────────────
    # Groups ensure near-duplicate pages never straddle splits, preventing leakage.
    idx = np.arange(len(y))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=RANDOM_SEED)
    tr_idx, temp_idx = next(gss.split(idx, y, group_ids))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_SEED)
    val_loc, te_loc = next(
        gss2.split(np.arange(len(temp_idx)), y[temp_idx], group_ids[temp_idx])
    )
    val_idx = temp_idx[val_loc]
    te_idx  = temp_idx[te_loc]

    y_tr, y_val, y_te = y[tr_idx], y[val_idx], y[te_idx]
    print(
        f"[train] Split — train={len(y_tr)} ({y_tr.sum()} phish), "
        f"val={len(y_val)} ({y_val.sum()} phish), "
        f"test={len(y_te)} ({y_te.sum()} phish)"
    )

    # ── 4. Voter A: XGBoost on scaled numeric features ────────────────────────
    xgb_cached = False
    if fast and VOTER_A_PATH.exists() and VOTER_A_META.exists():
        meta_a = np.load(VOTER_A_META)
        if np.array_equal(meta_a.get("tr_idx", np.array([])), tr_idx):
            print("[train] Loading cached Voter A (--fast)…")
            bundle_a  = joblib.load(VOTER_A_PATH)
            xgb_a     = bundle_a["xgb_a"]
            scaler    = bundle_a["scaler"]
            n_trees   = int(bundle_a["n_trees"])
            Xn_val    = scaler.transform(Xn_all[val_idx]).astype(np.float32)
            Xn_te     = scaler.transform(Xn_all[te_idx]).astype(np.float32)
            xgb_cached = True

    if not xgb_cached:
        neg_count = int((y_tr == 0).sum())
        pos_count = int((y_tr == 1).sum())
        spw       = neg_count / pos_count
        print(f"[train] scale_pos_weight={spw:.3f}  ({neg_count} safe / {pos_count} phish)")

        scaler = StandardScaler()
        Xn_tr  = scaler.fit_transform(Xn_all[tr_idx]).astype(np.float32)
        Xn_val = scaler.transform(Xn_all[val_idx]).astype(np.float32)
        Xn_te  = scaler.transform(Xn_all[te_idx]).astype(np.float32)

        # Carve 15 % of train for XGBoost early stopping (isolated from meta-learner)
        es_tr_idx, es_val_idx = train_test_split(
            np.arange(len(y_tr)), test_size=0.15, stratify=y_tr, random_state=RANDOM_SEED
        )

        print("[train] Training Voter A — XGBoost…")
        xgb_a = xgb.XGBClassifier(
            **XGB_PARAMS,
            **_xgb_gpu_kwargs(),
            n_estimators=XGB_N_ESTIMATORS,
            early_stopping_rounds=XGB_EARLY_STOPPING,
            eval_metric="logloss",
            scale_pos_weight=spw,
            random_state=RANDOM_SEED,
        )
        xgb_a.fit(
            Xn_tr[es_tr_idx], y_tr[es_tr_idx],
            eval_set=[(Xn_tr[es_val_idx], y_tr[es_val_idx])],
            verbose=False,
        )
        n_trees = xgb_a.best_iteration + 1
        joblib.dump({"xgb_a": xgb_a, "scaler": scaler, "n_trees": n_trees}, VOTER_A_PATH)
        np.savez(VOTER_A_META, tr_idx=tr_idx)
        print(f"[train] Voter A saved → {VOTER_A_PATH}")

    xgb_val_auc = roc_auc_score(y_val, xgb_a.predict_proba(Xn_val)[:, 1])
    print(f"[train] Voter A — {n_trees} trees  val AUC={xgb_val_auc:.4f}")

    # ── 5. Voter B: Fine-tuned RoBERTa ────────────────────────────────────────
    # Carve 15 % of train for RoBERTa early stopping — same pattern as XGBoost.
    # combined_val / y_val are kept 100 % unseen until the meta-learner step.
    bert_es_tr_idx, bert_es_val_idx = train_test_split(
        np.arange(len(y_tr)), test_size=0.15, stratify=y_tr,
        random_state=RANDOM_SEED + 1,   # different seed to XGBoost's carve-out
    )

    # Fix 1: pass only visible text — struct_core moves to XGBoost numeric features.
    # This prevents malicious JS/HTML payloads from eating the token budget.
    combined_tr     = [vis[i] for i in tr_idx]
    combined_val    = [vis[i] for i in val_idx]
    combined_te     = [vis[i] for i in te_idx]
    combined_bert_es_tr  = [combined_tr[i] for i in bert_es_tr_idx]
    combined_bert_es_val = [combined_tr[i] for i in bert_es_val_idx]

    bert_cached = False
    if fast and BERT_CLF_PATH.exists() and BERT_CLF_META.exists():
        meta = np.load(BERT_CLF_META)
        if np.array_equal(meta.get("tr_idx", np.array([])), tr_idx):
            print("[train] Loading cached PhishBERT (--fast)…")
            bert_clf    = PhishBERTClassifier.load(BERT_CLF_PATH)
            bert_cached = True

    if not bert_cached:
        print("[train] Fine-tuning Voter B — RoBERTa…")
        bert_clf = PhishBERTClassifier(random_state=RANDOM_SEED)
        bert_clf.fit(
            combined_bert_es_tr,  y_tr[bert_es_tr_idx],
            val_texts=combined_bert_es_val, val_y=y_tr[bert_es_val_idx],
        )
        bert_clf.save(BERT_CLF_PATH)
        np.savez(BERT_CLF_META, tr_idx=tr_idx)
        print(f"[train] PhishBERT saved → {BERT_CLF_PATH}")

    bert_val_auc = roc_auc_score(y_val, bert_clf.predict_proba(combined_val)[:, 1])
    print(f"[train] Voter B val AUC={bert_val_auc:.4f}")

    # ── 6. Meta-learner: LR on val-set predictions ────────────────────────────
    # Both voters were trained exclusively on tr_idx → val predictions are unbiased.
    # Val predictions are always computed — needed for the generalisation gap report.
    P_xgb_val  = xgb_a.predict_proba(Xn_val)[:, 1]
    P_bert_val = bert_clf.predict_proba(combined_val)[:, 1]
    meta_val   = np.column_stack([P_xgb_val, P_bert_val]).astype(np.float32)

    meta_cached = False
    if fast and META_LR_PATH.exists() and META_LR_META.exists():
        meta_m = np.load(META_LR_META)
        if np.array_equal(meta_m.get("tr_idx", np.array([])), tr_idx):
            print("[train] Loading cached Meta LR (--fast)…")
            meta_lr     = joblib.load(META_LR_PATH)
            meta_cached = True

    if not meta_cached:
        print("[train] Training meta LogisticRegression on [P_xgb, P_roberta] from val set…")
        meta_lr = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=1000,
            class_weight="balanced", random_state=RANDOM_SEED,
        )
        meta_lr.fit(meta_val, y_val)
        joblib.dump(meta_lr, META_LR_PATH)
        np.savez(META_LR_META, tr_idx=tr_idx)
        print(f"[train] Meta LR saved → {META_LR_PATH}")

    all_fast = fast and xgb_cached and bert_cached and meta_cached
    if all_fast:
        print("[train] All models loaded from cache — jumping straight to test evaluation.")

    lr_coef = meta_lr.coef_[0].tolist()
    print(f"[train] Meta LR weights — P_xgb: {lr_coef[0]:.3f}, P_roberta: {lr_coef[1]:.3f}")

    meta_val_auc = roc_auc_score(y_val, meta_lr.predict_proba(meta_val)[:, 1])
    print(f"[train] Meta LR val AUC={meta_val_auc:.4f}")

    # ── 7. Final test-set evaluation ──────────────────────────────────────────
    # meta_lr (LogisticRegression) outputs natively calibrated probabilities —
    # no Platt scaling step needed.
    P_xgb_te  = xgb_a.predict_proba(Xn_te)[:, 1]
    P_bert_te = bert_clf.predict_proba(combined_te)[:, 1]
    meta_te   = np.column_stack([P_xgb_te, P_bert_te]).astype(np.float32)
    y_final   = meta_lr.predict_proba(meta_te)[:, 1]

    test_auc = roc_auc_score(y_te, y_final)
    gap      = meta_val_auc - test_auc

    print(f"\n[train] ══════════════════════════════════════════════════════")
    print(f"[train] Final Test Results  (n={len(y_te)}, threshold={DECISION_THRESHOLD})")
    print(f"[train] ══════════════════════════════════════════════════════")
    stats_a    = _model_stats("Voter A — XGBoost (numeric features)", y_te, P_xgb_te)
    stats_b    = _model_stats("Voter B — RoBERTa (fine-tuned)",       y_te, P_bert_te)
    stats_meta = _model_stats("Meta blend (LR)",                      y_te, y_final)
    print(
        f"\n[train]   Val AUC:                        {meta_val_auc:.4f}"
        f"\n[train]   Test AUC:                       {test_auc:.4f}"
        f"\n[train]   Generalisation gap (val→test):  {gap:.4f}"
        + (" ⚠ possible overfit" if gap > 0.03 else " ✓ generalising well")
    )

    # ── 8. Save model bundle + training summary ───────────────────────────────
    bundle = {
        # Voters
        "xgb_numeric":    xgb_a,
        "bert_clf_path":  str(BERT_CLF_PATH),
        # Meta layer
        "meta_lr":        meta_lr,
        # Preprocessing (fit on train only)
        "scaler":         scaler,
        # Metadata
        "numeric_columns":  numeric_cols,
        "transformer_name": "xlm-roberta-base",
        "voter_b_type":     "PhishBERTClassifier",
        "meta_lr_coef":     lr_coef,
    }
    model_path = ARTIFACTS_DIR / "model.joblib"
    joblib.dump(bundle, model_path)
    print(f"[train] Model bundle saved → {model_path}")

    summary = {
        "n_files":    int(len(y)),
        "n_phishing": int(y.sum()),
        "n_benign":   int((y == 0).sum()),
        "decision_threshold": DECISION_THRESHOLD,
        "voter_a":            stats_a,
        "voter_b":            stats_b,
        "meta_lr":            stats_meta,
        "meta_val_auc":       round(float(meta_val_auc), 4),
        "test_auc":           round(float(test_auc),     4),
        "generalisation_gap": round(float(gap),          4),
        "xgb_params":         XGB_PARAMS,
        "xgb_n_trees":        n_trees,
        "voter_b_type":       "PhishBERTClassifier",
        "bert_clf_path":      str(BERT_CLF_PATH),
        "meta_lr_coef":       lr_coef,
        "numeric_columns":    numeric_cols,
    }
    with open(ARTIFACTS_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[train] Summary saved → {ARTIFACTS_DIR / 'training_summary.json'}")
    print(f"[train] Done — Test AUC={test_auc:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 train.py data/training [--fast]")
        sys.exit(1)
    fast_mode = "--fast" in sys.argv
    main(sys.argv[1], fast=fast_mode)
