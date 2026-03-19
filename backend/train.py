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

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit, LeaveOneOut, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from helpers import list_files, read_file
from model import HTMLFeatureExtractor, embed_texts, get_transformer

RANDOM_SEED = 42
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
EMB_CACHE = ARTIFACTS_DIR / "emb_cache.npz"
MINHASH_CACHE = ARTIFACTS_DIR / "minhash_cache.pkl"
_MINHASH_NUM_PERM = 128

PREPROCESSING_VERSION = "v3"

N_ESTIMATORS_MAX = 1000
EARLY_STOPPING_ROUNDS = 100


PARAM_GRID_NUMERIC = {
    "max_depth": [3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [5, 10, 20],
    "gamma": [0.3, 0.5, 1.0],
    "reg_alpha": [0.1, 1.0, 5.0],
    "reg_lambda": [2.0, 5.0, 10.0],
}

LR_C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

DECISION_THRESHOLD = 0.50


def _model_stats(name: str, y_true, y_proba, threshold: float = DECISION_THRESHOLD) -> dict:
    """Print and return precision-level stats for one model on a held-out set."""
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


class _UnionFind:
    def __init__(self, n):
        self._p = list(range(n))

    def find(self, x):
        while self._p[x] != x:
            self._p[x] = self._p[self._p[x]]  # path halving
            x = self._p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._p[rb] = ra


def _compute_minhash(html: str) -> np.ndarray:
    try:
        soup = BeautifulSoup(html, "html.parser")
        body = soup.find("body")
        text = body.get_text() if body else soup.get_text()
    except Exception:
        text = html

    normalised = re.sub(r"\s+", "", text).lower()
    m = MinHash(num_perm=_MINHASH_NUM_PERM)
    for i in range(len(normalised) - 4):
        m.update(normalised[i : i + 5].encode("utf-8"))
    return m.hashvalues.copy()


def assign_groups(
    paths: list, labels: list, raw_html: list, sim_threshold: float = 0.98
):
    n = len(paths)
    print(f"[dedup] Assigning groups to {n} files…")

    uf = _UnionFind(n)

    md5_to_first: dict[str, int] = {}
    md5s: list[str] = []
    for i, html in enumerate(raw_html):
        h = hashlib.md5(html.encode("utf-8", errors="ignore")).hexdigest()
        md5s.append(h)
        if h in md5_to_first:
            uf.union(md5_to_first[h], i)
        else:
            md5_to_first[h] = i

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
        n_hit = n - len(miss_idx)
        print(
            f"[dedup]   Computing MinHash for {len(miss_idx)} files "
            f"({n_hit} served from cache)…"
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

    for cls in (0, 1):
        cls_idx = [i for i in range(n) if labels[i] == cls]
        if not cls_idx:
            continue
        lsh = MinHashLSH(threshold=sim_threshold, num_perm=_MINHASH_NUM_PERM)
        for i in cls_idx:
            mh = _mh(md5s[i])
            for match_key in lsh.query(mh):
                uf.union(i, int(match_key))
            lsh.insert(str(i), mh)

    roots = [uf.find(i) for i in range(n)]
    unique_roots = sorted(set(roots))
    root_to_gid = {r: idx for idx, r in enumerate(unique_roots)}
    group_ids = np.array([root_to_gid[r] for r in roots], dtype=np.int32)
    n_groups = len(unique_roots)
    print(
        f"[dedup] {n} files → {n_groups} groups "
        f"(threshold={sim_threshold:.0%}, all files kept)"
    )

    return paths, labels, raw_html, group_ids


def _eval_xgb_combo(combo, X_tr, y_tr, scale_pos_weight):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    fold_aucs = []
    fold_iters = []
    for fi_tr, fi_val in cv.split(X_tr, y_tr):
        fi_es_tr, fi_es_stop = train_test_split(
            fi_tr,
            test_size=0.15,
            stratify=y_tr[fi_tr],
            random_state=RANDOM_SEED,
        )
        m = xgb.XGBClassifier(
            **combo,
            n_estimators=N_ESTIMATORS_MAX,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            nthread=1,
            random_state=RANDOM_SEED,
        )
        m.fit(
            X_tr[fi_es_tr],
            y_tr[fi_es_tr],
            eval_set=[(X_tr[fi_es_stop], y_tr[fi_es_stop])],
            verbose=False,
        )
        fold_aucs.append(
            roc_auc_score(y_tr[fi_val], m.predict_proba(X_tr[fi_val])[:, 1])
        )
        fold_iters.append(m.best_iteration + 1)
    return float(np.mean(fold_aucs)), int(round(np.mean(fold_iters)))


def _oof_fold_worker(
    fold_idx,
    fi_sub_tr,
    fi_sub_val,
    fi_val,
    Xn_tr,
    Xb_tr,
    y_tr,
    params_a,
    best_C_b,
    fold_spw,
    xgb_nthread,
):
    tmp_a = xgb.XGBClassifier(
        **params_a,
        n_estimators=N_ESTIMATORS_MAX,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        eval_metric="logloss",
        scale_pos_weight=fold_spw,
        nthread=xgb_nthread,
        random_state=RANDOM_SEED,
    )
    tmp_a.fit(
        Xn_tr[fi_sub_tr],
        y_tr[fi_sub_tr],
        eval_set=[(Xn_tr[fi_sub_val], y_tr[fi_sub_val])],
        verbose=False,
    )

    tmp_b = LogisticRegression(
        C=best_C_b,
        l1_ratio=0,
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    tmp_b.fit(Xb_tr[fi_sub_tr], y_tr[fi_sub_tr])

    p_a = tmp_a.predict_proba(Xn_tr[fi_val])[:, 1]
    p_b = tmp_b.predict_proba(Xb_tr[fi_val])[:, 1]

    return fold_idx, fi_val, p_a, p_b, tmp_a.best_iteration + 1


# ─────────────────────────────────────────────────────────────────────────────
# Voter A tuning — XGBoost with random search + 5-fold CV
# ─────────────────────────────────────────────────────────────────────────────


def tune_xgb(X_tr, y_tr, param_grid, label="", scale_pos_weight=1.0):
    rng = np.random.default_rng(RANDOM_SEED)
    param_keys = list(param_grid.keys())
    sampled = [
        {k: rng.choice(param_grid[k]).item() for k in param_keys} for _ in range(20)
    ]

    print(
        f"[train] {label}: searching 20 param combos × 5 folds in parallel "
        f"(scale_pos_weight={scale_pos_weight:.2f})…"
    )
    results = joblib.Parallel(n_jobs=-1, prefer="threads")(
        joblib.delayed(_eval_xgb_combo)(combo, X_tr, y_tr, scale_pos_weight)
        for combo in tqdm(sampled, desc=f"{label} search")
    )

    aucs = [r[0] for r in results]
    iters = [r[1] for r in results]

    best_idx = int(np.argmax(aucs))
    best_score = float(aucs[best_idx])
    best_params = sampled[best_idx]
    best_n_trees = iters[best_idx]

    print(
        f"[train] {label} best CV AUC={best_score:.4f}  "
        f"n_estimators={best_n_trees}  params: {best_params}"
    )

    # Refit on full X_tr — no val set, no early stopping
    best_model = xgb.XGBClassifier(
        **best_params,
        n_estimators=best_n_trees,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
    )
    best_model.fit(X_tr, y_tr)
    train_auc = roc_auc_score(y_tr, best_model.predict_proba(X_tr)[:, 1])
    print(
        f"[train] {label} train AUC={train_auc:.4f}  "
        f"(n_estimators={best_n_trees} fixed from CV, val set untouched)"
    )

    return best_model, best_params, best_n_trees


def tune_lr(X_tr, y_tr, X_val, y_val, label=""):
    print(
        f"[train] {label}: fitting LogisticRegressionCV "
        f"({len(LR_C_GRID)} C values × 5-fold, n_jobs=-1, scoring=roc_auc)…"
    )

    cv_lr = LogisticRegressionCV(
        Cs=LR_C_GRID,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED),
        l1_ratios=(0,),
        solver="lbfgs",
        max_iter=2000,
        class_weight="balanced",
        scoring="roc_auc",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        refit=True,
        use_legacy_attributes=False,
    )
    cv_lr.fit(X_tr, y_tr)

    best_C = float(cv_lr.C_)
    cv_scores = np.mean(cv_lr.scores_, axis=0)
    best_cv_auc = float(cv_scores.max())
    print(f"[train] {label} best CV AUC={best_cv_auc:.4f}  C={best_C}")

    train_auc = roc_auc_score(y_tr, cv_lr.predict_proba(X_tr)[:, 1])
    val_auc = roc_auc_score(y_val, cv_lr.predict_proba(X_val)[:, 1])
    gap = train_auc - val_auc
    print(
        f"[train] {label} train AUC={train_auc:.4f}  val AUC={val_auc:.4f}  "
        f"gap={gap:.4f}" + (" ⚠ overfit risk" if gap > 0.03 else " ✓")
    )

    return cv_lr, best_C


def main(training_root, fast=False):
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    print(f"[train] Scanning {training_root}…")
    paths, labels = list_files(training_root)
    print(
        f"[train] {len(paths)} files found — "
        f"{sum(labels)} phishing, {sum(1 for lbl in labels if lbl == 0)} benign"
    )

    print("[train] Reading HTML files…")
    raw = joblib.Parallel(n_jobs=-1, backend="threading")(
        joblib.delayed(read_file)(p) for p in tqdm(paths, desc="Reading")
    )

    paths, labels, raw, group_ids = assign_groups(
        paths, labels, raw, sim_threshold=0.98
    )
    y = np.array(labels)
    print(
        f"[train] Dataset: {len(y)} files — {y.sum()} phishing, {(y == 0).sum()} benign"
    )

    print("[train] Extracting HTML features…")
    extractor = HTMLFeatureExtractor()
    df_num, extras = extractor.transform(raw)
    vis = extras["visible_texts"]
    struct_cores = extras["struct_cores"]
    numeric_cols = list(df_num.columns)
    Xn_all = df_num[numeric_cols].fillna(0).values.astype(np.float32)
    print(f"[train] {len(numeric_cols)} numeric features extracted")

    if fast and EMB_CACHE.exists():
        print("[train] Loading cached embeddings (--fast, skipping RoBERTa)…")
        cache = np.load(EMB_CACHE, allow_pickle=True)
        cached_version = str(cache.get("version", np.array("unknown")))
        if cached_version != PREPROCESSING_VERSION:
            print(
                f"[train] ERROR: Cached embeddings built with '{cached_version}', "
                f"current requires '{PREPROCESSING_VERSION}'.\n"
                f"        Delete {EMB_CACHE} and re-run without --fast."
            )
            sys.exit(1)
        emb_bert = cache["emb_bert"]
        assert len(emb_bert) == len(y), (
            f"Embedding cache has {len(emb_bert)} rows but dataset has {len(y)}. "
            "Delete artifacts/emb_cache.npz and re-run without --fast."
        )
        print(f"[train] Embeddings loaded — {emb_bert.shape}")
    else:
        tokenizer, transformer, device = get_transformer()
        print("[train] Embedding visible text + structural core…")
        n = len(vis)
        all_embs = embed_texts(
            tokenizer, transformer, device, vis + struct_cores, desc="Embedding"
        )
        emb_bert = np.hstack([all_embs[:n], all_embs[n:]]).astype(np.float32)
        np.savez_compressed(
            EMB_CACHE, emb_bert=emb_bert, version=np.array(PREPROCESSING_VERSION)
        )
        print(
            f"[train] Embeddings cached → {EMB_CACHE}  "
            f"[preprocessing={PREPROCESSING_VERSION}]"
        )

    print(f"[train] Feature shapes — numeric: {Xn_all.shape}, bert: {emb_bert.shape}")
    idx = np.arange(len(y))

    gss = GroupShuffleSplit(n_splits=1, test_size=0.60, random_state=RANDOM_SEED)
    tr_idx, temp_idx = next(gss.split(idx, y, group_ids))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_SEED)
    val_loc, te_loc = next(
        gss2.split(np.arange(len(temp_idx)), y[temp_idx], group_ids[temp_idx])
    )
    val_idx = temp_idx[val_loc]
    te_idx = temp_idx[te_loc]

    Xn_tr_raw = Xn_all[tr_idx]
    Xn_val_raw = Xn_all[val_idx]
    Xn_te_raw = Xn_all[te_idx]
    Xb_tr_raw = emb_bert[tr_idx]
    Xb_val_raw = emb_bert[val_idx]
    Xb_te_raw = emb_bert[te_idx]
    y_tr, y_val, y_te = y[tr_idx], y[val_idx], y[te_idx]

    print(
        f"[train] Split — train={len(y_tr)} ({y_tr.sum()} phish), "
        f"val={len(y_val)} ({y_val.sum()} phish), "
        f"test={len(y_te)} ({y_te.sum()} phish)"
    )

    neg_count = int((y_tr == 0).sum())
    pos_count = int((y_tr == 1).sum())
    spw = neg_count / pos_count
    print(
        f"[train] Class weight — scale_pos_weight={spw:.3f} "
        f"({neg_count} safe / {pos_count} phish)"
    )

    print("[train] Fitting StandardScaler on numeric training features…")
    scaler = StandardScaler()
    Xn_tr = scaler.fit_transform(Xn_tr_raw).astype(np.float32)
    Xn_val = scaler.transform(Xn_val_raw).astype(np.float32)
    Xn_te = scaler.transform(Xn_te_raw).astype(np.float32)

    print("[train] Fitting PCA(n_components=128) on BERT training embeddings…")
    pca = PCA(n_components=128, random_state=RANDOM_SEED)
    Xb_tr = pca.fit_transform(Xb_tr_raw).astype(np.float32)
    Xb_val = pca.transform(Xb_val_raw).astype(np.float32)
    Xb_te = pca.transform(Xb_te_raw).astype(np.float32)
    var_retained = float(pca.explained_variance_ratio_.sum())
    print(
        f"[train] PCA: {emb_bert.shape[1]} → 128 dims ({var_retained:.3%} variance retained)"
    )

    print("[train] Tuning Voter A — XGBoost on scaled numeric features…")
    xgb_a, params_a, n_trees_a = tune_xgb(
        Xn_tr,
        y_tr,
        PARAM_GRID_NUMERIC,
        "Voter A (numeric)",
        scale_pos_weight=spw,
    )

    print("[train] Tuning Voter B — LogisticRegression(L2) on PCA embeddings…")
    lr_b, best_C_b = tune_lr(
        Xb_tr,
        y_tr,
        Xb_val,
        y_val,
        label="Voter B (LR-BERT)",
    )

    print("[train] Generating out-of-fold predictions for meta-learner…")
    print(
        "[train]   OOF sub-split: 85% fold-train / 15% fold-early-stop "
        "(fi_sub_val carved from fi_tr, never from held-out val set)."
    )

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    fold_configs = []
    for fold, (fi_tr, fi_val) in enumerate(kf.split(Xn_tr, y_tr)):
        fi_sub_tr, fi_sub_val = train_test_split(
            fi_tr,
            test_size=0.15,
            stratify=y_tr[fi_tr],
            random_state=RANDOM_SEED + fold,
        )
        fold_spw = float((y_tr[fi_sub_tr] == 0).sum()) / max(
            float((y_tr[fi_sub_tr] == 1).sum()), 1
        )
        fold_configs.append((fold, fi_sub_tr, fi_sub_val, fi_val, fold_spw))

    n_cores = os.cpu_count() or 4
    xgb_nthread = max(1, n_cores // 5)
    print(
        f"[train]   Running 5 OOF folds in parallel "
        f"(xgb_nthread={xgb_nthread} per fold, {n_cores} cores total)…"
    )

    oof_results = joblib.Parallel(n_jobs=5, prefer="threads")(
        joblib.delayed(_oof_fold_worker)(
            fold,
            fi_sub_tr,
            fi_sub_val,
            fi_val,
            Xn_tr,
            Xb_tr,
            y_tr,
            params_a,
            best_C_b,
            fold_spw,
            xgb_nthread,
        )
        for fold, fi_sub_tr, fi_sub_val, fi_val, fold_spw in fold_configs
    )

    oof_meta = np.zeros((len(y_tr), 2), dtype=np.float32)
    for fold_idx, fi_val, p_a, p_b, best_iter in sorted(
        oof_results, key=lambda r: r[0]
    ):
        oof_meta[fi_val, 0] = p_a
        oof_meta[fi_val, 1] = p_b
        print(
            f"[train]   fold {fold_idx + 1}/5 — "
            f"Voter A stopped @{best_iter} trees  (oof={len(fi_val)})"
        )

    val_meta = np.column_stack(
        [
            xgb_a.predict_proba(Xn_val)[:, 1],
            lr_b.predict_proba(Xb_val)[:, 1],
        ]
    ).astype(np.float32)

    print("[train] Training LogisticRegression meta-learner on OOF [P_xgb, P_lr]…")
    meta_lr = LogisticRegression(
        C=1.0,
        l1_ratio=0,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )
    meta_lr.fit(oof_meta, y_tr)

    # LOO CV on meta features — 2D input so each LR fit is sub-millisecond.
    print("[train] Meta LR — running LOO CV on OOF meta features…")
    loo = LeaveOneOut()
    loo_meta_preds = np.zeros(len(y_tr))
    for _tr_idx, _te_idx in loo.split(oof_meta):
        _m = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=1000,
            class_weight="balanced", random_state=RANDOM_SEED,
        )
        _m.fit(oof_meta[_tr_idx], y_tr[_tr_idx])
        loo_meta_preds[_te_idx] = _m.predict_proba(oof_meta[_te_idx])[:, 1]
    meta_loo_auc = roc_auc_score(y_tr, loo_meta_preds)

    meta_val_auc = roc_auc_score(y_val, meta_lr.predict_proba(val_meta)[:, 1])
    print(
        f"[train] Meta LR LOO AUC={meta_loo_auc:.4f}  "
        f"val AUC={meta_val_auc:.4f}"
        + (" ⚠ overfit risk" if meta_loo_auc - meta_val_auc > 0.03 else " ✓")
    )
    meta_oof_auc = roc_auc_score(y_tr, meta_lr.predict_proba(oof_meta)[:, 1])

    lr_meta_coef = meta_lr.coef_[0].tolist()
    print(
        f"[train] Meta LR weights — P_xgb: {lr_meta_coef[0]:.3f}, "
        f"P_lr: {lr_meta_coef[1]:.3f}"
    )

    # ── Platt calibration on val set (val used exactly once) ─────────────────
    print("[train] Calibrating meta-model (Platt scaling) on val set…")
    _raw_val = meta_lr.predict_proba(val_meta)[:, 1].reshape(-1, 1)
    meta_calibrator = LogisticRegression(
        C=1e10,
        solver="lbfgs",
        max_iter=1000,
    ).fit(_raw_val, y_val)

    cal_val_auc = roc_auc_score(
        y_val,
        meta_calibrator.predict_proba(_raw_val)[:, 1],
    )

    # LOO CV on Platt calibrator — 1D input, val set size, essentially instant.
    print("[train] Platt calibrator — running LOO CV on val set…")
    loo_cal_preds = np.zeros(len(y_val))
    for _tr_idx, _te_idx in loo.split(_raw_val):
        _cal = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
        _cal.fit(_raw_val[_tr_idx], y_val[_tr_idx])
        loo_cal_preds[_te_idx] = _cal.predict_proba(_raw_val[_te_idx])[:, 1]
    cal_loo_auc = roc_auc_score(y_val, loo_cal_preds)

    print(
        f"[train] Calibrator LOO AUC={cal_loo_auc:.4f}  "
        f"fitted val AUC={cal_val_auc:.4f}"
        + (" ⚠ calibration unstable" if abs(cal_loo_auc - cal_val_auc) > 0.02 else " ✓")
    )

    te_a = xgb_a.predict_proba(Xn_te)[:, 1]
    te_b = lr_b.predict_proba(Xb_te)[:, 1]
    meta_te = np.column_stack([te_a, te_b]).astype(np.float32)

    _raw_te = meta_lr.predict_proba(meta_te)[:, 1].reshape(-1, 1)
    y_proba_final = meta_calibrator.predict_proba(_raw_te)[:, 1]
    auc = roc_auc_score(y_te, y_proba_final)
    auc_xgb = roc_auc_score(y_te, te_a)
    auc_lr = roc_auc_score(y_te, te_b)
    gap_final = cal_val_auc - auc

    print(f"\n[train] ══════════════════════════════════════════════════════")
    print(f"[train] Final Test Results  (n={len(y_te)}, threshold={DECISION_THRESHOLD})")
    print(f"[train] ══════════════════════════════════════════════════════")
    stats_a    = _model_stats("Voter A — XGBoost (numeric features)", y_te, te_a)
    stats_b    = _model_stats("Voter B — LR-BERT (PCA embeddings)",   y_te, te_b)
    stats_meta = _model_stats("Meta blend (Platt-calibrated)",        y_te, y_proba_final)
    print(
        f"\n[train]   Val AUC (calibrated):              {cal_val_auc:.4f}"
        f"\n[train]   Meta LR LOO AUC (train):           {meta_loo_auc:.4f}"
        f"\n[train]   Calibrator LOO AUC (val):          {cal_loo_auc:.4f}"
        f"\n[train]   Generalisation gap (val→test):     {gap_final:.4f}"
        + (" ⚠ possible overfit" if gap_final > 0.03 else " ✓ generalising well")
    )

    model_path = ARTIFACTS_DIR / "model.joblib"
    bundle = {
        # ── Base voters ───────────────────────────────────────────────────────
        "xgb_numeric": xgb_a,  # Voter A: XGBoost on scaled numerics
        "lr_bert": lr_b,  # Voter B: LogisticRegression on PCA embeddings
        # ── Meta layer ────────────────────────────────────────────────────────
        "meta_lr": meta_lr,  # LR meta-learner fit on OOF
        "meta_calibrator": meta_calibrator,  # Platt scaler fit on val
        # ── Preprocessing transforms (fit on train only) ──────────────────────
        "scaler": scaler,
        "pca": pca,
        # ── Metadata ─────────────────────────────────────────────────────────
        "numeric_columns": numeric_cols,
        "transformer_name": "xlm-roberta-base",
        "pca_n_components": 128,
        "lr_best_C": best_C_b,
        "meta_lr_coef": lr_meta_coef,
    }
    joblib.dump(bundle, model_path)
    print(f"[train] Model bundle saved → {model_path}")

    summary = {
        "n_files": int(len(y)),
        "n_phishing": int(y.sum()),
        "n_benign": int((y == 0).sum()),
        "decision_threshold": DECISION_THRESHOLD,
        "voter_a_numeric":   stats_a,
        "voter_b_lr_bert":   stats_b,
        "meta_calibrated":   stats_meta,
        "meta_oof_auc":      round(float(meta_oof_auc), 4),
        "meta_loo_auc":      round(float(meta_loo_auc), 4),
        "meta_val_auc":      round(float(meta_val_auc), 4),
        "cal_val_auc":       round(float(cal_val_auc),  4),
        "cal_loo_auc":       round(float(cal_loo_auc),  4),
        "generalisation_gap_val_test": round(float(gap_final), 4),
        "params_voter_a": params_a,
        "n_trees_voter_a": n_trees_a,
        "lr_best_C": best_C_b,
        "pca_n_components": 128,
        "pca_variance_retained": round(var_retained, 4),
        "meta_lr_coef": lr_meta_coef,
        "numeric_columns": numeric_cols,
    }
    with open(ARTIFACTS_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[train] Summary saved → {ARTIFACTS_DIR / 'training_summary.json'}")
    print(f"[train] Done — Calibrated Meta AUC={auc:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 train.py data/training [--fast]")
        sys.exit(1)
    fast_mode = "--fast" in sys.argv
    main(sys.argv[1], fast=fast_mode)
