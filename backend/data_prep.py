import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import polars as pl
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor

from feature_extractor import FEATURE_COLUMNS, extract_url_features

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
FEATURE_MATRIX_CSV = ARTIFACTS_DIR / "feature_matrix.csv"
FEATURE_MATRIX_PARTIAL_CSV = ARTIFACTS_DIR / "feature_matrix.partial.csv"
FEATURE_MATRIX_PARQUET = ARTIFACTS_DIR / "feature_matrix.parquet"
FEATURE_MATRIX_META = ARTIFACTS_DIR / "feature_matrix_meta.json"

DATASET_URL = "hf://datasets/Mitake/PhishingURLsANDBenignURLs/PhishBenignDataset.csv"


def _pick_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def _normalize_classification(value) -> int:
    if value is None:
        return -1

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return -1
        return int(value)

    text = str(value).strip().lower()
    if text in {"phishing", "malicious", "fraud", "scam", "1", "true", "yes"}:
        return 1
    if text in {"benign", "legitimate", "safe", "0", "false", "no"}:
        return 0

    try:
        return int(float(text))
    except (TypeError, ValueError):
        return -1


def _extract_row_task(task) -> Optional[Dict]:
    url, classification = task
    try:
        features = extract_url_features(url)
    except Exception:
        return None
    return {
        "url": url,
        **features,
        "classification": classification,
    }


def _flush_batch(batch: List[Dict], partial_path: Path, header_written: bool) -> None:
    pd.DataFrame(batch).to_csv(
        partial_path,
        mode="a" if header_written else "w",
        header=not header_written,
        index=False,
    )


def _collect_with_progress(iterator, total: int, partial_path: Optional[Path] = None) -> list:
    collected = []
    pending_batch: List[Dict] = []
    processed = 0
    header_written = False

    for result in iterator:
        processed += 1
        if result is not None:
            collected.append(result)
            if partial_path is not None:
                pending_batch.append(result)

        if processed % 10_000 == 0:
            if partial_path is not None and pending_batch:
                _flush_batch(pending_batch, partial_path, header_written)
                header_written = True
                pending_batch = []
            print(
                f"[data_prep] {processed:,}/{total:,} rows processed "
                f"({len(collected):,} valid) — partial saved to {partial_path.name if partial_path else 'n/a'}",
                flush=True,
            )

    # Flush any remainder under 10,000
    if partial_path is not None and pending_batch:
        _flush_batch(pending_batch, partial_path, header_written)

    skipped = total - len(collected)
    print(
        f"[data_prep] {total:,}/{total:,} complete — {len(collected):,} valid, {skipped:,} skipped (malformed).",
        flush=True,
    )
    return collected


def _parallel_extract_rows(tasks, partial_path: Optional[Path] = None) -> list:
    if not tasks:
        return []

    total = len(tasks)
    max_workers = max(1, os.cpu_count() or 1)

    if total == 1 or max_workers == 1:
        return _collect_with_progress(
            (_extract_row_task(task) for task in tasks), total, partial_path
        )

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            return _collect_with_progress(
                executor.map(_extract_row_task, tasks, chunksize=256), total, partial_path
            )
    except (PermissionError, OSError):
        # Fallback for restricted environments where process semaphores are blocked.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return _collect_with_progress(
                executor.map(_extract_row_task, tasks), total, partial_path
            )


def load_feature_matrix_if_exists() -> Optional[pd.DataFrame]:
    if not FEATURE_MATRIX_CSV.exists():
        return None

    df = pd.read_csv(FEATURE_MATRIX_CSV)
    _validate_feature_matrix(df)
    return df


def build_feature_matrix(max_rows: Optional[int] = None, force_rebuild: bool = False, use_existing: bool = True) -> Dict[str, str]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if use_existing and not force_rebuild and FEATURE_MATRIX_CSV.exists():
        existing_df = pd.read_csv(FEATURE_MATRIX_CSV)
        _validate_feature_matrix(existing_df)
        return _save_metadata(
            df=existing_df,
            source="existing",
            max_rows=max_rows,
            loaded_existing=True,
            parquet_error="",
            parquet_saved=FEATURE_MATRIX_PARQUET.exists(),
        )

    # Remove any leftover partial file from a previous failed run.
    FEATURE_MATRIX_PARTIAL_CSV.unlink(missing_ok=True)

    raw = pl.read_csv(DATASET_URL)
    cols = raw.columns

    url_col = _pick_column(cols, ["url", "URL", "link", "text", "domain"])
    label_col = _pick_column(cols, ["label", "target", "class", "is_phishing", "Label"])

    if url_col is None:
        raise ValueError(f"Could not find a URL column in dataset. Columns: {cols}")

    if max_rows is not None:
        raw = raw.head(max_rows)

    tasks = []
    for row in raw.iter_rows(named=True):
        url = str(row.get(url_col) or "").strip()
        if not url:
            continue
        classification = _normalize_classification(row.get(label_col) if label_col is not None else None)
        tasks.append((url, classification))

    rows = _parallel_extract_rows(tasks, partial_path=FEATURE_MATRIX_PARTIAL_CSV)
    df = pd.DataFrame(rows)
    _validate_feature_matrix(df)

    # Promote the partial file to the final path and clean up.
    df.to_csv(FEATURE_MATRIX_CSV, index=False)
    FEATURE_MATRIX_PARTIAL_CSV.unlink(missing_ok=True)

    parquet_saved = False
    parquet_error = ""
    try:
        df.to_parquet(FEATURE_MATRIX_PARQUET, index=False)
        parquet_saved = True
    except Exception as exc:
        parquet_error = str(exc)

    return _save_metadata(
        df=df,
        source=DATASET_URL,
        max_rows=max_rows,
        loaded_existing=False,
        parquet_error=parquet_error,
        parquet_saved=parquet_saved,
    )


def _save_metadata(
    df: pd.DataFrame,
    source: str,
    max_rows: Optional[int],
    loaded_existing: bool,
    parquet_error: str,
    parquet_saved: bool,
) -> Dict[str, str]:
    metadata = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(df),
        "columns": len(df.columns),
        "csv_path": str(FEATURE_MATRIX_CSV),
        "parquet_saved": parquet_saved,
        "parquet_path": str(FEATURE_MATRIX_PARQUET) if parquet_saved else "",
        "parquet_error": parquet_error,
        "max_rows": max_rows,
        "loaded_existing": loaded_existing,
        "source": source,
        "feature_columns": FEATURE_COLUMNS,
    }

    with open(FEATURE_MATRIX_META, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return {
        "rows": str(metadata["rows"]),
        "columns": str(metadata["columns"]),
        "csv_path": metadata["csv_path"],
        "meta_path": str(FEATURE_MATRIX_META),
        "parquet_saved": str(metadata["parquet_saved"]),
        "parquet_path": metadata["parquet_path"],
        "parquet_error": metadata["parquet_error"],
        "loaded_existing": str(metadata["loaded_existing"]),
        "source": metadata["source"],
    }


def _validate_feature_matrix(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Feature matrix is empty; no rows were extracted from dataset URLs.")

    required_non_feature = {"url", "classification"}
    missing_base = [c for c in required_non_feature if c not in df.columns]
    if missing_base:
        raise ValueError(f"Missing required base columns in matrix: {missing_base}")

    missing_features = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing expected feature columns: {missing_features}")

    allowed = {"url", "classification", *FEATURE_COLUMNS}
    extra_features = [c for c in df.columns if c not in allowed]
    if extra_features:
        raise ValueError(f"Unexpected columns in feature matrix: {extra_features}")

    feature_frame = df[FEATURE_COLUMNS]
    if feature_frame.isna().any().any():
        na_cols = feature_frame.columns[feature_frame.isna().any()].tolist()
        raise ValueError(f"NaN values found in feature columns: {na_cols}")

    feature_values = feature_frame.to_numpy(dtype=float)
    if not np.isfinite(feature_values).all():
        raise ValueError("Non-finite values found in feature matrix (inf or -inf).")


if __name__ == "__main__":
    result = build_feature_matrix()
    print(result)
