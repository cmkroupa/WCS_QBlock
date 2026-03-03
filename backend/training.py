import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from data_prep import FEATURE_MATRIX_CSV, build_feature_matrix, _validate_feature_matrix
from feature_extractor import FEATURE_COLUMNS

TRAINING_SUMMARY_PATH = Path(__file__).parent / "artifacts" / "stub_training_summary.json"


def train_models_from_feature_matrix(csv_path: Optional[str] = None) -> Dict[str, str]:
    matrix_path = Path(csv_path) if csv_path else FEATURE_MATRIX_CSV
    if not matrix_path.exists():
        build_feature_matrix(force_rebuild=True, use_existing=False)
        if not matrix_path.exists():
            raise FileNotFoundError(f"Feature matrix not found after build attempt: {matrix_path}")

    df = pd.read_csv(matrix_path)
    _validate_feature_matrix(df)

    y = df["classification"]
    X = df.drop(columns=["url", "classification"], errors="ignore")

    logreg_info = train_logreg_stub(X, y)
    distilbert_info = train_distilbert_stub(df["url"], y)
    xgboost_info = train_xgboost_stub(X, y)

    summary = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_path": str(matrix_path),
        "rows": int(len(df)),
        "feature_count": int(len(FEATURE_COLUMNS)),
        "dropped_columns": ["url"],
        "target_column": "classification",
        "models": {
            "logreg": logreg_info,
            "distilbert": distilbert_info,
            "xgboost_blender": xgboost_info,
        },
    }

    TRAINING_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {
        "status": "ok",
        "rows": str(summary["rows"]),
        "feature_count": str(summary["feature_count"]),
        "matrix_path": summary["matrix_path"],
        "training_summary_path": str(TRAINING_SUMMARY_PATH),
        "models": str(summary["models"]),
    }


def train_logreg_stub(X: pd.DataFrame, y: pd.Series) -> Dict:
    # Stub by requirement: training placeholder only.
    return {
        "model": "LogReg_STUB",
        "trained_rows": int(len(X)),
        "input_features": int(X.shape[1]),
        "class_distribution": y.value_counts(dropna=False).to_dict(),
    }


def train_distilbert_stub(urls: pd.Series, y: pd.Series) -> Dict:
    # Stub by requirement: training placeholder only.
    avg_url_length = float(urls.fillna("").astype(str).str.len().mean()) if len(urls) else 0.0
    return {
        "model": "DistilBERT_STUB",
        "trained_rows": int(len(urls)),
        "avg_url_length": avg_url_length,
        "class_distribution": y.value_counts(dropna=False).to_dict(),
    }


def train_xgboost_stub(X: pd.DataFrame, y: pd.Series) -> Dict:
    # Stub by requirement: training placeholder only.
    return {
        "model": "XGBoost_STUB",
        "trained_rows": int(len(X)),
        "input_features": int(X.shape[1]),
        "class_distribution": y.value_counts(dropna=False).to_dict(),
    }
