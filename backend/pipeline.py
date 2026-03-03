from dataclasses import dataclass
from typing import Dict, Tuple

import requests

from feature_extractor import extract_url_features


@dataclass
class BranchResult:
    prediction: str
    score: float
    details: Dict


def logistic_regression_stub(feature_matrix: Dict[str, float]) -> BranchResult:
    # Stub by requirement: always classify as phishing.
    return BranchResult(
        prediction="phishing",
        score=1.0,
        details={"model": "LogReg_STUB", "feature_count": len(feature_matrix)},
    )


def distilbert_stub(url: str, webpage_text: str) -> BranchResult:
    # Stub by requirement: always classify as phishing.
    return BranchResult(
        prediction="phishing",
        score=1.0,
        details={
            "model": "DistilBERT_STUB",
            "url_length": len(url),
            "webpage_text_chars": len(webpage_text),
        },
    )


def xgboost_blend_stub(logreg_result: BranchResult, bert_result: BranchResult) -> Dict:
    # Stub by requirement: always classify as phishing.
    return {
        "final_prediction": "phishing",
        "final_score": 1.0,
        "blender": "XGBoost_STUB",
        "inputs": {
            "logreg_prediction": logreg_result.prediction,
            "logreg_score": logreg_result.score,
            "bert_prediction": bert_result.prediction,
            "bert_score": bert_result.score,
        },
    }


def fetch_webpage_text(url: str, timeout: int = 5) -> Tuple[str, str]:
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "QRCodePhishingBlocker/1.0"})
        response.raise_for_status()
        text = response.text[:20000]
        return text, "ok"
    except Exception as exc:
        return "", f"fetch_error: {exc}"


def run_pipeline(url: str) -> Dict:
    # QRCode scan -> URL (input to this function)
    # split(a): feature extraction matrix -> LogReg(feature matrix)
    feature_matrix = extract_url_features(url)
    logreg_result = logistic_regression_stub(feature_matrix)

    # split(b): DistilBERT(url and curl webpage)
    webpage_text, fetch_status = fetch_webpage_text(url)
    bert_result = distilbert_stub(url, webpage_text)

    # blend(): XGBoost decision
    blended = xgboost_blend_stub(logreg_result, bert_result)

    return {
        "url": url,
        "split_a": {
            "feature_matrix": feature_matrix,
            "logreg": {
                "prediction": logreg_result.prediction,
                "score": logreg_result.score,
                "details": logreg_result.details,
            },
        },
        "split_b": {
            "fetch_status": fetch_status,
            "distilbert": {
                "prediction": bert_result.prediction,
                "score": bert_result.score,
                "details": bert_result.details,
            },
        },
        "blend": blended,
    }
