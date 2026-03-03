import logging
from pathlib import Path

from flask import Flask, jsonify, request

from data_prep import build_feature_matrix, load_feature_matrix_if_exists
from pipeline import run_pipeline
from training import train_models_from_feature_matrix

app = Flask(__name__)
logger = logging.getLogger("qblock.startup")
_STARTUP_STATUS = {
    "ready": False,
    "feature_matrix": {"ready": False, "message": "not_started"},
    "training": {"ready": False, "message": "not_started"},
}


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


def _as_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def run_startup_bootstrap_and_training() -> None:
    global _STARTUP_STATUS

    logger.info("Startup bootstrap: checking cached feature matrix...")
    matrix_csv_path = None

    try:
        cached = load_feature_matrix_if_exists()
        if cached is not None:
            matrix_csv_path = str(Path(__file__).parent / "artifacts" / "feature_matrix.csv")
            _STARTUP_STATUS["feature_matrix"] = {
                "ready": True,
                "message": f"loaded_cached_matrix rows={len(cached)}",
            }
            logger.info("Startup bootstrap: loaded cached matrix with %s rows.", len(cached))
    except Exception as exc:
        logger.warning("Startup bootstrap: cached matrix invalid; rebuilding. reason=%s", exc)
        _STARTUP_STATUS["feature_matrix"] = {
            "ready": False,
            "message": f"cached_matrix_invalid rebuilding: {exc}",
        }

    if matrix_csv_path is None:
        logger.info("Startup bootstrap: building feature matrix from dataset...")
        try:
            build_result = build_feature_matrix(force_rebuild=True, use_existing=False)
            matrix_csv_path = build_result["csv_path"]
            _STARTUP_STATUS["feature_matrix"] = {
                "ready": True,
                "message": f"rebuilt_matrix rows={build_result['rows']}",
            }
            logger.info(
                "Startup bootstrap: rebuilt feature matrix rows=%s path=%s",
                build_result["rows"],
                matrix_csv_path,
            )
        except Exception as exc:
            _STARTUP_STATUS["feature_matrix"] = {
                "ready": False,
                "message": f"bootstrap_build_failed: {exc}",
            }
            _STARTUP_STATUS["training"] = {
                "ready": False,
                "message": "skipped_training_due_to_matrix_failure",
            }
            _STARTUP_STATUS["ready"] = False
            logger.exception("Startup bootstrap failed during matrix build.")
            return

    logger.info("Startup bootstrap: training stub models from matrix...")
    try:
        train_result = train_models_from_feature_matrix(csv_path=matrix_csv_path)
        _STARTUP_STATUS["training"] = {
            "ready": True,
            "message": f"trained rows={train_result['rows']} features={train_result['feature_count']}",
        }
        _STARTUP_STATUS["ready"] = True
        logger.info(
            "Startup bootstrap: training complete rows=%s features=%s summary=%s",
            train_result["rows"],
            train_result["feature_count"],
            train_result["training_summary_path"],
        )
    except Exception as exc:
        _STARTUP_STATUS["training"] = {
            "ready": False,
            "message": f"training_failed: {exc}",
        }
        _STARTUP_STATUS["ready"] = False
        logger.exception("Startup bootstrap failed during training.")


@app.route("/api/scan", methods=["POST", "OPTIONS"])
def scan_url():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    url = str(payload.get("url", "")).strip()

    if not url:
        return jsonify({"error": "Missing URL"}), 400

    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"

    result = run_pipeline(url)
    return jsonify(result)


@app.route("/api/build-feature-matrix", methods=["POST", "OPTIONS"])
def api_build_feature_matrix():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    max_rows = payload.get("max_rows")
    force_rebuild = _as_bool(payload.get("force_rebuild"), default=False)

    if max_rows is not None:
        try:
            max_rows = int(max_rows)
            if max_rows <= 0:
                max_rows = None
        except ValueError:
            return jsonify({"error": "max_rows must be an integer"}), 400

    matrix_result = build_feature_matrix(max_rows=max_rows, force_rebuild=force_rebuild, use_existing=True)
    train_result = train_models_from_feature_matrix(matrix_result["csv_path"])

    return jsonify(
        {
            "feature_matrix": matrix_result,
            "training": train_result,
        }
    )


@app.route("/api/train-models", methods=["POST", "OPTIONS"])
def api_train_models():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    csv_path = payload.get("csv_path")

    result = train_models_from_feature_matrix(csv_path=csv_path)
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "qrcode-phishing-blocker-backend",
            "artifacts_dir": str(Path(__file__).parent / "artifacts"),
            "startup": _STARTUP_STATUS,
        }
    )


if __name__ == "__main__":
    (Path(__file__).parent / "artifacts").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    # With Flask reloader, run bootstrap only in the active serving process.
    import os

    if not app.debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        run_startup_bootstrap_and_training()
    app.run(host="0.0.0.0", port=5001, debug=True)
