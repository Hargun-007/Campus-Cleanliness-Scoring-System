"""
Campus Cleanliness Scoring System - Flask Routes
Defines endpoints for image upload, real-time prediction, results display, and health metrics.
"""

import json
import logging
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime
from pathlib import Path
from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from src.web.services import process_image_analysis, validate_uploaded_file

logger = logging.getLogger("web_routes")

web_bp = Blueprint("web", __name__)


def _project_path(*parts: str) -> Path:
    return Path(current_app.config["PROJECT_ROOT"]).joinpath(*parts)


def _ensure_community_table(connection):
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS community_submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL UNIQUE,
            submitted_at TEXT NOT NULL,
            analysis_json TEXT
        )
        """
    )
    columns = {
        row[1] for row in connection.execute("PRAGMA table_info(community_submissions)")
    }
    if "analysis_json" not in columns:
        connection.execute("ALTER TABLE community_submissions ADD COLUMN analysis_json TEXT")


def _community_submissions():
    database = Path(current_app.config["COMMUNITY_DATABASE"])
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database)) as connection:
        with connection:
            _ensure_community_table(connection)
            rows = connection.execute(
                "SELECT filename, submitted_at, analysis_json "
                "FROM community_submissions ORDER BY id DESC"
            ).fetchall()
    return [
        {
            "filename": filename,
            "submitted_at": submitted_at,
            "analysis": json.loads(analysis_json) if analysis_json else None,
        }
        for filename, submitted_at, analysis_json in rows
    ]


def _analyze_community_submission(filename):
    database = Path(current_app.config["COMMUNITY_DATABASE"])
    with closing(sqlite3.connect(database)) as connection:
        with connection:
            _ensure_community_table(connection)
            submission = connection.execute(
                "SELECT filename FROM community_submissions WHERE filename = ?",
                (filename,),
            ).fetchone()
    if submission is None:
        abort(404)

    image_path = Path(current_app.config["COMMUNITY_UPLOAD_FOLDER"]) / submission[0]
    if not image_path.is_file():
        raise FileNotFoundError(f"Community photo is missing: {image_path}")

    result = process_image_analysis(
        image_path_input=str(image_path),
        upload_folder=current_app.config["COMMUNITY_UPLOAD_FOLDER"],
        results_folder=current_app.config["RESULTS_FOLDER"],
    )
    analysis = {
        "score_data": result["score_data"],
        "detection_count": result["detection_count"],
        "annotated_image_url": result["annotated_image_url"],
    }

    with closing(sqlite3.connect(database)) as connection:
        with connection:
            connection.execute(
                "UPDATE community_submissions SET analysis_json = ? WHERE filename = ?",
                (json.dumps(analysis), filename),
            )
    return analysis


@web_bp.route("/", methods=["GET"])
def index():
    """Renders the homepage with upload interface, sample gallery, and project info."""
    # Discover available sample campus photos for quick testing
    campus_dir = _project_path("data", "campus_test")
    samples = []
    if campus_dir.exists():
        for p in campus_dir.iterdir():
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                samples.append({
                    "name": p.name,
                    "label": p.stem.replace("campus_", "").replace("_", " ").title()
                })

    return render_template(
        "index.html",
        samples=samples,
        community_submissions=_community_submissions(),
    )


@web_bp.route("/submit", methods=["POST"])
def submit_community_photo():
    """Publishes a validated campus photo and analyzes it when the model is available."""
    uploaded_file = request.files.get("community_image")
    if not uploaded_file or not uploaded_file.filename:
        flash("Choose a campus photo before submitting.", "error")
        return redirect(url_for("web.index"))

    is_valid, error_message = validate_uploaded_file(uploaded_file)
    if not is_valid:
        flash(error_message, "error")
        return redirect(url_for("web.index"))

    extension = uploaded_file.filename.rsplit(".", 1)[1].lower()
    filename = f"community_{uuid.uuid4().hex}.{extension}"
    upload_folder = Path(current_app.config["COMMUNITY_UPLOAD_FOLDER"])
    upload_folder.mkdir(parents=True, exist_ok=True)
    saved_path = upload_folder / filename
    uploaded_file.save(saved_path)

    try:
        database = Path(current_app.config["COMMUNITY_DATABASE"])
        database.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(database)) as connection:
            with connection:
                _ensure_community_table(connection)
                connection.execute(
                    "INSERT INTO community_submissions "
                    "(filename, submitted_at, analysis_json) VALUES (?, ?, ?)",
                    (
                        filename,
                        datetime.now().astimezone().strftime("%Y-%m-%d %H:%M"),
                        None,
                    ),
                )
    except sqlite3.Error:
        saved_path.unlink(missing_ok=True)
        logger.exception("Could not save community photo metadata")
        flash("Could not save your photo to the gallery. Please try again.", "error")
        return redirect(url_for("web.index"))

    try:
        analysis = _analyze_community_submission(filename)
    except Exception as error:
        logger.exception("Community photo was posted, but analysis failed")
        if isinstance(error, ModuleNotFoundError):
            flash(
                f"Your photo was posted. Analysis needs the missing dependency '{error.name}'. "
                "Install project dependencies, then choose Analyze on the gallery photo.",
                "warning",
            )
        else:
            flash(
                "Your photo was posted, but analysis could not run. "
                "Choose Analyze on the gallery photo to try again.",
                "warning",
            )
        return redirect(url_for("web.index"))

    flash(
        f"Your photo was posted and analyzed. Cleanliness score: "
        f"{analysis['score_data']['score']}/100; "
        f"{analysis['detection_count']} litter item(s) detected.",
        "success",
    )
    return redirect(url_for("web.index"))


@web_bp.route("/community/<filename>", methods=["GET"])
def community_photo(filename):
    """Serves a submitted photo from the configured community storage folder."""
    return send_from_directory(current_app.config["COMMUNITY_UPLOAD_FOLDER"], filename)


@web_bp.route("/community/<filename>/analyze", methods=["POST"])
def analyze_community_photo(filename):
    """Analyzes or retries analysis for a photo that is already in the gallery."""
    try:
        analysis = _analyze_community_submission(filename)
    except Exception as error:
        logger.exception("Could not analyze community photo %s", filename)
        if isinstance(error, ModuleNotFoundError):
            flash(
                f"Analysis needs the missing dependency '{error.name}'. "
                "Install project dependencies and try again.",
                "error",
            )
        else:
            flash("Could not analyze this photo. Please try again later.", "error")
        return redirect(url_for("web.index"))

    flash(
        f"Analysis complete: cleanliness score {analysis['score_data']['score']}/100; "
        f"{analysis['detection_count']} litter item(s) detected.",
        "success",
    )
    return redirect(url_for("web.index"))


@web_bp.route("/analyze", methods=["POST"])
def analyze():
    """Handles image upload and runs real litter detection and cleanliness scoring."""
    file = request.files.get("image")
    sample_name = request.form.get("sample_name")

    conf_str = request.form.get("confidence", "0.25")
    try:
        conf_val = float(conf_str)
        conf_val = min(0.95, max(0.05, conf_val))
    except ValueError:
        conf_val = 0.25

    try:
        if file and file.filename != "":
            results = process_image_analysis(
                file_storage=file,
                upload_folder=current_app.config["UPLOAD_FOLDER"],
                results_folder=current_app.config["RESULTS_FOLDER"],
                conf_threshold=conf_val
            )
        elif sample_name:
            sample_path = _project_path("data", "campus_test", sample_name)
            if not sample_path.exists():
                flash("Selected sample photograph does not exist.", "error")
                return redirect(url_for("web.index"))
            results = process_image_analysis(
                image_path_input=str(sample_path),
                upload_folder=current_app.config["UPLOAD_FOLDER"],
                results_folder=current_app.config["RESULTS_FOLDER"],
                conf_threshold=conf_val
            )
        else:
            flash("Please choose an image file to analyze or click a sample photograph.", "error")
            return redirect(url_for("web.index"))

        # Check if caller requested JSON response (e.g. AJAX or API client)
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.args.get("format") == "json":
            return jsonify(results)

        return render_template("results.html", result=results)

    except ValueError as ve:
        logger.warning(f"Validation error: {ve}")
        flash(str(ve), "error")
        return redirect(url_for("web.index"))
    except ModuleNotFoundError as e:
        logger.error("An image analysis dependency is missing: %s", e.name, exc_info=True)
        flash(
            f"Image analysis requires the missing dependency '{e.name}'. "
            "Install the project dependencies with: python -m pip install -r requirements.txt",
            "error",
        )
        return redirect(url_for("web.index"))
    except Exception as e:
        logger.error(f"Inference error during analysis: {e}", exc_info=True)
        flash("An error occurred during model analysis. Please verify image integrity.", "error")
        return redirect(url_for("web.index"))


@web_bp.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Returns official evaluation benchmark metrics."""
    metrics_file = _project_path("reports", "evaluation_metrics_test.json")
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"status": "no_metrics_found", "message": "Run evaluate.py first."}), 404


@web_bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint indicating model readiness and environment status."""
    model_path = Path(os.getenv("MODEL_PATH", str(_project_path("models", "best.pt"))))
    baseline_path = Path(os.getenv("FALLBACK_MODEL_PATH", str(_project_path("models", "yolov8n.pt"))))
    if not model_path.is_absolute():
        model_path = _project_path(str(model_path))
    if not baseline_path.is_absolute():
        baseline_path = _project_path(str(baseline_path))
    model_exists = model_path.exists()
    baseline_exists = baseline_path.exists()
    return jsonify({
        "status": "healthy",
        "fine_tuned_model_loaded": model_exists,
        "baseline_model_available": baseline_exists,
        "active_weights": str(model_path if model_exists else baseline_path)
    })
