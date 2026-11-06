"""
Campus Cleanliness Scoring System - Flask Application Entry Point
Ramdeobaba University, Nagpur | Deep Learning Lab Capstone
Team: Arya Mohod, Darsh Kadu, Hargun Singh Jaggi, Ayaan Ahmed
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.web.routes import web_bp


def create_app(config_override: dict = None) -> Flask:
    """Application factory for the Campus Cleanliness Scoring System."""
    app = Flask(
        __name__,
        template_folder=str(PROJECT_ROOT / "templates"),
        static_folder=str(PROJECT_ROOT / "static"),
    )

    # Default application configurations
    app.config["PROJECT_ROOT"] = str(PROJECT_ROOT)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "campus-cleanliness-secret-dev-2026")
    app.config["UPLOAD_FOLDER"] = str(PROJECT_ROOT / "static" / "uploads")
    app.config["RESULTS_FOLDER"] = str(PROJECT_ROOT / "static" / "results")
    app.config["COMMUNITY_UPLOAD_FOLDER"] = str(PROJECT_ROOT / "static" / "community")
    app.config["COMMUNITY_DATABASE"] = str(PROJECT_ROOT / "data" / "community_submissions.sqlite")
    app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_CONTENT_LENGTH", 16 * 1024 * 1024))  # 16 MB

    if config_override:
        app.config.update(config_override)

    # Ensure upload and results directories exist
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["RESULTS_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["COMMUNITY_UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["COMMUNITY_DATABASE"]).parent.mkdir(parents=True, exist_ok=True)

    # Register blueprints
    app.register_blueprint(web_bp)

    # Custom Error Handlers
    @app.errorhandler(413)
    def request_entity_too_large(error):
        return render_template("index.html", error_message="File size exceeds maximum 16 MB limit."), 413

    @app.errorhandler(404)
    def not_found(error):
        return render_template("index.html", error_message="Requested page not found."), 404

    @app.errorhandler(500)
    def internal_error(error):
        return render_template("index.html", error_message="An internal system error occurred. Please retry."), 500

    return app


app = create_app()

if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"

    print("=" * 70)
    print(" Campus Cleanliness Scoring System - Web Server Starting")
    print(" Ramdeobaba University, Nagpur | Deep Learning Lab Capstone")
    print(f" URL: http://{host}:{port}")
    print(" Development server running. Press Ctrl+C to terminate.")
    print("=" * 70)

    app.run(host=host, port=port, debug=debug)
