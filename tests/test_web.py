"""
Integration Tests for Flask Web Application
Tests HTTP routes, upload validation, security boundaries, and model inference integration.
"""

import io
import sqlite3
from contextlib import closing

import pytest
from PIL import Image
from app import create_app


@pytest.fixture
def client():
    """Configures test client for Flask application."""
    app = create_app({"TESTING": True, "SECRET_KEY": "test-key"})
    with app.test_client() as client:
        yield client


@pytest.fixture
def synthetic_jpeg_bytes():
    """Generates an in-memory synthetic JPEG image for testing uploads."""
    img = Image.new("RGB", (320, 320), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


@pytest.fixture
def community_client(tmp_path):
    """Uses isolated storage for community photo submissions."""
    app = create_app({
        "TESTING": True,
        "SECRET_KEY": "test-key",
        "COMMUNITY_UPLOAD_FOLDER": str(tmp_path / "community"),
        "COMMUNITY_DATABASE": str(tmp_path / "community.sqlite"),
    })
    with app.test_client() as client:
        yield client


class TestFlaskWebRoutes:
    """Test suite for web application endpoints and security controls."""

    def test_homepage_status(self, client):
        """GET / must return HTTP 200 and render campus cleanliness title."""
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "Campus Cleanliness" in html
        assert "YOLOv8n" in html
        assert "dropzone" in html

    def test_health_check_endpoint(self, client):
        """GET /health must return JSON with status healthy."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"
        assert "active_weights" in data

    def test_metrics_api_endpoint(self, client):
        """GET /api/metrics must return JSON containing mAP or status."""
        resp = client.get("/api/metrics")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "metrics" in data
        assert "map_50" in data["metrics"]

    def test_community_photo_is_saved_and_shown_in_gallery(
        self, community_client, synthetic_jpeg_bytes, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(
            "src.web.routes.process_image_analysis",
            lambda **kwargs: {
                "score_data": {
                    "score": 82,
                    "rating_label": "Good",
                    "area_ratio": 0.03,
                },
                "detection_count": 1,
                "annotated_image_url": "results/test-annotated.jpg",
            },
        )
        response = community_client.post(
            "/submit",
            data={"community_image": (synthetic_jpeg_bytes, "campus.jpg")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert b"Your photo was posted and analyzed" in response.data
        assert b"Cleanliness score: 82/100" in response.data
        assert b"Campus Community Gallery" in response.data
        assert b"community_" in response.data

        stored_photos = list((tmp_path / "community").iterdir())
        assert len(stored_photos) == 1
        assert stored_photos[0].read_bytes().startswith(b"\xff\xd8")

        with closing(sqlite3.connect(tmp_path / "community.sqlite")) as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM community_submissions"
            ).fetchone()[0] == 1

        gallery = community_client.get("/")
        assert gallery.status_code == 200
        assert b"community_" in gallery.data
        assert b"Cleanliness: 82/100" in gallery.data
        assert b"1 litter item(s) detected" in gallery.data
        assert b"test-annotated.jpg" in gallery.data
        image_url = next(
            line.split(b'"')[1].decode()
            for line in gallery.data.splitlines()
            if b"/community/" in line and b"<img" in line
        )
        stored_image = community_client.get(image_url)
        assert stored_image.status_code == 200
        assert stored_image.data.startswith(b"\xff\xd8")

    def test_community_photo_rejects_non_image_extension(self, community_client):
        response = community_client.post(
            "/submit",
            data={"community_image": (io.BytesIO(b"not an image"), "notes.txt")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )
        assert response.status_code == 200
        assert b"Unsupported file type" in response.data

    def test_community_photo_remains_posted_and_can_be_analyzed_later(
        self, community_client, synthetic_jpeg_bytes, tmp_path, monkeypatch
    ):
        def fail_analysis(**kwargs):
            raise ModuleNotFoundError("missing model package", name="ultralytics")

        monkeypatch.setattr("src.web.routes.process_image_analysis", fail_analysis)
        response = community_client.post(
            "/submit",
            data={"community_image": (synthetic_jpeg_bytes, "campus.jpg")},
            content_type="multipart/form-data",
            follow_redirects=True,
        )

        assert response.status_code == 200
        assert b"Your photo was posted" in response.data
        assert b"Analyze Photo" in response.data
        assert len(list((tmp_path / "community").iterdir())) == 1
        with closing(sqlite3.connect(tmp_path / "community.sqlite")) as connection:
            assert connection.execute(
                "SELECT COUNT(*) FROM community_submissions"
            ).fetchone()[0] == 1

        def successful_analysis(**kwargs):
            return {
                "score_data": {"score": 91, "rating_label": "Excellent"},
                "detection_count": 0,
                "annotated_image_url": "results/retry.jpg",
            }

        monkeypatch.setattr("src.web.routes.process_image_analysis", successful_analysis)
        with closing(sqlite3.connect(tmp_path / "community.sqlite")) as connection:
            filename = connection.execute(
                "SELECT filename FROM community_submissions"
            ).fetchone()[0]
        retry = community_client.post(
            f"/community/{filename}/analyze",
            follow_redirects=True,
        )
        assert retry.status_code == 200
        assert b"Analysis complete: cleanliness score 91/100" in retry.data
        assert b"Cleanliness: 91/100" in retry.data

    def test_community_gallery_migrates_existing_database(self, tmp_path):
        database_path = tmp_path / "community.sqlite"
        with closing(sqlite3.connect(database_path)) as connection:
            connection.execute(
                "CREATE TABLE community_submissions ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "filename TEXT NOT NULL UNIQUE, submitted_at TEXT NOT NULL)"
            )

        app = create_app({
            "TESTING": True,
            "COMMUNITY_UPLOAD_FOLDER": str(tmp_path / "community"),
            "COMMUNITY_DATABASE": str(database_path),
        })
        assert app.test_client().get("/").status_code == 200
        with closing(sqlite3.connect(database_path)) as connection:
            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(community_submissions)")
            }
        assert "analysis_json" in columns

    def test_invalid_file_extension_rejected(self, client):
        """Uploading a non-image file (.txt) must be rejected with flash error and redirect."""
        fake_txt = (io.BytesIO(b"malicious script or invalid file"), "test.txt")
        resp = client.post(
            "/analyze",
            data={"image": fake_txt},
            content_type="multipart/form-data",
            follow_redirects=True
        )
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "Unsupported file type" in html or "Invalid" in html

    def test_empty_upload_submission_redirects(self, client):
        """Submitting form with no file selected must redirect and flash an error."""
        resp = client.post("/analyze", data={}, follow_redirects=True)
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "Please choose an image file" in html

    def test_valid_image_upload_performs_real_inference(self, client, synthetic_jpeg_bytes):
        """Uploading a valid JPEG image must execute inference and render results page."""
        data = {
            "image": (synthetic_jpeg_bytes, "campus_sample.jpg"),
            "confidence": "0.25"
        }
        resp = client.post(
            "/analyze",
            data=data,
            content_type="multipart/form-data"
        )
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "Cleanliness Analysis Report" in html
        assert "Cleanliness Score" in html or "Detailed Litter Inventory" in html
        assert "Annotated Detection" in html

    def test_sample_image_analysis_via_post(self, client):
        """POST /analyze with sample_name parameter processes pre-existing campus sample."""
        data = {
            "sample_name": "campus_corridor_clean.jpg",
            "confidence": "0.25"
        }
        resp = client.post("/analyze", data=data)
        assert resp.status_code == 200
        html = resp.get_data(as_text=True)
        assert "Cleanliness Analysis Report" in html
        assert "100" in html  # Clean corridor gets 100/100
