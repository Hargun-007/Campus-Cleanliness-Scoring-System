"""
Campus Cleanliness Scoring System - Web Service Layer
Handles file uploads, security validation, inference execution, and lifecycle management.
"""

import logging
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple
from PIL import Image
from werkzeug.datastructures import FileStorage

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if TYPE_CHECKING:
    from src.model.predict import LitterDetector

logger = logging.getLogger("web_services")

# Global singleton detector
_DETECTOR_INSTANCE: Optional["LitterDetector"] = None

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_FILE_SIZE_BYTES = 16 * 1024 * 1024  # 16 MB


def get_detector() -> "LitterDetector":
    """Returns the singleton LitterDetector instance, initializing lazily."""
    global _DETECTOR_INSTANCE
    if _DETECTOR_INSTANCE is None:
        from src.model.predict import LitterDetector

        model_path = Path(os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "best.pt")))
        fallback_path = Path(os.getenv("FALLBACK_MODEL_PATH", str(PROJECT_ROOT / "models" / "yolov8n.pt")))
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        if not fallback_path.is_absolute():
            fallback_path = PROJECT_ROOT / fallback_path
        device = os.getenv("DEVICE", "cpu")
        conf = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
        iou = float(os.getenv("IOU_THRESHOLD", "0.45"))

        logger.info(f"Initializing singleton LitterDetector (model={model_path})...")
        _DETECTOR_INSTANCE = LitterDetector(
            model_path=str(model_path),
            fallback_path=str(fallback_path),
            device=device,
            conf_threshold=conf,
            iou_threshold=iou
        )
    return _DETECTOR_INSTANCE


def validate_uploaded_file(file: FileStorage) -> Tuple[bool, str]:
    """Validates that uploaded file has an allowed extension and valid image format."""
    if not file or not file.filename:
        return False, "No file was selected for upload."

    filename = file.filename.strip()
    if "." not in filename:
        return False, "File must have an extension (.jpg, .jpeg, .png, .webp)."

    ext = filename.rsplit(".", 1)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type '.{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}."

    # Validate image bytes
    try:
        file.seek(0)
        img = Image.open(file.stream)
        img.verify()
        file.seek(0)
    except Exception as e:
        return False, f"Corrupted or invalid image file: {e}"

    return True, ""


def process_image_analysis(
    file_storage: Optional[FileStorage] = None,
    image_path_input: Optional[str] = None,
    upload_folder: str = "static/uploads",
    results_folder: str = "static/results",
    conf_threshold: Optional[float] = None,
    iou_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """Saves image safely, runs litter detection, and computes cleanliness metrics."""
    up_dir = Path(upload_folder)
    res_dir = Path(results_folder)
    if not up_dir.is_absolute():
        up_dir = PROJECT_ROOT / up_dir
    if not res_dir.is_absolute():
        res_dir = PROJECT_ROOT / res_dir
    up_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    if file_storage:
        valid, err = validate_uploaded_file(file_storage)
        if not valid:
            raise ValueError(err)

        ext = file_storage.filename.rsplit(".", 1)[1].lower()
        unique_token = uuid.uuid4().hex[:12]
        saved_upload_name = f"upload_{unique_token}.{ext}"
        saved_upload_path = up_dir / saved_upload_name

        file_storage.save(str(saved_upload_path))
        target_path = saved_upload_path
        display_original = f"uploads/{saved_upload_name}"

    elif image_path_input:
        target_path = Path(image_path_input)
        if not target_path.is_absolute():
            target_path = PROJECT_ROOT / target_path
        if not target_path.exists():
            raise FileNotFoundError(f"Sample image not found: {image_path_input}")
        display_original = f"uploads/{target_path.name}"
        # Copy to uploads for web serving if not already there
        dest_copy = up_dir / target_path.name
        if not dest_copy.exists():
            import shutil
            shutil.copy2(target_path, dest_copy)
    else:
        raise ValueError("Either file_storage or image_path_input must be provided.")

    annotated_filename = f"annotated_{uuid.uuid4().hex[:12]}.jpg"

    # Run inference
    detector = get_detector()
    inference_result = detector.predict(
        image_input=str(target_path),
        output_dir=str(res_dir),
        output_filename=annotated_filename,
        conf=conf_threshold,
        iou=iou_threshold
    )

    annotated_web_path = f"results/{annotated_filename}"

    return {
        "original_image_url": display_original,
        "annotated_image_url": annotated_web_path,
        "score_data": inference_result["scoring"],
        "detections": inference_result["detections"],
        "detection_count": inference_result["detection_count"],
        "image_dimensions": inference_result["image_dimensions"],
        "model_metadata": inference_result["model_metadata"]
    }
