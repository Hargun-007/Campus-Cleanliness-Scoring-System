"""
Campus Cleanliness Scoring System - Model Inference Engine
Loads YOLOv8n detector, predicts litter bounding boxes, annotates images,
and calculates cleanliness scores.
"""

import argparse
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Set Ultralytics config dir to writable local path
os.environ["YOLO_CONFIG_DIR"] = str(PROJECT_ROOT / "runs" / ".ultralytics")

import cv2
import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scoring.cleanliness_score import calculate_cleanliness_score
from src.scoring.score_config import ScoringConfig, DEFAULT_SCORING_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("predict")


class LitterDetector:
    """Inference wrapper for YOLOv8 litter detection model."""

    def __init__(
        self,
        model_path: str = "models/best.pt",
        fallback_path: str = "models/yolov8n.pt",
        allow_fallback: bool = False,
        device: str = "cpu",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model_path = model_path
        self.is_fine_tuned = False

        if not os.path.exists(model_path):
            if not allow_fallback:
                raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
            if fallback_path and os.path.exists(fallback_path):
                logger.warning(
                    f"Fine-tuned model '{model_path}' not found! "
                    f"Falling back to baseline weights '{fallback_path}' (allow_fallback=True)."
                )
                target_model = fallback_path
                self.is_fine_tuned = False
            else:
                raise FileNotFoundError(
                    f"Neither target model '{model_path}' nor fallback '{fallback_path}' was found."
                )
        else:
            self.is_fine_tuned = True
            target_model = model_path

        logger.info(f"Loading YOLO detector from {target_model} on device={device}...")
        self.model = YOLO(target_model)
        self.loaded_model_path = target_model

        if self.is_fine_tuned and len(self.model.names) != 1:
            raise ValueError(
                f"Fine-tuned model must have exactly 1 class, found {len(self.model.names)}: {self.model.names}"
            )

    def validate_image(self, image_input: Union[str, Path, bytes, np.ndarray, Image.Image]) -> Tuple[bool, Optional[np.ndarray], str]:
        """Validates that input can be decoded into a BGR uint8 image array.

        Returns:
            Tuple of (is_valid, bgr_numpy_array, error_message).
        """
        try:
            if isinstance(image_input, (str, Path)):
                img_path = Path(image_input)
                if not img_path.exists():
                    return False, None, f"Image file not found: {img_path}"
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    return False, None, "Failed to decode image file format."
                return True, bgr, ""

            elif isinstance(image_input, bytes):
                nparr = np.frombuffer(image_input, np.uint8)
                bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if bgr is None:
                    return False, None, "Invalid image bytes, could not decode."
                return True, bgr, ""

            elif isinstance(image_input, Image.Image):
                img = ImageOps.exif_transpose(image_input).convert("RGB")
                bgr = np.array(img, dtype=np.uint8)[:, :, ::-1].copy()
                return True, bgr, ""

            elif isinstance(image_input, np.ndarray):
                if image_input.ndim != 3 or image_input.shape[2] != 3:
                    return False, None, f"Expected 3-channel image array, got shape {image_input.shape}"
                if image_input.dtype != np.uint8:
                    image_input = image_input.astype(np.uint8)
                return True, image_input, ""

            return False, None, f"Unsupported image input type: {type(image_input)}"
        except Exception as e:
            return False, None, f"Exception during image validation: {e}"

    def predict(
        self,
        image_input: Union[str, Path, bytes, np.ndarray, Image.Image],
        output_dir: Optional[str] = "static/results",
        output_filename: Optional[str] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        scoring_config: Optional[ScoringConfig] = None
    ) -> Dict[str, Any]:
        """Performs litter detection and calculates cleanliness score.

        Args:
            image_input: Image path, bytes, or numpy array.
            output_dir: Directory to save the annotated image.
            output_filename: Custom filename for annotated output image.
            conf: Override confidence threshold.
            iou: Override IoU NMS threshold.
            scoring_config: Optional custom ScoringConfig.

        Returns:
            Structured dictionary of detections, statistics, and score.
        """
        valid, bgr_img, err_msg = self.validate_image(image_input)
        if not valid:
            raise ValueError(f"Invalid image: {err_msg}")

        height, width, _ = bgr_img.shape
        c_thresh = conf if conf is not None else self.conf_threshold
        i_thresh = iou if iou is not None else self.iou_threshold

        # Run inference through Ultralytics using BGR image directly
        results = self.model.predict(
            source=bgr_img,
            conf=c_thresh,
            iou=i_thresh,
            device=self.device,
            verbose=False
        )

        first_res = results[0]
        boxes = first_res.boxes

        detections: List[Dict[str, Any]] = []
        annotated_bgr = bgr_img.copy()

        if boxes is not None and len(boxes) > 0:
            xyxy_arr = boxes.xyxy.cpu().numpy()
            conf_arr = boxes.conf.cpu().numpy()
            cls_arr = boxes.cls.cpu().numpy().astype(int)

            for i in range(len(xyxy_arr)):
                x1, y1, x2, y2 = xyxy_arr[i]
                c = float(conf_arr[i])
                raw_cls = int(cls_arr[i])

                # Map class to 'litter' for single-class project
                class_name = "litter"
                w = max(0.0, float(x2 - x1))
                h = max(0.0, float(y2 - y1))
                area = w * h

                det_entry = {
                    "box": [round(float(x1), 1), round(float(y1), 1), round(float(x2), 1), round(float(y2), 1)],
                    "confidence": round(c, 4),
                    "class_id": 0,
                    "class_name": class_name,
                    "area": round(area, 1),
                    "relative_area": round(area / float(height * width), 5)
                }
                detections.append(det_entry)

                # Draw bounding box and label
                # Box in high-visibility Emerald Green / Warning Orange depending on confidence
                box_color = (40, 167, 69) if c >= 0.50 else (0, 140, 255)  # BGR
                cv2.rectangle(
                    annotated_bgr,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    box_color,
                    2
                )

                label_text = f"Litter: {c:.2f}"
                (txt_w, txt_h), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    annotated_bgr,
                    (int(x1), max(0, int(y1) - txt_h - 4)),
                    (int(x1) + txt_w + 4, max(txt_h + 4, int(y1))),
                    box_color,
                    -1
                )
                cv2.putText(
                    annotated_bgr,
                    label_text,
                    (int(x1) + 2, max(txt_h + 1, int(y1) - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        # Calculate Cleanliness Score
        score_res = calculate_cleanliness_score(
            detections=detections,
            image_shape=(height, width),
            config=scoring_config
        )

        # Save annotated image if output_dir provided
        annotated_path = None
        if output_dir:
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            if not output_filename:
                output_filename = f"annotated_{uuid.uuid4().hex[:10]}.jpg"
            save_dest = out_dir / output_filename
            cv2.imwrite(str(save_dest), annotated_bgr)
            annotated_path = str(save_dest)

        return {
            "image_dimensions": {"width": width, "height": height},
            "detection_count": len(detections),
            "detections": detections,
            "scoring": score_res,
            "annotated_image_path": annotated_path,
            "model_metadata": {
                "loaded_checkpoint": self.loaded_model_path,
                "is_fine_tuned": self.is_fine_tuned,
                "confidence_threshold": c_thresh,
                "iou_threshold": i_thresh,
                "device": self.device
            }
        }


def main():
    parser = argparse.ArgumentParser(description="Run litter detection and cleanliness scoring on an image.")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--model", default="models/best.pt", help="Path to fine-tuned YOLO model")
    parser.add_argument("--fallback", default="models/yolov8n.pt", help="Path to baseline fallback model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--out-dir", default="static/results", help="Directory to save annotated image")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    detector = LitterDetector(
        model_path=args.model,
        fallback_path=args.fallback,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )

    result = detector.predict(args.image, output_dir=args.out_dir)
    print("\n--- Detection & Cleanliness Results ---")
    print(f"Litter Items Detected: {result['detection_count']}")
    print(f"Cleanliness Score:     {result['scoring']['score']} / 100 ({result['scoring']['rating_label']})")
    print(f"Baseline Count Score:  {result['scoring']['baseline_score']} / 100")
    print(f"Relative Area Ratio:   {result['scoring']['area_ratio']:.4f}")
    print(f"Annotated Image:       {result['annotated_image_path']}")
    print(f"Interpretation:        {result['scoring']['interpretation']}")


if __name__ == "__main__":
    main()
