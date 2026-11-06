"""
Campus Cleanliness Scoring System - Real Campus Batch Evaluation
Evaluates fine-tuned litter detector across a folder of real-world campus photographs,
computes individual cleanliness scores, and documents diagnostic failure modes.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath("."))

from src.model.predict import LitterDetector
from src.scoring.score_config import DEFAULT_SCORING_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("batch_test_campus")


def batch_test_campus(
    images_dir: str = "data/campus_test",
    model_path: str = "models/best.pt",
    output_dir: str = "reports/campus_test_runs",
    conf: float = 0.25,
    iou: float = 0.45
) -> Dict[str, Any]:
    """Runs batch inference on real campus photographs and generates a summary report."""
    in_dir = Path(images_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        raise FileNotFoundError(f"Campus test directory not found at {images_dir}")

    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = [p for p in in_dir.iterdir() if p.suffix.lower() in image_exts]

    if not images:
        logger.warning(f"No images found in {images_dir}. Place campus photos there to evaluate.")
        return {"total_images": 0, "results": []}

    detector = LitterDetector(
        model_path=model_path,
        conf_threshold=conf,
        iou_threshold=iou
    )

    logger.info(f"Processing {len(images)} campus images from {images_dir}...")

    results = []
    total_litter_items = 0
    scores = []

    for img_path in images:
        try:
            res = detector.predict(
                image_input=img_path,
                output_dir=str(out_dir),
                output_filename=f"annotated_{img_path.name}"
            )

            score = res["scoring"]["score"]
            count = res["detection_count"]
            area_ratio = res["scoring"]["area_ratio"]

            total_litter_items += count
            scores.append(score)

            results.append({
                "filename": img_path.name,
                "cleanliness_score": score,
                "litter_count": count,
                "area_ratio": area_ratio,
                "rating_label": res["scoring"]["rating_label"],
                "annotated_image": res["annotated_image_path"]
            })
            logger.info(f"  {img_path.name}: Score={score}/100, Detections={count}")
        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {e}")

    mean_score = round(sum(scores) / len(scores), 2) if scores else 0.0

    report = {
        "total_campus_images_evaluated": len(results),
        "total_litter_detected": total_litter_items,
        "mean_campus_cleanliness_score": mean_score,
        "results": results,
        "failure_mode_analysis": {
            "small_litter_items": "Items smaller than 16x16 pixels (cigarette butts, pins, tiny wrappers) may fall below YOLOv8n receptive field resolution.",
            "partial_occlusion": "Litter partially hidden beneath benches, behind pillars, or in tall grass exhibits reduced feature visibility.",
            "shadows_and_lighting": "Harsh afternoon sun and deep shadows on concrete can cause false positive shadows or false negative missed items.",
            "textured_backgrounds": "Dry leaves, pebbles, and speckled terrazzo tiles can occasionally trigger false detections.",
            "motion_blur": "Low-quality camera shots or moving captures blur high-frequency edges necessary for litter classification."
        }
    }

    report_file = Path("reports") / "campus_test_results.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Campus batch evaluation complete. Report saved to {report_file}")
    logger.info(f"Mean Cleanliness Score: {mean_score}/100 across {len(results)} images.")

    return report


def main():
    parser = argparse.ArgumentParser(description="Batch test real campus photographs.")
    parser.add_argument("--images-dir", default="data/campus_test", help="Path to campus photos folder")
    parser.add_argument("--model", default="models/best.pt", help="Path to fine-tuned model")
    parser.add_argument("--out-dir", default="reports/campus_test_runs", help="Output annotated images folder")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    args = parser.parse_args()

    batch_test_campus(
        images_dir=args.images_dir,
        model_path=args.model,
        output_dir=args.out_dir,
        conf=args.conf,
        iou=args.iou
    )


if __name__ == "__main__":
    main()

