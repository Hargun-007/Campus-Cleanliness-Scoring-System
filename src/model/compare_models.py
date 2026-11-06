"""
Campus Cleanliness Scoring System - Model & Scoring Comparison
Compares a scratch-trained YOLOv8n baseline against the transfer-learned fine-tuned YOLOv8n model,
and contrasts the count-only cleanliness score against the count-plus-area score.
"""

import argparse
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Set Ultralytics config dir to writable local path
os.environ["YOLO_CONFIG_DIR"] = os.path.abspath("runs/.ultralytics")

import cv2
import numpy as np
from ultralytics import YOLO, settings

# Ensure paths point inside workspace
settings.update({
    "runs_dir": os.path.abspath("runs"),
    "datasets_dir": os.path.abspath("data"),
    "weights_dir": os.path.abspath("models")
})

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath("."))

from src.scoring.cleanliness_score import calculate_cleanliness_score
from src.scoring.score_config import DEFAULT_SCORING_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("compare_models")


def train_baseline_from_scratch(
    data_yaml: str = "data/processed/dataset.yaml",
    dest_path: str = "models/baseline_scratch.pt",
    epochs: int = 3,
    batch_size: int = 8,
    img_size: int = 640,
    seed: int = 42,
    device: str = "cpu"
) -> str:
    """Trains a YOLOv8n model from scratch without pretrained weights as an experimental baseline."""
    logger.info(f"Training scratch baseline model (yolov8n.yaml) on {data_yaml}...")
    model = YOLO("yolov8n.yaml")
    model.train(
        data=os.path.abspath(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        optimizer="Adam",
        lr0=0.001,
        seed=seed,
        device=device,
        project=os.path.abspath("runs/detect"),
        name="train_baseline_scratch",
        exist_ok=True,
        verbose=False,
        plots=False
    )
    save_dir = Path("runs/detect/train_baseline_scratch/weights")
    best_ckpt = save_dir / "best.pt"
    dest_file = Path(dest_path)
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    if best_ckpt.exists():
        shutil.copy2(best_ckpt, dest_file)
        logger.info(f"Saved scratch baseline checkpoint to {dest_file}")
    else:
        last_ckpt = save_dir / "last.pt"
        if last_ckpt.exists():
            shutil.copy2(last_ckpt, dest_file)
            logger.info(f"Saved last checkpoint as scratch baseline to {dest_file}")
    return str(dest_file)


def evaluate_single_model(model_path: str, data_yaml: str, split: str = "test", device: str = "cpu") -> Dict[str, float]:
    """Evaluates a single model using Ultralytics validation with default conf and iou."""
    model = YOLO(model_path)
    metrics = model.val(
        data=os.path.abspath(data_yaml),
        split=split,
        imgsz=640,
        batch=8,
        device=device,
        verbose=False,
        plots=False
    )
    p = float(metrics.box.mp)
    r = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0

    return {
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1_score": round(f1, 4),
        "map_50": round(map50, 4),
        "map_50_95": round(map50_95, 4)
    }


def compare_models_and_scoring(
    finetuned_model: str = "models/best.pt",
    baseline_model: str = "models/baseline_scratch.pt",
    data_yaml: str = "data/processed/dataset.yaml",
    test_images_dir: str = "data/processed/test/images",
    output_dir: str = "reports",
    train_baseline: bool = False,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Performs side-by-side benchmark comparison between scratch baseline and fine-tuned models,
    and analyzes scoring divergence.
    """
    logger.info("=" * 65)
    logger.info("Campus Cleanliness Scoring System - Baseline vs Fine-Tuned Benchmark")
    logger.info(f"  Scratch Baseline:     {baseline_model}")
    logger.info(f"  Fine-Tuned Model:     {finetuned_model}")
    logger.info("=" * 65)

    if not Path(finetuned_model).exists():
        raise FileNotFoundError(f"Fine-tuned model checkpoint not found at {finetuned_model}")

    if not Path(baseline_model).exists():
        if train_baseline:
            logger.info(f"Baseline '{baseline_model}' not found, training from scratch (--train-baseline enabled)...")
            train_baseline_from_scratch(data_yaml=data_yaml, dest_path=baseline_model, device=device)
        else:
            raise FileNotFoundError(
                f"Baseline model checkpoint not found at {baseline_model}. "
                "Pass --train-baseline to train YOLOv8n from scratch."
            )

    # 1. Model Detection Metrics on held-out test split using default conf and iou
    logger.info("Evaluating Fine-Tuned model on test split...")
    ft_metrics = evaluate_single_model(finetuned_model, data_yaml, split="test", device=device)

    logger.info("Evaluating Scratch Baseline model on test split...")
    base_metrics = evaluate_single_model(baseline_model, data_yaml, split="test", device=device)

    # 2. Scoring Method Comparison on Randomly Sampled Test Split Images
    logger.info("Comparing count-only vs count-plus-area scoring methods...")
    ft_detector = YOLO(finetuned_model)

    test_dir = Path(test_images_dir)
    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    candidate_images = [p for p in test_dir.iterdir() if p.suffix.lower() in image_exts]

    rng = random.Random(42)
    if len(candidate_images) > 20:
        sampled_images = rng.sample(candidate_images, 20)
    else:
        sampled_images = sorted(candidate_images)

    scoring_comparisons = []
    count_scores = []
    area_count_scores = []

    for img_path in sampled_images:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        h, w, _ = bgr.shape
        res = ft_detector.predict(source=bgr, conf=0.25, iou=0.45, device=device, verbose=False)[0]

        detections = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                detections.append({
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf[i])
                })

        score_res = calculate_cleanliness_score(detections, (h, w), DEFAULT_SCORING_CONFIG)
        composite_score = score_res["score"]
        count_only_score = score_res["baseline_score"]

        count_scores.append(count_only_score)
        area_count_scores.append(composite_score)

        scoring_comparisons.append({
            "image": img_path.name,
            "litter_count": len(detections),
            "area_ratio": score_res["area_ratio"],
            "count_only_score": count_only_score,
            "count_plus_area_score": composite_score,
            "difference": composite_score - count_only_score
        })

    # Average score divergence
    mean_count_score = round(float(np.mean(count_scores)), 2) if count_scores else 100.0
    mean_area_score = round(float(np.mean(area_count_scores)), 2) if area_count_scores else 100.0
    mean_divergence = round(abs(mean_area_score - mean_count_score), 2)

    comparison_report = {
        "model_comparison": {
            "scratch_yolov8n_baseline": {
                "checkpoint": baseline_model,
                "notes": "YOLOv8n trained from scratch (yolov8n.yaml) on the litter dataset without pretrained weights",
                "metrics": base_metrics
            },
            "finetuned_yolov8n": {
                "checkpoint": finetuned_model,
                "notes": "Fine-tuned YOLOv8n initialized with COCO weights",
                "metrics": ft_metrics
            },
            "metric_deltas": {
                "precision_delta": round(ft_metrics["precision"] - base_metrics["precision"], 4),
                "recall_delta": round(ft_metrics["recall"] - base_metrics["recall"], 4),
                "f1_delta": round(ft_metrics["f1_score"] - base_metrics["f1_score"], 4),
                "map50_delta": round(ft_metrics["map_50"] - base_metrics["map_50"], 4),
                "map50_95_delta": round(ft_metrics["map_50_95"] - base_metrics["map_50_95"], 4)
            }
        },
        "scoring_comparison": {
            "mean_count_only_score": mean_count_score,
            "mean_count_plus_area_score": mean_area_score,
            "mean_divergence_points": mean_divergence,
            "interpretation": (
                "The count-only baseline score and composite score differ only by the area term. "
                "The composite score penalizes both the total count of litter items and the relative "
                "surface area occupied by their bounding box union."
            ),
            "samples_analyzed": len(scoring_comparisons),
            "sample_details": scoring_comparisons[:10]
        },
        "human_rating_evaluation": {
            "status": "unavailable",
            "message": (
                "No ground-truth human cleanliness ratings CSV provided. To calibrate scoring, collect 50+ campus "
                "photographs, have 3 independent reviewers score each image 0-100, compute average human ratings, "
                "and calculate Mean Absolute Error (MAE) between algorithmic score and human ratings."
            )
        }
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / "model_comparison.json"

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)

    logger.info(f"Comparison report saved to {report_file}")
    logger.info("Benchmark Summary:")
    logger.info(f"  Scratch Baseline mAP@0.5:    {base_metrics['map_50']:.4f}")
    logger.info(f"  Fine-Tuned YOLOv8n mAP@0.5:  {ft_metrics['map_50']:.4f}")
    logger.info(f"  Delta mAP@0.5:               {comparison_report['model_comparison']['metric_deltas']['map50_delta']:+.4f}")
    logger.info(f"  Mean Score Divergence:       {mean_divergence} points")

    return comparison_report


def main():
    parser = argparse.ArgumentParser(description="Compare scratch baseline vs fine-tuned YOLOv8n models.")
    parser.add_argument("--finetuned", default="models/best.pt", help="Fine-tuned model checkpoint")
    parser.add_argument("--baseline", default="models/baseline_scratch.pt", help="Baseline model checkpoint")
    parser.add_argument("--data", default="data/processed/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--train-baseline", action="store_true", help="Train baseline from scratch if not found")
    parser.add_argument("--out-dir", default="reports", help="Reports directory")
    parser.add_argument("--device", default="cpu", help="Device (cpu or cuda)")
    args = parser.parse_args()

    compare_models_and_scoring(
        finetuned_model=args.finetuned,
        baseline_model=args.baseline,
        data_yaml=args.data,
        train_baseline=args.train_baseline,
        output_dir=args.out_dir,
        device=args.device
    )


if __name__ == "__main__":
    main()
