"""
Campus Cleanliness Scoring System - Model Evaluator
Evaluates the fine-tuned YOLOv8n model on the held-out test split (15%) and exports
formal performance metrics (Precision, Recall, F1, mAP50, mAP50-95).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Set Ultralytics config dir to writable local path
os.environ["YOLO_CONFIG_DIR"] = os.path.abspath("runs/.ultralytics")

import numpy as np
import torch
from ultralytics import YOLO, settings

# Ensure paths point inside workspace
settings.update({
    "runs_dir": os.path.abspath("runs"),
    "datasets_dir": os.path.abspath("data"),
    "weights_dir": os.path.abspath("models")
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("evaluate")


def evaluate_model(
    model_path: str = "models/best.pt",
    data_yaml: str = "data/processed/dataset.yaml",
    split: str = "test",
    img_size: int = 640,
    batch_size: int = 8,
    iou_threshold: Optional[float] = None,
    device: Optional[str] = None,
    output_dir: str = "reports"
) -> Dict[str, Any]:
    """Runs Ultralytics validation on the specified split (default: test) and saves metrics JSON.

    Finds the optimal operating confidence threshold from the VAL split and reports
    metrics on the target split using the full PR curve.

    Args:
        model_path: Path to YOLO model weights (.pt).
        data_yaml: Path to dataset configuration YAML.
        split: Target split to evaluate ('test' or 'val').
        img_size: Image dimension for evaluation.
        batch_size: Evaluation batch size.
        iou_threshold: Optional IoU NMS threshold (default None uses Ultralytics 0.7).
        device: 'cpu' or 'cuda'.
        output_dir: Directory where evaluation_metrics_{split}.json is saved.

    Returns:
        Dictionary of computed evaluation metrics.
    """
    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    yaml_file = Path(data_yaml)
    if not yaml_file.exists():
        raise FileNotFoundError(f"Dataset YAML not found at: {data_yaml}")

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 65)
    logger.info("Campus Cleanliness Scoring System - Model Evaluation")
    logger.info(f"  Model checkpoint: {model_path}")
    logger.info(f"  Dataset YAML:     {data_yaml}")
    logger.info(f"  Target Split:     {split}")
    logger.info(f"  Device:           {device}")
    logger.info(f"  IoU threshold:    {iou_threshold if iou_threshold is not None else 'Default (0.7)'}")
    logger.info("=" * 65)

    model = YOLO(str(model_file))

    # 1. Run validation on VAL split first to derive recommended confidence threshold
    logger.info("Running validation on VAL split to find optimal F1 confidence threshold...")
    val_extra = {}
    if iou_threshold is not None:
        val_extra["iou"] = iou_threshold

    val_metrics = model.val(
        data=str(yaml_file.resolve()),
        split="val",
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=os.path.abspath("runs/detect"),
        name="eval_val_tune",
        exist_ok=True,
        plots=False,
        verbose=False,
        **val_extra
    )

    f1_curve = getattr(val_metrics.box, "f1_curve", None)
    px = getattr(val_metrics.box, "px", None)
    recommended_conf_threshold = None

    if f1_curve is not None and px is not None:
        try:
            f1_arr = np.asarray(f1_curve)
            px_arr = np.asarray(px)
            if f1_arr.ndim == 2:
                mean_f1 = f1_arr.mean(axis=0)
            else:
                mean_f1 = f1_arr
            best_idx = int(np.argmax(mean_f1))
            recommended_conf_threshold = round(float(px_arr[best_idx]), 4)
            logger.info(f"Optimal VAL F1: {mean_f1[best_idx]:.4f} at confidence = {recommended_conf_threshold}")
        except Exception as e:
            logger.warning(f"Could not calculate recommended confidence threshold: {e}")
            recommended_conf_threshold = None

    # 2. Run validation on target split (default: test) using Ultralytics default conf (0.001) for full PR curve
    logger.info(f"Running evaluation on target split: {split}...")
    split_metrics = model.val(
        data=str(yaml_file.resolve()),
        split=split,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=os.path.abspath("runs/detect"),
        name=f"eval_{split}",
        exist_ok=True,
        plots=True,
        verbose=True,
        **val_extra
    )

    # Directly extract metrics without silent fallback to 0.0
    precision = float(split_metrics.box.mp)
    recall = float(split_metrics.box.mr)
    map50 = float(split_metrics.box.map50)
    map50_95 = float(split_metrics.box.map)

    if (precision + recall) > 0:
        f1_score = 2.0 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    # Count images and label instances in evaluated split
    num_instances = int(sum(split_metrics.nt_per_class)) if getattr(split_metrics, "nt_per_class", None) is not None else 0
    num_images = int(sum(split_metrics.nt_per_image)) if getattr(split_metrics, "nt_per_image", None) is not None else len(split_metrics.speed)

    eval_results = {
        "model_path": str(model_file),
        "split_evaluated": split,
        "recommended_conf_threshold": recommended_conf_threshold,
        "evaluated_dataset": {
            "num_images": num_images,
            "num_label_instances": num_instances
        },
        "evaluation_settings": {
            "img_size": img_size,
            "iou_threshold": iou_threshold if iou_threshold is not None else 0.7,
            "device": device
        },
        "metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "map_50": round(map50, 4),
            "map_50_95": round(map50_95, 4)
        },
        "metric_explanations": {
            "precision": "Mean precision across classes at optimal F1 confidence threshold.",
            "recall": "Mean recall across classes at optimal F1 confidence threshold.",
            "f1_score": "Harmonic mean of precision and recall at optimal F1 operating point.",
            "map_50": "Mean Average Precision at IoU threshold = 0.50 (full PR curve).",
            "map_50_95": "Mean Average Precision averaged over IoU thresholds from 0.50 to 0.95 (step 0.05)."
        }
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / f"evaluation_metrics_{split}.json"

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    logger.info(f"Evaluation report saved to {report_file}")
    logger.info("Summary of Metrics:")
    logger.info(f"  Recommended Conf: {recommended_conf_threshold}")
    logger.info(f"  Images:           {num_images}, Instances: {num_instances}")
    logger.info(f"  Precision:        {eval_results['metrics']['precision']:.4f}")
    logger.info(f"  Recall:           {eval_results['metrics']['recall']:.4f}")
    logger.info(f"  F1-Score:         {eval_results['metrics']['f1_score']:.4f}")
    logger.info(f"  mAP@0.50:         {eval_results['metrics']['map_50']:.4f}")
    logger.info(f"  mAP@0.5-0.95:     {eval_results['metrics']['map_50_95']:.4f}")

    return eval_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO litter detector on test split.")
    parser.add_argument("--model", default="models/best.pt", help="Path to model checkpoint")
    parser.add_argument("--data", default="data/processed/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--split", default="test", choices=["test", "val"], help="Split to evaluate")
    parser.add_argument("--iou", type=float, default=None, help="IoU threshold (default: Ultralytics 0.7)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default=None, help="Device (cpu or cuda)")
    parser.add_argument("--out-dir", default="reports", help="Reports directory")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        split=args.split,
        img_size=args.imgsz,
        batch_size=args.batch,
        iou_threshold=args.iou,
        device=args.device,
        output_dir=args.out_dir
    )


if __name__ == "__main__":
    main()
