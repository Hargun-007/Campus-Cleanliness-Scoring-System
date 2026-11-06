"""
Campus Cleanliness Scoring System - Model Fine-Tuning
Fine-tunes a pretrained YOLOv8n model on the TACO single-class litter dataset.
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

# Set Ultralytics config dir to writable local path
os.environ["YOLO_CONFIG_DIR"] = os.path.abspath("runs/.ultralytics")

import torch
from ultralytics import YOLO, settings

# Ensure all paths point inside workspace
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
logger = logging.getLogger("train")


def verify_dataset(data_yaml_path: str) -> bool:
    """Verifies that the dataset configuration exists and contains images and labels."""
    cfg_file = Path(data_yaml_path)
    if not cfg_file.exists():
        logger.error(f"Dataset configuration YAML not found at: {data_yaml_path}")
        return False

    with open(cfg_file, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    base_path = Path(data_cfg.get("path", "."))
    train_rel = data_cfg.get("train", "train/images")
    val_rel = data_cfg.get("val", "val/images")

    train_img_dir = (base_path / train_rel) if not Path(train_rel).is_absolute() else Path(train_rel)
    val_img_dir = (base_path / val_rel) if not Path(val_rel).is_absolute() else Path(val_rel)

    train_images = list(train_img_dir.glob("*.*"))
    val_images = list(val_img_dir.glob("*.*"))

    logger.info(f"Dataset verification for: {data_yaml_path}")
    logger.info(f"  Train images directory: {train_img_dir} (found {len(train_images)} images)")
    logger.info(f"  Val images directory:   {val_img_dir} (found {len(val_images)} images)")

    if len(train_images) == 0:
        logger.error("No training images found! Run data preparation first.")
        return False

    if len(val_images) == 0:
        logger.error("No validation images found! Run data preparation first.")
        return False

    return True


def train_model(
    data_yaml: str = "data/processed/dataset.yaml",
    base_model: str = "models/yolov8n.pt",
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    optimizer: str = "Adam",
    learning_rate: float = 0.001,
    freeze_layers: int = 10,
    patience: int = 10,
    seed: int = 42,
    device: Optional[str] = None,
    output_dir: str = "runs/detect",
    models_dir: str = "models"
) -> Dict[str, Any]:
    """Fine-tunes YOLOv8n on the custom litter dataset.

    Args:
        data_yaml: Path to Ultralytics dataset.yaml.
        base_model: Starting weights (e.g. models/yolov8n.pt or yolov8n.pt).
        epochs: Maximum training epochs.
        batch_size: Training batch size.
        img_size: Input image dimension.
        optimizer: Optimizer name ('Adam', 'SGD', etc.).
        patience: Early stopping patience epochs.
        seed: Random seed for reproducibility.
        device: 'cuda', 'cpu', or None for auto-detection.
        output_dir: Directory where runs are stored.
        models_dir: Directory to save finalized best.pt checkpoint.

    Returns:
        Dictionary with training outcomes and checkpoint paths.
    """
    # 1. Dataset verification
    if not verify_dataset(data_yaml):
        raise RuntimeError(f"Dataset verification failed for {data_yaml}")

    # 2. Device determination
    if device is None or device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    logger.info("=" * 65)
    logger.info("Campus Cleanliness Scoring System - Model Fine-Tuning")
    logger.info(f"  Base weights:     {base_model}")
    logger.info(f"  Dataset YAML:     {data_yaml}")
    logger.info(f"  Target Device:    {device}")
    logger.info(f"  Epochs:           {epochs}")
    logger.info(f"  Batch size:       {batch_size}")
    logger.info(f"  Image size:       {img_size}")
    logger.info(f"  Optimizer:        {optimizer}")
    logger.info(f"  Early stopping:   patience={patience}")
    logger.info(f"  Seed:             {seed}")
    logger.info("=" * 65)

    # 3. Load model
    if not os.path.exists(base_model):
        logger.info(f"Local base model {base_model} not found, requesting YOLO('yolov8n.pt')...")
        model = YOLO("yolov8n.pt")
    else:
        model = YOLO(base_model)

    # 4. Execute training with fallback handling for OOM
    current_batch = batch_size
    train_success = False
    results = None
    abs_project = os.path.abspath(output_dir)

    while current_batch >= 2 and not train_success:
        try:
            logger.info(f"Initiating training with batch size = {current_batch}...")
            results = model.train(
                data=os.path.abspath(data_yaml),
                epochs=epochs,
                batch=current_batch,
                imgsz=img_size,
                optimizer=optimizer,
                lr0=learning_rate,
                freeze=freeze_layers,
                patience=patience,
                seed=seed,
                device=device,
                project=abs_project,
                name="train_litter",
                exist_ok=True,
                verbose=True,
                plots=True
            )
            train_success = True
        except RuntimeError as e:
            err_str = str(e).lower()
            if "out of memory" in err_str or "cuda error: out of memory" in err_str:
                torch.cuda.empty_cache()
                current_batch //= 2
                logger.warning(f"CUDA Out of Memory! Retrying with smaller batch size: {current_batch}")
            else:
                logger.error(f"Training failed with RuntimeError: {e}")
                raise

    if not train_success:
        raise RuntimeError("Training failed across all batch size attempts.")

    # 5. Locate and save best.pt and last.pt
    run_save_dir = Path(output_dir) / "train_litter"
    weights_dir = run_save_dir / "weights"
    best_ckpt = weights_dir / "best.pt"
    last_ckpt = weights_dir / "last.pt"

    dest_dir = Path(models_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_best = dest_dir / "best.pt"
    dest_last = dest_dir / "last.pt"

    if best_ckpt.exists():
        shutil.copy2(best_ckpt, dest_best)
        logger.info(f"Successfully saved best checkpoint to: {dest_best} ({dest_best.stat().st_size} bytes)")
    else:
        logger.warning(f"best.pt was not produced at {best_ckpt}. Check training logs.")

    if last_ckpt.exists():
        shutil.copy2(last_ckpt, dest_last)
        logger.info(f"Successfully saved last checkpoint to: {dest_last}")

    logger.info("=" * 65)
    logger.info("Training pipeline finished successfully.")
    logger.info(f"  Experiment Directory: {run_save_dir}")
    logger.info(f"  Best Weights:         {dest_best}")
    logger.info("=" * 65)

    return {
        "status": "success",
        "best_checkpoint": str(dest_best),
        "last_checkpoint": str(dest_last),
        "run_directory": str(run_save_dir),
        "epochs_completed": epochs,
        "batch_size_used": current_batch,
        "device": device
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n on single-class litter dataset.")
    parser.add_argument("--data", default="data/processed/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--model", default="models/yolov8n.pt", help="Path to base pretrained weights")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--optimizer", default="Adam", help="Optimizer (default: Adam)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--device", default=None, help="Device ('cuda', 'cpu', or None for auto)")
    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.imgsz,
        optimizer=args.optimizer,
        patience=args.patience,
        device=args.device
    )


if __name__ == "__main__":
    main()
