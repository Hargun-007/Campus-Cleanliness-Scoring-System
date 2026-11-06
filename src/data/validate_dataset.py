"""
Campus Cleanliness Scoring System - Dataset Validator
Performs rigorous structural and semantic validation on processed train/val/test splits.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
from PIL import Image
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("validate_dataset")


def validate_yolo_label_line(line: str, line_no: int, file_path: Path) -> Tuple[bool, str]:
    """Validates a single YOLO label line.

    Expected format: <class_id> <x_center> <y_center> <width> <height>
    """
    parts = line.strip().split()
    if not parts:
        return True, ""  # empty line

    if len(parts) != 5:
        return False, f"Line {line_no} in {file_path.name} has {len(parts)} tokens, expected 5."

    try:
        class_id = int(parts[0])
        xc = float(parts[1])
        yc = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
    except ValueError as e:
        return False, f"Line {line_no} in {file_path.name} contains invalid numeric value: {e}"

    if class_id != 0:
        return False, f"Line {line_no} in {file_path.name} has unexpected class_id {class_id} (expected 0 for litter)."

    for val, name in [(xc, "x_center"), (yc, "y_center"), (w, "width"), (h, "height")]:
        if not (0.0 <= val <= 1.0):
            return False, f"Line {line_no} in {file_path.name} has {name}={val} out of bounds [0.0, 1.0]."

    if w <= 0 or h <= 0:
        return False, f"Line {line_no} in {file_path.name} has non-positive width/height: w={w}, h={h}."

    return True, ""


def validate_split(split_name: str, base_dir: Path) -> Dict[str, Any]:
    """Validates images and labels within a specific split directory."""
    img_dir = base_dir / split_name / "images"
    lbl_dir = base_dir / split_name / "labels"

    stats = {
        "split": split_name,
        "image_count": 0,
        "label_count": 0,
        "box_count": 0,
        "empty_label_files": 0,
        "corrupt_images": 0,
        "missing_labels": 0,
        "invalid_label_lines": 0,
        "errors": []
    }

    if not img_dir.exists() or not lbl_dir.exists():
        stats["errors"].append(f"Directory {img_dir} or {lbl_dir} does not exist.")
        return stats

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in img_dir.iterdir() if p.suffix.lower() in image_extensions]
    stats["image_count"] = len(images)

    for img_path in images:
        # Check image decodability
        try:
            with Image.open(img_path) as im:
                im.verify()
        except Exception as e:
            stats["corrupt_images"] += 1
            stats["errors"].append(f"Corrupt image: {img_path.name} ({e})")
            continue

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            stats["missing_labels"] += 1
            stats["errors"].append(f"Missing label file for image: {img_path.name}")
            continue

        stats["label_count"] += 1

        # Check label file contents
        try:
            with open(lbl_path, "r", encoding="utf-8") as lf:
                lines = [l.strip() for l in lf if l.strip()]

            if not lines:
                stats["empty_label_files"] += 1
            else:
                for idx, line in enumerate(lines, start=1):
                    valid, err_msg = validate_yolo_label_line(line, idx, lbl_path)
                    if not valid:
                        stats["invalid_label_lines"] += 1
                        stats["errors"].append(err_msg)
                    else:
                        stats["box_count"] += 1
        except Exception as e:
            stats["errors"].append(f"Could not read label {lbl_path.name}: {e}")

    return stats


def validate_dataset(processed_dir: str = "data/processed") -> bool:
    """Validates complete dataset structure, splits, and dataset.yaml."""
    base = Path(processed_dir)
    yaml_path = base / "dataset.yaml"

    logger.info(f"Validating processed dataset at {base}...")

    if not yaml_path.exists():
        logger.error(f"dataset.yaml not found at {yaml_path}")
        return False

    with open(yaml_path, "r", encoding="utf-8") as yf:
        try:
            cfg = yaml.safe_load(yf)
        except Exception as e:
            logger.error(f"Malformed YAML in {yaml_path}: {e}")
            return False

    if "names" not in cfg or cfg["names"].get(0) != "litter":
        logger.error(f"Invalid names configuration in {yaml_path}: {cfg.get('names')}")
        return False

    all_valid = True
    total_images = 0
    total_boxes = 0

    for split in ("train", "val", "test"):
        res = validate_split(split, base)
        logger.info(
            f"Split [{split.upper()}]: {res['image_count']} images, "
            f"{res['box_count']} boxes, {res['empty_label_files']} empty negative images, "
            f"{len(res['errors'])} errors."
        )

        total_images += res["image_count"]
        total_boxes += res["box_count"]

        if res["errors"]:
            all_valid = False
            for err in res["errors"][:5]:  # print first few errors
                logger.error(f"  [{split}] {err}")

    logger.info("=" * 60)
    logger.info(f"Dataset Validation Complete.")
    logger.info(f"  Total Images across splits: {total_images}")
    logger.info(f"  Total Bounding Boxes:       {total_boxes}")
    logger.info(f"  Overall Status:             {'PASSED' if all_valid and total_images > 0 else 'FAILED'}")
    logger.info("=" * 60)

    return all_valid and total_images > 0


def main():
    parser = argparse.ArgumentParser(description="Validate processed YOLO dataset.")
    parser.add_argument("--processed-dir", default="data/processed", help="Path to processed directory")
    args = parser.parse_args()

    success = validate_dataset(args.processed_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

