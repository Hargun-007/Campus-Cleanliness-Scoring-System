"""
Campus Cleanliness Scoring System - Annotation Converter
Converts COCO format annotations from the TACO dataset into YOLO single-class format.
"""

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("convert_annotations")


def convert_coco_bbox_to_yolo(
    coco_bbox: List[float],
    img_width: float,
    img_height: float
) -> Optional[Tuple[float, float, float, float]]:
    """Converts a COCO bounding box [x_min, y_min, width, height] to YOLO normalized
    [x_center, y_center, width, height].

    Args:
        coco_bbox: [x_min, y_min, width, height] in pixel coordinates.
        img_width: Image width in pixels.
        img_height: Image height in pixels.

    Returns:
        Tuple of (x_center, y_center, width, height) normalized to [0.0, 1.0],
        or None if bounding box is invalid/degenerate.
    """
    if len(coco_bbox) != 4:
        return None

    x_min, y_min, w, h = coco_bbox

    # Check for NaN or Inf
    if any(math.isnan(v) or math.isinf(v) for v in (x_min, y_min, w, h)):
        return None

    # Check for non-positive dimensions
    if w <= 0 or h <= 0 or img_width <= 0 or img_height <= 0:
        return None

    # Clip coordinates to image boundary
    x_max = min(img_width, max(0.0, x_min + w))
    y_max = min(img_height, max(0.0, y_min + h))
    x_min_c = max(0.0, min(img_width, x_min))
    y_min_c = max(0.0, min(img_height, y_min))

    actual_w = x_max - x_min_c
    actual_h = y_max - y_min_c

    # If clipped box has negligible or zero area, skip
    if actual_w <= 0.5 or actual_h <= 0.5:
        return None

    # Calculate center and normalize
    x_center = (x_min_c + actual_w / 2.0) / img_width
    y_center = (y_min_c + actual_h / 2.0) / img_height
    norm_w = actual_w / img_width
    norm_h = actual_h / img_height

    # Final clamping to [0.0, 1.0]
    x_center = min(1.0, max(0.0, x_center))
    y_center = min(1.0, max(0.0, y_center))
    norm_w = min(1.0, max(0.0, norm_w))
    norm_h = min(1.0, max(0.0, norm_h))

    return (round(x_center, 6), round(y_center, 6), round(norm_w, 6), round(norm_h, 6))


def convert_taco_dataset(
    annotations_path: str = "data/raw/annotations.json",
    images_dir: str = "data/raw/images",
    output_dir: str = "data/converted",
    target_class_id: int = 0
) -> Dict[str, Any]:
    """Reads TACO COCO annotations, maps categories to single class 'litter' (ID 0),
    and exports YOLO format labels for all available images on disk.

    Args:
        annotations_path: Path to raw annotations.json.
        images_dir: Directory containing downloaded raw images.
        output_dir: Directory where converted label files and images list will be saved.
        target_class_id: YOLO class ID for litter (default: 0).

    Returns:
        Statistics dictionary of the conversion process.
    """
    ann_file = Path(annotations_path)
    if not ann_file.exists():
        raise FileNotFoundError(f"Annotations file not found at {annotations_path}")

    with open(ann_file, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    out_path = Path(output_dir)
    labels_out = out_path / "labels"
    labels_out.mkdir(parents=True, exist_ok=True)

    img_dir_path = Path(images_dir)

    # Index annotations by image_id
    ann_by_image = defaultdict(list)
    for ann in coco_data.get("annotations", []):
        ann_by_image[ann["image_id"]].append(ann)

    # Build category lookup
    valid_category_ids = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}

    stats = {
        "total_images_in_json": len(coco_data.get("images", [])),
        "total_annotations_in_json": len(coco_data.get("annotations", [])),
        "matched_images_on_disk": 0,
        "missing_images_on_disk": 0,
        "converted_labels_written": 0,
        "empty_negative_images": 0,
        "valid_bounding_boxes": 0,
        "skipped_annotations_reasons": defaultdict(int),
        "processed_samples": []
    }

    logger.info(f"Loaded {stats['total_images_in_json']} images and {stats['total_annotations_in_json']} annotations.")

    for img_info in coco_data.get("images", []):
        img_id = img_info["id"]
        rel_file_name = img_info.get("file_name", "")

        # Try to locate image on disk
        img_path = img_dir_path / rel_file_name
        if not img_path.exists():
            # Also try flat basename
            flat_path = img_dir_path / Path(rel_file_name).name
            if flat_path.exists():
                img_path = flat_path
            else:
                stats["missing_images_on_disk"] += 1
                continue

        # Verify image dimensions from actual disk file
        try:
            with Image.open(img_path) as pil_img:
                disk_w, disk_h = pil_img.size
        except Exception as e:
            stats["skipped_annotations_reasons"]["corrupt_image_on_disk"] += 1
            logger.warning(f"Could not open image {img_path}: {e}")
            continue

        stats["matched_images_on_disk"] += 1

        # Process annotations for this image
        coco_anns = ann_by_image.get(img_id, [])
        yolo_lines = []

        # Determine reference dimensions from JSON metadata
        json_w = float(img_info.get("width", disk_w))
        json_h = float(img_info.get("height", disk_h))

        for ann in coco_anns:
            cat_id = ann.get("category_id")
            if cat_id not in valid_category_ids:
                stats["skipped_annotations_reasons"]["invalid_category_id"] += 1
                continue

            raw_bbox = ann.get("bbox")
            if not raw_bbox or len(raw_bbox) != 4:
                stats["skipped_annotations_reasons"]["missing_or_malformed_bbox"] += 1
                continue

            # Convert bbox
            yolo_box = convert_coco_bbox_to_yolo(raw_bbox, json_w, json_h)
            if yolo_box is None:
                stats["skipped_annotations_reasons"]["degenerate_or_oob_bbox"] += 1
                continue

            xc, yc, w, h = yolo_box
            yolo_lines.append(f"{target_class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            stats["valid_bounding_boxes"] += 1

        # Label file name matches base image stem
        safe_name = f"{img_path.parent.name}_{img_path.stem}" if img_path.parent != img_dir_path else img_path.stem
        label_file = labels_out / f"{safe_name}.txt"

        with open(label_file, "w", encoding="utf-8") as lf:
            if yolo_lines:
                lf.write("\n".join(yolo_lines) + "\n")
            else:
                # Valid empty file for genuine negative / no-detection samples
                stats["empty_negative_images"] += 1

        stats["converted_labels_written"] += 1
        stats["processed_samples"].append({
            "image_id": img_id,
            "image_path": str(img_path),
            "label_path": str(label_file),
            "stem": safe_name,
            "width": disk_w,
            "height": disk_h,
            "box_count": len(yolo_lines)
        })

    logger.info("Conversion Summary:")
    logger.info(f"  Matched on disk: {stats['matched_images_on_disk']}")
    logger.info(f"  Labels written: {stats['converted_labels_written']} (Empty negatives: {stats['empty_negative_images']})")
    logger.info(f"  Valid YOLO boxes: {stats['valid_bounding_boxes']}")
    logger.info(f"  Skipped annotations breakdown: {dict(stats['skipped_annotations_reasons'])}")

    # Save manifest
    manifest_file = out_path / "conversion_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": len(stats["processed_samples"]),
            "matched_images_on_disk": stats["matched_images_on_disk"],
            "valid_boxes": stats["valid_bounding_boxes"],
            "empty_negatives": stats["empty_negative_images"],
            "skipped_reasons": dict(stats["skipped_annotations_reasons"]),
            "samples": stats["processed_samples"]
        }, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert TACO COCO annotations to YOLO single-class format.")
    parser.add_argument("--annotations", default="data/raw/annotations.json", help="Path to annotations.json")
    parser.add_argument("--images-dir", default="data/raw/images", help="Path to images directory")
    parser.add_argument("--output-dir", default="data/converted", help="Output directory")
    args = parser.parse_args()

    convert_taco_dataset(args.annotations, args.images_dir, args.output_dir)


if __name__ == "__main__":
    main()

