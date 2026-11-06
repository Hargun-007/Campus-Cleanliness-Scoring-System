"""
Campus Cleanliness Scoring System - Dataset Splitter
Splits converted dataset into 70% Train, 15% Validation, and 15% Test partitions,
preventing data leakage and generating an Ultralytics-compatible dataset.yaml.
"""

import argparse
import json
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("split_dataset")


def split_dataset(
    manifest_path: str = "data/converted/conversion_manifest.json",
    output_dir: str = "data/processed",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, Any]:
    """Splits samples into train/val/test splits, organizes files, and writes dataset.yaml.

    Args:
        manifest_path: Path to conversion manifest JSON.
        output_dir: Root processed directory (data/processed).
        train_ratio: Fraction for training split (default: 0.70).
        val_ratio: Fraction for validation split (default: 0.15).
        test_ratio: Fraction for test split (default: 0.15).
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with split statistics.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-4, "Ratios must sum to 1.0"

    man_file = Path(manifest_path)
    if not man_file.exists():
        raise FileNotFoundError(f"Manifest file not found at {manifest_path}")

    with open(man_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    samples: List[Dict[str, Any]] = manifest.get("samples", [])
    if not samples:
        raise ValueError(f"No samples found in manifest {manifest_path}")

    out_base = Path(output_dir)

    # Clean existing split directories for idempotency
    for split_name in ("train", "val", "test"):
        (out_base / split_name / "images").mkdir(parents=True, exist_ok=True)
        (out_base / split_name / "labels").mkdir(parents=True, exist_ok=True)

    # Group samples by base image id or stem prefix to avoid multi-crop / near-duplicate leakage
    # In TACO, images in batch_X with same stem or scene can be grouped
    grouped_samples: Dict[str, List[Dict[str, Any]]] = {}
    for s in samples:
        group_key = str(s.get("image_id", s["stem"]))
        if group_key not in grouped_samples:
            grouped_samples[group_key] = []
        grouped_samples[group_key].append(s)

    unique_keys = sorted(list(grouped_samples.keys()))

    # Deterministic shuffle
    rng = random.Random(seed)
    rng.shuffle(unique_keys)

    n_total = len(unique_keys)
    n_train = max(1, int(round(n_total * train_ratio)))
    n_val = max(1, int(round(n_total * val_ratio)))
    n_test = n_total - n_train - n_val
    if n_test <= 0:
        n_test = max(1, int(n_total * test_ratio))
        n_train = n_total - n_val - n_test

    train_keys = set(unique_keys[:n_train])
    val_keys = set(unique_keys[n_train:n_train + n_val])
    test_keys = set(unique_keys[n_train + n_val:])

    splits = {
        "train": [],
        "val": [],
        "test": []
    }

    for key, group in grouped_samples.items():
        if key in train_keys:
            target_split = "train"
        elif key in val_keys:
            target_split = "val"
        else:
            target_split = "test"
        splits[target_split].extend(group)

    logger.info(f"Splits allocated (seed={seed}):")
    logger.info(f"  Train: {len(splits['train'])} samples ({len(train_keys)} groups)")
    logger.info(f"  Val:   {len(splits['val'])} samples ({len(val_keys)} groups)")
    logger.info(f"  Test:  {len(splits['test'])} samples ({len(test_keys)} groups)")

    # Copy files into processed split directories
    for split_name, split_samples in splits.items():
        img_dest_dir = out_base / split_name / "images"
        lbl_dest_dir = out_base / split_name / "labels"

        # Clear old contents to ensure clean idempotent run
        for f in img_dest_dir.glob("*.*"):
            f.unlink()
        for f in lbl_dest_dir.glob("*.txt"):
            f.unlink()

        manifest_entries = []
        for s in split_samples:
            src_img = Path(s["image_path"])
            src_lbl = Path(s["label_path"])

            dest_img = img_dest_dir / f"{s['stem']}{src_img.suffix}"
            dest_lbl = lbl_dest_dir / f"{s['stem']}.txt"

            shutil.copy2(src_img, dest_img)
            shutil.copy2(src_lbl, dest_lbl)

            manifest_entries.append({
                "image_id": s["image_id"],
                "stem": s["stem"],
                "image_file": dest_img.name,
                "label_file": dest_lbl.name,
                "box_count": s["box_count"]
            })

        # Save individual split manifest
        split_manifest_path = out_base / f"{split_name}_manifest.json"
        with open(split_manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "split": split_name,
                "count": len(manifest_entries),
                "total_boxes": sum(e["box_count"] for e in manifest_entries),
                "samples": manifest_entries
            }, f, indent=2)

    # Generate Ultralytics dataset.yaml
    # Use forward slashes for cross-platform compatibility
    abs_processed = out_base.resolve().as_posix()
    dataset_yaml_content = {
        "path": abs_processed,
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {
            0: "litter"
        }
    }

    yaml_path = out_base / "dataset.yaml"
    with open(yaml_path, "w", encoding="utf-8") as yf:
        yaml.dump(dataset_yaml_content, yf, default_flow_style=False, sort_keys=False)

    logger.info(f"Generated Ultralytics dataset configuration at {yaml_path}")

    return {
        "train_count": len(splits["train"]),
        "val_count": len(splits["val"]),
        "test_count": len(splits["test"]),
        "dataset_yaml": str(yaml_path)
    }


def main():
    parser = argparse.ArgumentParser(description="Split converted TACO dataset into train/val/test splits.")
    parser.add_argument("--manifest", default="data/converted/conversion_manifest.json", help="Path to conversion manifest")
    parser.add_argument("--output-dir", default="data/processed", help="Path to processed directory")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train ratio (default: 0.70)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Val ratio (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    split_dataset(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()

