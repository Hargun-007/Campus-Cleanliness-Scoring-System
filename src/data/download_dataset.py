"""
Campus Cleanliness Scoring System - Dataset Downloader
Downloads TACO dataset annotations and images from public mirrors.
"""

import argparse
import json
import logging
import os
import sys
import time
import urllib.request
from pathlib import Path
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("download_dataset")

TACO_ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
)


def download_annotations(dest_path: str = "data/raw/annotations.json") -> bool:
    """Downloads TACO annotations.json if not already present."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 10000:
        logger.info(f"Annotations file already exists at {dest} ({dest.stat().st_size} bytes).")
        return True

    logger.info(f"Downloading TACO annotations from {TACO_ANNOTATIONS_URL}...")
    try:
        req = urllib.request.Request(
            TACO_ANNOTATIONS_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        logger.info(f"Saved annotations to {dest} ({dest.stat().st_size} bytes).")
        return True
    except Exception as e:
        logger.error(f"Failed to download annotations: {e}")
        return False


def download_image(url: str, dest_path: Path, max_retries: int = 3, timeout: int = 15) -> bool:
    """Downloads a single image with retries and verifies its integrity."""
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        # Check if valid image
        try:
            with Image.open(dest_path) as img:
                img.verify()
            return True
        except Exception:
            dest_path.unlink(missing_ok=True)

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = resp.read()

            temp_path = dest_path.with_suffix(".tmp")
            with open(temp_path, "wb") as f:
                f.write(data)

            # Verify image can be decoded
            with Image.open(temp_path) as img:
                img.verify()

            temp_path.replace(dest_path)
            return True
        except Exception as e:
            if attempt == max_retries:
                logger.warning(f"Failed to download {url} after {max_retries} attempts: {e}")
            time.sleep(0.5 * attempt)

    return False


def download_taco_images(
    annotations_path: str = "data/raw/annotations.json",
    images_dir: str = "data/raw/images",
    max_images: int = 150,
    prefer_thumbnails: bool = True
) -> int:
    """Downloads images referenced in TACO annotations.

    Args:
        annotations_path: Path to annotations.json.
        images_dir: Root directory to save images.
        max_images: Maximum number of images to download (use <= 0 for all).
        prefer_thumbnails: Use flickr_640_url (fast, 640px) instead of 4K original.

    Returns:
        Number of successfully downloaded/verified images.
    """
    ann_file = Path(annotations_path)
    if not ann_file.exists():
        logger.error(f"Annotations file not found at {annotations_path}. Download it first.")
        return 0

    with open(ann_file, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    images = coco_data.get("images", [])
    if max_images > 0:
        images = images[:max_images]

    total = len(images)
    logger.info(f"Starting download of {total} TACO images into {images_dir}...")

    success_count = 0
    img_dir_path = Path(images_dir)

    for idx, img_info in enumerate(images, start=1):
        rel_path = img_info.get("file_name", f"image_{img_info.get('id', idx)}.jpg")
        dest_path = img_dir_path / rel_path

        # Determine download URL
        url = None
        if prefer_thumbnails and img_info.get("flickr_640_url"):
            url = img_info["flickr_640_url"]
        elif img_info.get("flickr_url"):
            url = img_info["flickr_url"]

        if not url:
            logger.warning(f"No valid download URL for image ID {img_info.get('id')}")
            continue

        if download_image(url, dest_path):
            success_count += 1
        else:
            # Fallback to alternate url
            alt_url = img_info.get("flickr_url") if prefer_thumbnails else img_info.get("flickr_640_url")
            if alt_url and alt_url != url:
                if download_image(alt_url, dest_path):
                    success_count += 1

        if idx % 25 == 0 or idx == total:
            logger.info(f"Progress: [{idx}/{total}] - Successfully ready: {success_count} images.")

    logger.info(f"Completed TACO download. Total verified images: {success_count}/{total}.")
    return success_count


def main():
    parser = argparse.ArgumentParser(description="Download TACO dataset annotations and images.")
    parser.add_argument("--annotations-path", default="data/raw/annotations.json", help="Path to annotations.json")
    parser.add_argument("--images-dir", default="data/raw/images", help="Target directory for images")
    parser.add_argument("--max-images", type=int, default=150, help="Max images to download (0 for all)")
    parser.add_argument("--full-res", action="store_true", help="Download full resolution instead of 640px thumbnails")
    args = parser.parse_args()

    download_annotations(args.annotations_path)
    download_taco_images(
        annotations_path=args.annotations_path,
        images_dir=args.images_dir,
        max_images=args.max_images,
        prefer_thumbnails=not args.full_res
    )


if __name__ == "__main__":
    main()

