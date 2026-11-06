"""
Campus Cleanliness Scoring System - Cleanliness Scoring Algorithm
Computes an explainable, deterministic 0-100 cleanliness score from detected litter
items and their bounding-box footprints.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from src.scoring.score_config import ScoringConfig, DEFAULT_SCORING_CONFIG


def calculate_cleanliness_score(
    detections: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
    config: Optional[ScoringConfig] = None
) -> Dict[str, Any]:
    """Calculates the cleanliness score using both litter item count and bounding-box area.

    Formula:
        area_ratio = clamp(sum(bbox_areas) / total_image_area, 0.0, 1.0)
        count_penalty = 1.0 - exp(-alpha * litter_count)
        area_penalty = min(1.0, area_ratio / area_scale)
        cleanliness_score = round(100.0 * (1.0 - (count_weight * count_penalty + area_weight * area_penalty)))

    Args:
        detections: List of detection dictionaries. Each dict should have 'box' or 'bbox'
                    as [x1, y1, x2, y2] (pixel coords) or 'area' in pixels, and 'confidence'.
        image_shape: Tuple of (height, width) of the original image in pixels.
        config: ScoringConfig instance. Defaults to DEFAULT_SCORING_CONFIG.

    Returns:
        Dictionary containing score, baseline_score, count, area_ratio, breakdown,
        rating classification, and cautionary note.
    """
    if config is None:
        config = DEFAULT_SCORING_CONFIG

    height, width = image_shape
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image dimensions: height={height}, width={width}")

    total_image_area = float(height * width)
    litter_count = len(detections)

    # If no litter detected, area and penalties are zero, score is 100
    if litter_count == 0:
        return {
            "score": 100,
            "baseline_score": 100,
            "litter_count": 0,
            "total_litter_area_px": 0.0,
            "image_area_px": total_image_area,
            "area_ratio": 0.0,
            "count_penalty": 0.0,
            "area_penalty": 0.0,
            "composite_penalty": 0.0,
            "rating_label": "Excellent / Clean",
            "interpretation": (
                "Zero litter detections recorded. Campus area appears completely clean "
                "under current detection parameters."
            ),
            "cautionary_note": (
                "Note: A score of 100 indicates no visible litter was detected within "
                "the camera's field of view. It does not guarantee the complete absence "
                "of occluded, microscopic, or out-of-frame waste."
            ),
            "parameters_used": {
                "alpha": config.alpha,
                "area_scale": config.area_scale,
                "count_weight": config.count_weight,
                "area_weight": config.area_weight,
            }
        }

    # Union bounding-box area calculation via boolean mask
    mask = np.zeros((height, width), dtype=bool)
    fallback_area = 0.0

    for det in detections:
        if "box" in det and det["box"] is not None:
            x1, y1, x2, y2 = det["box"]
            ix1 = max(0, min(width, int(round(x1))))
            iy1 = max(0, min(height, int(round(y1))))
            ix2 = max(0, min(width, int(round(x2))))
            iy2 = max(0, min(height, int(round(y2))))
            if ix2 > ix1 and iy2 > iy1:
                mask[iy1:iy2, ix1:ix2] = True
        elif "area" in det and det["area"] is not None:
            fallback_area += float(det["area"])

    total_union_area = float(mask.sum()) + fallback_area

    # Relative area ratio clamped between 0.0 and 1.0
    area_ratio = min(1.0, max(0.0, total_union_area / total_image_area))

    # Calculate penalties
    count_penalty = 1.0 - math.exp(-config.alpha * float(litter_count))
    area_penalty = min(1.0, area_ratio / config.area_scale)

    # Weighted combination
    composite_penalty = (config.count_weight * count_penalty) + (config.area_weight * area_penalty)
    composite_penalty = min(1.0, max(0.0, composite_penalty))

    # Raw score mapped to 0..100 and rounded
    raw_score = 100.0 * (1.0 - composite_penalty)
    cleanliness_score = int(round(min(100.0, max(0.0, raw_score))))

    # Baseline count-only score for comparison (uses config.alpha)
    baseline_score = calculate_baseline_score(litter_count, alpha=config.alpha)

    # Interpret score category
    rating_label, interpretation = get_score_interpretation(cleanliness_score, litter_count)

    return {
        "score": cleanliness_score,
        "baseline_score": baseline_score,
        "litter_count": litter_count,
        "total_litter_area_px": round(total_union_area, 2),
        "image_area_px": round(total_image_area, 2),
        "area_ratio": round(area_ratio, 4),
        "count_penalty": round(count_penalty, 4),
        "area_penalty": round(area_penalty, 4),
        "composite_penalty": round(composite_penalty, 4),
        "rating_label": rating_label,
        "interpretation": interpretation,
        "cautionary_note": (
            "Note: Cleanliness scores are algorithmic heuristics derived from detected bounding boxes "
            "and do not constitute certified environmental audits. Bounding boxes may overlap or capture "
            "portions of background terrain."
        ),
        "parameters_used": {
            "alpha": config.alpha,
            "area_scale": config.area_scale,
            "count_weight": config.count_weight,
            "area_weight": config.area_weight,
        }
    }


def calculate_baseline_score(litter_count: int, alpha: float = 0.35) -> int:
    """Calculates the baseline count-only cleanliness score without considering area.

    Formula:
        score = round(100 * exp(-alpha * litter_count))
    """
    if litter_count <= 0:
        return 100
    penalty = 1.0 - math.exp(-alpha * float(litter_count))
    score = round(100.0 * (1.0 - penalty))
    return int(min(100, max(0, score)))


def get_score_interpretation(score: int, litter_count: int) -> Tuple[str, str]:
    """Provides a balanced, human-readable qualitative interpretation of the score."""
    if score >= 90:
        label = "Excellent / Clean"
        text = (
            f"The analyzed campus area displays high visual cleanliness ({score}/100) with "
            f"{litter_count} item(s) detected. Minor or negligible litter footprint."
        )
    elif score >= 70:
        label = "Good / Satisfactory"
        text = (
            f"The area is moderately clean ({score}/100) with {litter_count} litter item(s) detected. "
            "Routine maintenance and scheduled sweeping are recommended."
        )
    elif score >= 45:
        label = "Fair / Needs Attention"
        text = (
            f"Noticeable litter accumulation detected ({score}/100, {litter_count} items). "
            "Prompt inspection and cleaning by sanitation staff are recommended."
        )
    else:
        label = "Poor / High Litter Density"
        text = (
            f"Critical accumulation of litter detected ({score}/100, {litter_count} items). "
            "Immediate campus sanitation response is advised."
        )
    return label, text

