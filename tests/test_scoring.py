"""
Unit Tests for Cleanliness Scoring Algorithm
Tests boundary conditions, zero-detection cases, monotonicity, and configuration validation.
"""

import math
import pytest
from src.scoring.cleanliness_score import (
    calculate_cleanliness_score,
    calculate_baseline_score,
    get_score_interpretation,
)
from src.scoring.score_config import ScoringConfig, DEFAULT_SCORING_CONFIG


class TestScoringAlgorithm:
    """Test suite for cleanliness scoring algorithm."""

    def test_zero_detections_returns_perfect_score(self):
        """Zero detections must produce a score of exactly 100 with zero penalties."""
        result = calculate_cleanliness_score(
            detections=[],
            image_shape=(1080, 1920),
            config=DEFAULT_SCORING_CONFIG
        )
        assert result["score"] == 100
        assert result["baseline_score"] == 100
        assert result["litter_count"] == 0
        assert result["area_ratio"] == 0.0
        assert result["count_penalty"] == 0.0
        assert result["area_penalty"] == 0.0
        assert result["rating_label"] == "Excellent / Clean"

    def test_overlapping_boxes_union_area(self):
        """Two overlapping 100x100 boxes in a 200x200 image must give area_ratio 0.4375."""
        detections = [
            {"box": [0, 0, 100, 100], "confidence": 0.9},
            {"box": [50, 50, 150, 150], "confidence": 0.9}
        ]
        result = calculate_cleanliness_score(detections, (200, 200))
        assert result["area_ratio"] == pytest.approx(0.4375, abs=1e-5)
        assert result["total_litter_area_px"] == 17500.0

    def test_score_boundaries(self):
        """Scores must strictly remain within the closed interval [0, 100] under extreme values."""
        # Extreme high litter count and large area coverage
        huge_detections = [
            {"box": [0, 0, 1000, 1000], "confidence": 0.99, "area": 1000000}
            for _ in range(500)
        ]
        result = calculate_cleanliness_score(
            detections=huge_detections,
            image_shape=(1000, 1000),
            config=DEFAULT_SCORING_CONFIG
        )
        assert 0 <= result["score"] <= 100
        assert result["score"] == 0
        assert 0 <= result["baseline_score"] <= 100

    def test_monotonic_count_behavior(self):
        """Increasing the number of litter items must monotonically decrease (or equal) score."""
        image_shape = (1000, 1000)
        scores = []
        counts = [0, 1, 2, 5, 10, 20, 50]

        for n in counts:
            dets = [
                {"box": [10, 10, 30, 30], "confidence": 0.8, "area": 400}
                for _ in range(n)
            ]
            res = calculate_cleanliness_score(dets, image_shape, DEFAULT_SCORING_CONFIG)
            scores.append(res["score"])

        # Check monotonic non-increasing
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"Score increased from count {counts[i]} to {counts[i+1]}"

    def test_monotonic_area_behavior(self):
        """Increasing bounding box area coverage must monotonically decrease (or equal) score."""
        image_shape = (1000, 1000)
        # 1 small item vs 1 medium item vs 1 massive item
        small_det = [{"box": [100, 100, 120, 120], "confidence": 0.9, "area": 400}]
        med_det = [{"box": [100, 100, 300, 300], "confidence": 0.9, "area": 40000}]
        large_det = [{"box": [0, 0, 800, 800], "confidence": 0.9, "area": 640000}]

        score_small = calculate_cleanliness_score(small_det, image_shape)["score"]
        score_med = calculate_cleanliness_score(med_det, image_shape)["score"]
        score_large = calculate_cleanliness_score(large_det, image_shape)["score"]

        assert score_small >= score_med >= score_large

    def test_invalid_image_dimensions_raises(self):
        """Zero or negative image dimensions must raise ValueError to prevent division by zero."""
        with pytest.raises(ValueError):
            calculate_cleanliness_score([], (0, 100))
        with pytest.raises(ValueError):
            calculate_cleanliness_score([], (100, -50))

    def test_baseline_count_score_function(self):
        """Baseline count score must match formula and remain bounded."""
        assert calculate_baseline_score(0) == 100
        assert calculate_baseline_score(-5) == 100
        score_5 = calculate_baseline_score(5, alpha=0.25)
        expected = round(100 * math.exp(-0.25 * 5))
        assert score_5 == expected
        assert 0 <= score_5 <= 100


class TestScoringConfig:
    """Test suite for scoring configuration parameters and validation."""

    def test_valid_custom_config(self):
        """Custom configuration with valid weights must initialize successfully."""
        cfg = ScoringConfig(alpha=0.4, area_scale=0.2, count_weight=0.6, area_weight=0.4)
        assert cfg.alpha == 0.4
        assert cfg.count_weight == 0.6

    def test_invalid_weights_sum_raises(self):
        """Weights not summing to 1.0 must raise ValueError."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ScoringConfig(count_weight=0.7, area_weight=0.5)

    def test_negative_weights_raises(self):
        """Negative weights must raise ValueError."""
        with pytest.raises(ValueError):
            ScoringConfig(count_weight=-0.2, area_weight=1.2)

    def test_non_positive_alpha_raises(self):
        """Non-positive alpha must raise ValueError."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            ScoringConfig(alpha=0.0)

