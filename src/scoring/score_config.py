"""
Campus Cleanliness Scoring System - Scoring Configuration
Defines hyperparameters and constraints for the cleanliness scoring algorithm.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ScoringConfig:
    """Configuration container for cleanliness scoring parameters.

    Attributes:
        alpha: Exponential rate for litter count penalty. Controls how rapidly
               score drops with additional items.
        area_scale: Scaling factor for relative litter area ratio. When bounding
                    boxes occupy this fraction of the image, area penalty reaches 1.0.
        count_weight: Weight assigned to count-based penalty in composite score (0 to 1).
        area_weight: Weight assigned to area-based penalty in composite score (0 to 1).
        baseline_alpha: Exponential rate used in the baseline count-only scoring method.
    """

    alpha: float = 0.35
    area_scale: float = 0.15
    count_weight: float = 0.50
    area_weight: float = 0.50
    baseline_alpha: float = 0.25

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if self.area_scale <= 0:
            raise ValueError(f"area_scale must be positive, got {self.area_scale}")
        if self.count_weight < 0:
            raise ValueError(f"count_weight cannot be negative, got {self.count_weight}")
        if self.area_weight < 0:
            raise ValueError(f"area_weight cannot be negative, got {self.area_weight}")

        weight_sum = self.count_weight + self.area_weight
        if not (0.999 <= weight_sum <= 1.001):
            raise ValueError(
                f"Weights count_weight ({self.count_weight}) + area_weight ({self.area_weight}) "
                f"must sum to 1.0 (got {weight_sum:.4f})"
            )

        if self.baseline_alpha <= 0:
            raise ValueError(f"baseline_alpha must be positive, got {self.baseline_alpha}")


DEFAULT_SCORING_CONFIG = ScoringConfig()

