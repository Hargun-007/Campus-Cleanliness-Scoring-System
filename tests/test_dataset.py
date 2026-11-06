"""
Unit Tests for Dataset Preprocessing and Annotation Conversion
Tests COCO-to-YOLO coordinate transforms, boundary clipping, and label format verification.
"""

import math
import pytest
from pathlib import Path
from src.data.convert_annotations import convert_coco_bbox_to_yolo
from src.data.validate_dataset import validate_yolo_label_line


class TestCoordinateConversion:
    """Test suite for COCO [x_min, y_min, w, h] to YOLO [xc, yc, w, h] normalized transforms."""

    def test_standard_bbox_conversion(self):
        """A 200x200 box starting at (100, 100) in a 1000x1000 image."""
        coco_box = [100.0, 100.0, 200.0, 200.0]
        img_w, img_h = 1000.0, 1000.0

        yolo = convert_coco_bbox_to_yolo(coco_box, img_w, img_h)
        assert yolo is not None
        xc, yc, nw, nh = yolo

        # Center is at (200, 200) -> normalized is 0.2
        assert xc == pytest.approx(0.2, abs=1e-5)
        assert yc == pytest.approx(0.2, abs=1e-5)
        # Width and height normalized are 200/1000 = 0.2
        assert nw == pytest.approx(0.2, abs=1e-5)
        assert nh == pytest.approx(0.2, abs=1e-5)

    def test_full_image_bbox(self):
        """A box covering the entire image should produce (0.5, 0.5, 1.0, 1.0)."""
        coco_box = [0.0, 0.0, 640.0, 480.0]
        yolo = convert_coco_bbox_to_yolo(coco_box, 640.0, 480.0)
        assert yolo is not None
        xc, yc, nw, nh = yolo
        assert xc == pytest.approx(0.5, abs=1e-5)
        assert yc == pytest.approx(0.5, abs=1e-5)
        assert nw == pytest.approx(1.0, abs=1e-5)
        assert nh == pytest.approx(1.0, abs=1e-5)

    def test_clipping_partially_out_of_bounds_bbox(self):
        """Boxes extending past image borders must be clipped to [0, W] and [0, H]."""
        # Box starts at 900, width 300 in 1000x1000 image (extends to 1200)
        coco_box = [900.0, 100.0, 300.0, 200.0]
        yolo = convert_coco_bbox_to_yolo(coco_box, 1000.0, 1000.0)
        assert yolo is not None
        xc, yc, nw, nh = yolo

        # Clipped box is from 900 to 1000 -> width is 100, center is 950 -> xc = 0.95, nw = 0.1
        assert xc == pytest.approx(0.95, abs=1e-5)
        assert nw == pytest.approx(0.10, abs=1e-5)

    def test_completely_out_of_bounds_returns_none(self):
        """Boxes completely outside image boundaries must return None without raising."""
        # Box starts far to the right
        coco_box = [1500.0, 2000.0, 100.0, 100.0]
        assert convert_coco_bbox_to_yolo(coco_box, 1000.0, 1000.0) is None

    def test_negative_or_zero_dimensions_returns_none(self):
        """Boxes with non-positive width or height must return None."""
        assert convert_coco_bbox_to_yolo([10, 10, 0, 50], 100, 100) is None
        assert convert_coco_bbox_to_yolo([10, 10, -20, 50], 100, 100) is None

    def test_nan_or_inf_coordinates_returns_none(self):
        """Coordinates containing NaN or Inf must return None safely."""
        assert convert_coco_bbox_to_yolo([float("nan"), 10, 20, 30], 100, 100) is None
        assert convert_coco_bbox_to_yolo([10, float("inf"), 20, 30], 100, 100) is None


class TestLabelValidation:
    """Test suite for YOLO label format and class-ID validation."""

    def test_valid_label_line(self):
        """Valid line with class 0 and coordinates in [0, 1]."""
        line = "0 0.500000 0.500000 0.200000 0.300000"
        valid, msg = validate_yolo_label_line(line, 1, Path("sample.txt"))
        assert valid is True
        assert msg == ""

    def test_invalid_class_id_fails(self):
        """Class ID other than 0 must be rejected."""
        line = "1 0.5 0.5 0.2 0.3"
        valid, msg = validate_yolo_label_line(line, 1, Path("sample.txt"))
        assert valid is False
        assert "unexpected class_id 1" in msg

    def test_out_of_range_coordinates_fails(self):
        """Coordinates outside [0, 1] must be rejected."""
        line = "0 1.25 0.5 0.2 0.3"
        valid, msg = validate_yolo_label_line(line, 1, Path("sample.txt"))
        assert valid is False
        assert "out of bounds" in msg

    def test_malformed_token_count_fails(self):
        """Lines with fewer or more than 5 tokens must be rejected."""
        line = "0 0.5 0.5 0.2"
        valid, msg = validate_yolo_label_line(line, 1, Path("sample.txt"))
        assert valid is False
        assert "4 tokens, expected 5" in msg

