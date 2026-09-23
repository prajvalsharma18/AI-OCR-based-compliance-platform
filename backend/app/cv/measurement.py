"""Calibration and physical measurement calculation engine.

Computes axis-specific millimetres-per-pixel ratios from user package dimensions
and user-supplied package bounding box, then applies them to pixel-measured numeral glyphs.
"""

from typing import List, Optional
from app.schemas.measurement import CalibrationMetadata


def compute_calibration_metadata(
    package_width_mm: float,
    package_height_mm: float,
    package_bbox_px: List[int],
    package_width_px: Optional[float] = None,
    package_height_px: Optional[float] = None,
    method: str = "package_dimensions",
    confidence: Optional[float] = None,
    quality: Optional[str] = None,
) -> CalibrationMetadata:
    """Computes axis-specific calibration factors from user dimensions and package bounding box.

    Args:
        package_width_mm: Physical package width in mm (> 0).
        package_height_mm: Physical package height in mm (> 0).
        package_bbox_px: Pixel bounding box [x_min, y_min, x_max, y_max].
        package_width_px: Optional precomputed width in px (defaults to x_max - x_min).
        package_height_px: Optional precomputed height in px (defaults to y_max - y_min).
        method: Calibration reference methodology.
        confidence: Optional confidence score for boundary detection.
        quality: Optional quality assessment string.

    Raises:
        ValueError: If dimensions are non-positive or package_bbox_px is invalid.
    """
    if package_width_mm <= 0 or package_height_mm <= 0:
        raise ValueError("Package physical dimensions must be strictly positive (> 0 mm).")

    if not package_bbox_px or len(package_bbox_px) != 4:
        raise ValueError("package_bbox_px must contain exactly 4 coordinates [x_min, y_min, x_max, y_max].")

    if package_width_px is None:
        package_width_px = float(package_bbox_px[2] - package_bbox_px[0])
    if package_height_px is None:
        package_height_px = float(package_bbox_px[3] - package_bbox_px[1])

    if package_width_px <= 0 or package_height_px <= 0:
        raise ValueError("Package pixel dimensions must be strictly positive (> 0 px).")

    width_mm_per_px = float(package_width_mm) / float(package_width_px)
    height_mm_per_px = float(package_height_mm) / float(package_height_px)

    return CalibrationMetadata(
        method=method,  # type: ignore
        package_width_mm=round(float(package_width_mm), 2),
        package_height_mm=round(float(package_height_mm), 2),
        package_bbox_px=[int(c) for c in package_bbox_px],
        package_width_px=round(float(package_width_px), 2),
        package_height_px=round(float(package_height_px), 2),
        width_mm_per_px=round(width_mm_per_px, 6),
        height_mm_per_px=round(height_mm_per_px, 6),
        confidence=confidence,
        quality=quality,  # type: ignore
    )


def calculate_numeral_height_mm(
    numeral_height_px: int,
    height_mm_per_px: float,
) -> float:
    """Calculates physical numeral height in millimetres using vertical calibration scale."""
    if numeral_height_px < 0:
        raise ValueError("Numeral height in pixels cannot be negative.")
    if height_mm_per_px <= 0:
        raise ValueError("Calibration height_mm_per_px must be strictly positive.")

    return round(float(numeral_height_px) * float(height_mm_per_px), 2)
