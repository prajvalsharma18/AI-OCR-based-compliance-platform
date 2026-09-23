"""Pydantic v2 schemas for Module 2A: Calibrated Numeral Size Measurement.

Defines the contract for physical measurement of packaged commodity numerals
using user-supplied calibration dimensions and Module 1 semantic extraction coordinates.
Does NOT output legal compliance verdicts.
"""

from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

MeasurementQuality = Literal["good", "moderate", "low", "failed"]
CalibrationMethod = Literal[
    "package_dimensions",
    "package_dimensions_auto_bbox",
    "package_dimensions_manual_bbox",
]


class CalibrationMetadata(BaseModel):
    """Real-world calibration parameters computed from user package dimensions and bbox."""
    model_config = ConfigDict(extra="forbid")

    method: CalibrationMethod = Field(
        default="package_dimensions",
        description="Calibration reference methodology."
    )
    package_width_mm: float = Field(
        ...,
        gt=0,
        description="User-supplied package physical width in millimetres."
    )
    package_height_mm: float = Field(
        ...,
        gt=0,
        description="User-supplied package physical height in millimetres."
    )
    package_bbox_px: list[int] = Field(
        ...,
        description="Pixel bounding box of package: [x_min, y_min, x_max, y_max]."
    )
    package_width_px: float = Field(
        ...,
        gt=0,
        description="Package bounding box width in pixels (x_max - x_min)."
    )
    package_height_px: float = Field(
        ...,
        gt=0,
        description="Package bounding box height in pixels (y_max - y_min)."
    )
    width_mm_per_px: float = Field(
        ...,
        gt=0,
        description="Horizontal calibration ratio: millimetres per pixel."
    )
    height_mm_per_px: float = Field(
        ...,
        gt=0,
        description="Vertical calibration ratio: millimetres per pixel."
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score of package boundary detection (0.0 to 1.0)."
    )
    quality: Optional[MeasurementQuality] = Field(
        default=None,
        description="Quality assessment of package boundary calibration."
    )


class NumeralMeasurement(BaseModel):
    """Precise physical measurement result for an individual statutory numeral."""
    model_config = ConfigDict(extra="forbid")

    field: str = Field(
        ...,
        description="Target declaration field (e.g. 'net_quantity', 'mrp', 'dates')."
    )
    raw_text: str = Field(
        ...,
        description="Original visible raw text declaration supporting the numeral."
    )
    numeral: str = Field(
        ...,
        description="Target numeral string isolated for measurement (e.g. '1', '500', '120.00')."
    )
    semantic_region_normalized: list[float] = Field(
        ...,
        description="Approximate semantic bounding box from Module 1 [x_min, y_min, x_max, y_max] (0.0 to 1.0)."
    )
    numeral_bbox_px: list[int] = Field(
        ...,
        description="Refined pixel bounding box of the measured numeral glyph [x_min, y_min, x_max, y_max]."
    )
    numeral_height_px: int = Field(
        ...,
        ge=0,
        description="Exact height of the numeral glyph in pixels."
    )
    numeral_height_mm: float = Field(
        ...,
        ge=0,
        description="Calibrated physical height of the numeral in millimetres."
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Measurement confidence score based on contour solidity and edge sharpness (0.0 to 1.0)."
    )
    measurement_quality: MeasurementQuality = Field(
        ...,
        description="Quality assessment: 'good', 'moderate', 'low', or 'failed'."
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional diagnostic assessment of the segmentation and measurement."
    )


class MeasurementImageMetadata(BaseModel):
    """Image dimensions in pixels."""
    model_config = ConfigDict(extra="forbid")

    width: int = Field(..., gt=0, description="Image width in pixels.")
    height: int = Field(..., gt=0, description="Image height in pixels.")


class MeasurementResponseData(BaseModel):
    """Root data payload for successful numeral measurement."""
    model_config = ConfigDict(extra="forbid")

    image: MeasurementImageMetadata
    calibration: CalibrationMetadata
    measurements: list[NumeralMeasurement] = Field(default_factory=list)
    debug_image_base64: Optional[str] = Field(
        default=None,
        description="Optional base64-encoded JPEG image with debug visualization overlays."
    )


class MeasurementResponse(BaseModel):
    """Standardized API response envelope for Module 2A."""
    success: Literal[True] = True
    data: MeasurementResponseData


class MeasurementErrorDetail(BaseModel):
    code: str
    message: str


class MeasurementErrorResponse(BaseModel):
    """Standardized error API response envelope for Module 2A."""
    success: Literal[False] = False
    error: MeasurementErrorDetail
