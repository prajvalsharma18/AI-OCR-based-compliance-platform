"""Pydantic v2 schemas for Module 3B: Rule 8 Spatial Clearance Analysis.

Defines the contract for statutory clear-space compliance evaluation under
the Legal Metrology (Packaged Commodities) Rules, 2011, Rule 8(1) proviso and SIH 2026 PS 26034.

Rule 8 Requirement:
For Net Quantity declarations, the area surrounding the quantity declaration must be free
from printed information:
- Top: equal to numeral height (H)
- Bottom: equal to numeral height (H)
- Left: equal to 2 x numeral height (2H)
- Right: equal to 2 x numeral height (2H)

Non-quantity declaration numerals (MRP, Dates, Phone, PIN, Barcode, etc.) are explicitly
excluded from Rule 8 and marked as 'not_applicable'.
"""

from typing import Any, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

from app.schemas.measurement import MeasurementQuality

DirectionStatus = Literal[
    "above_threshold",
    "below_threshold",
    "indeterminate",
    "not_applicable",
]

Rule8InspectionStatus = Literal[
    "compliant",
    "non_compliant",
    "indeterminate_low_confidence",
    "indeterminate_missing_measurement",
    "not_applicable",
]


class DirectionalRequirement(BaseModel):
    """Statutory clear-space requirement in millimetres for each cardinal direction."""
    model_config = ConfigDict(extra="forbid")

    top_mm: float = Field(..., ge=0.0, description="Required top clearance: H mm.")
    bottom_mm: float = Field(..., ge=0.0, description="Required bottom clearance: H mm.")
    left_mm: float = Field(..., ge=0.0, description="Required left clearance: 2H mm.")
    right_mm: float = Field(..., ge=0.0, description="Required right clearance: 2H mm.")


class DirectionalClearancePx(BaseModel):
    """Measured spatial clearance distance from quantity declaration bbox in pixels."""
    model_config = ConfigDict(extra="forbid")

    top_px: Optional[float] = Field(default=None, description="Clearance from declaration top to nearest occupied element above in pixels.")
    bottom_px: Optional[float] = Field(default=None, description="Clearance from declaration bottom to nearest occupied element below in pixels.")
    left_px: Optional[float] = Field(default=None, description="Clearance from declaration left to nearest occupied element to the left in pixels.")
    right_px: Optional[float] = Field(default=None, description="Clearance from declaration right to nearest occupied element to the right in pixels.")


class DirectionalClearanceMm(BaseModel):
    """Measured spatial clearance distance converted to physical millimetres via calibration."""
    model_config = ConfigDict(extra="forbid")

    top_mm: Optional[float] = Field(default=None, description="Clearance above in physical mm (scaled via vertical mm/px).")
    bottom_mm: Optional[float] = Field(default=None, description="Clearance below in physical mm (scaled via vertical mm/px).")
    left_mm: Optional[float] = Field(default=None, description="Clearance to left in physical mm (scaled via horizontal mm/px).")
    right_mm: Optional[float] = Field(default=None, description="Clearance to right in physical mm (scaled via horizontal mm/px).")


class DirectionalResults(BaseModel):
    """Compliance threshold evaluation result for each cardinal direction."""
    model_config = ConfigDict(extra="forbid")

    top: DirectionStatus = Field(..., description="Top clearance threshold outcome.")
    bottom: DirectionStatus = Field(..., description="Bottom clearance threshold outcome.")
    left: DirectionStatus = Field(..., description="Left clearance threshold outcome.")
    right: DirectionStatus = Field(..., description="Right clearance threshold outcome.")


class Rule8MeasurementSummary(BaseModel):
    """Physical measurement attributes of the target numeral evaluated against Rule 8."""
    model_config = ConfigDict(extra="forbid")

    height_px: Optional[int] = Field(default=None, ge=0, description="Height of numeral glyph in pixels.")
    height_mm: Optional[float] = Field(default=None, ge=0.0, description="Calibrated height of numeral glyph in mm (H).")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Measurement confidence from Module 2A.")
    quality: Optional[MeasurementQuality] = Field(default=None, description="Measurement quality rating.")


class Rule8Finding(BaseModel):
    """Complete Rule 8 spatial clearance compliance finding for a declaration numeral."""
    model_config = ConfigDict(extra="forbid")

    field: str = Field(..., description="Declaration field identifier (e.g. 'net_quantity').")
    rule_reference: str = Field(
        default="Legal Metrology (Packaged Commodities) Rules, 2011 — Rule 8(1) proviso",
        description="Statutory rule citation."
    )
    clearance_reference: str = Field(
        default="quantity_declaration_bbox",
        description="Geometric reference from which clear space extends: 'quantity_declaration_bbox' per Rule 8(1) proviso."
    )
    declaration_bbox_px: Optional[List[int]] = Field(
        default=None,
        description="Pixel bounding box of the entire quantity declaration [x_min, y_min, x_max, y_max] from which clearance is measured."
    )
    numeral_bbox_px: Optional[List[int]] = Field(
        default=None,
        description="Detected numeral pixel bounding box [x_min, y_min, x_max, y_max]."
    )
    semantic_bbox_px: Optional[List[int]] = Field(
        default=None,
        description="Pixel bounding box of the semantic quantity declaration [x_min, y_min, x_max, y_max]."
    )
    measurement: Optional[Rule8MeasurementSummary] = Field(
        default=None,
        description="Physical numeral measurement details from Module 2A."
    )
    required_clearance_mm: Optional[DirectionalRequirement] = Field(
        default=None,
        description="Required directional clearances in physical mm (H top/bottom, 2H left/right)."
    )
    actual_clearance_px: Optional[DirectionalClearancePx] = Field(
        default=None,
        description="Measured clearances in pixels."
    )
    actual_clearance_mm: Optional[DirectionalClearanceMm] = Field(
        default=None,
        description="Measured clearances in physical mm."
    )
    direction_results: DirectionalResults = Field(
        ...,
        description="Threshold status for top, bottom, left, and right."
    )
    inspection_status: Rule8InspectionStatus = Field(
        ...,
        description="Overall Rule 8 inspection status preserving confidence."
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Overall spatial evaluation confidence score."
    )
    notes: Optional[str] = Field(
        default=None,
        description="Diagnostic explanations, boundary notes, or exclusion reasons."
    )


class Rule8Summary(BaseModel):
    """Aggregate statistics for Rule 8 evaluation."""
    model_config = ConfigDict(extra="forbid")

    total_evaluated: int = Field(default=0, ge=0)
    compliant_count: int = Field(default=0, ge=0)
    non_compliant_count: int = Field(default=0, ge=0)
    indeterminate_count: int = Field(default=0, ge=0)
    not_applicable_count: int = Field(default=0, ge=0)


class Rule8ResponseData(BaseModel):
    """Root data payload for Rule 8 compliance evaluation response."""
    model_config = ConfigDict(extra="forbid")

    summary: Rule8Summary
    findings: List[Rule8Finding] = Field(default_factory=list)
    debug_image_base64: Optional[str] = Field(
        default=None,
        description="Optional base64-encoded JPEG image with debug clearance overlays."
    )
    disclaimer: str = Field(
        default=(
            "Automated inspection-support tool only under SIH 2026 PS 26034. "
            "Findings do not constitute final statutory enforcement orders under the Legal Metrology Act, 2009."
        ),
        description="Legal Metrology advisory notice."
    )


class Rule8Response(BaseModel):
    """Standardized API response envelope for Module 3B."""
    success: Literal[True] = True
    data: Rule8ResponseData


class Rule8ErrorDetail(BaseModel):
    code: str
    message: str


class Rule8ErrorResponse(BaseModel):
    """Standardized error API response envelope for Module 3B."""
    success: Literal[False] = False
    error: Rule8ErrorDetail


class Rule8EvaluationRequest(BaseModel):
    """Request payload for Rule 8 evaluation endpoint."""
    model_config = ConfigDict(extra="allow")

    extraction: Optional[Any] = Field(
        default=None,
        description="Module 1 extraction JSON dict, string, or envelope."
    )
    measurements: Optional[Any] = Field(
        default=None,
        description="Module 2A measurement JSON dict, string, or envelope."
    )
    image_base64: Optional[str] = Field(
        default=None,
        description="Optional base64-encoded package image (with or without data URI header) for spatial pixel analysis."
    )
    include_debug_image: bool = Field(
        default=False,
        description="If True, renders diagnostic visualization with directional clearance lines and bounds."
    )
