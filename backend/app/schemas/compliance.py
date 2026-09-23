"""Pydantic v2 schemas for Module 3: Rule 7 Numeral-Height Evaluation Engine.

Defines the contract for statutory numeral-height compliance evaluation under
the Legal Metrology (Packaged Commodities) Rules, 2011 (Rule 7 & Second Schedule Table 1).
Maintains strict separation between mathematical measurement and statutory finding status,
preserving measurement uncertainty without making definitive legal enforcement orders.
"""

from typing import Annotated, Any, List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field

from app.schemas.measurement import MeasurementQuality

FontFormCategory = Literal[
    "normal",
    "molded",
    "perforated",
    "embossed",
    "formed",
    "blown",
]

RuleSourceMode = Literal[
    "sih_ps_26034",
    "doca_statutory_2011",
]

RuleType = Literal[
    "quantity_weight_volume",
    "quantity_length_area_number",
    "general_declaration",
    "unsupported",
    "not_applicable",
]

ComparisonStatus = Literal[
    "above_threshold",
    "below_threshold",
    "indeterminate",
    "unsupported",
    "not_applicable",
]

InspectionFindingStatus = Literal[
    "compliant",
    "non_compliant",
    "indeterminate_low_confidence",
    "indeterminate_missing_measurement",
    "unsupported",
    "not_applicable",
]


class Rule7MeasurementSummary(BaseModel):
    """Encapsulates the physical measurement attributes evaluated against Rule 7."""
    model_config = ConfigDict(extra="forbid")

    height_mm: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Calibrated physical height of the numeral in millimetres."
    )
    height_px: Optional[int] = Field(
        default=None,
        ge=0,
        description="Exact height of the numeral glyph in pixels."
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Measurement confidence score from Module 2A."
    )
    quality: Optional[MeasurementQuality] = Field(
        default=None,
        description="Measurement quality rating: 'good', 'moderate', 'low', or 'failed'."
    )


class Rule7Requirement(BaseModel):
    """Statutory requirement parameters derived deterministically from Rule 7."""
    model_config = ConfigDict(extra="forbid")

    rule_reference: str = Field(
        default="Legal Metrology (Packaged Commodities) Rules, 2011 — Rule 7 (Second Schedule Table 1)",
        description="Statutory legal rule citation."
    )
    rule_source_mode: RuleSourceMode = Field(
        default="sih_ps_26034",
        description="Evaluation mode: 'sih_ps_26034' (SIH technical specification) or 'doca_statutory_2011' (official DoCA PCR 2011 Table-I)."
    )
    rule_type: RuleType = Field(
        ...,
        description="Classification of applied rule (quantity_weight_volume, general_declaration, etc.)."
    )
    condition: str = Field(
        ...,
        description="Condition identifier (e.g. 'below_200_g_ml', 'between_200_and_500_g_ml', 'above_500_g_ml')."
    )
    required_height_mm: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Minimum statutory numeral height in millimetres."
    )
    font_category: FontFormCategory = Field(
        default="normal",
        description="Font/packaging classification: normal or molded/perforated/embossed/formed/blown."
    )
    normalized_value: Optional[float] = Field(
        default=None,
        description="Standardized numerical quantity for range selection (e.g. 500.0 for 0.5 kg)."
    )
    normalized_unit: Optional[str] = Field(
        default=None,
        description="Standardized unit of comparison ('g' for weight, 'ml' for volume)."
    )


class Rule7Comparison(BaseModel):
    """Mathematical comparison between measured height and statutory requirement."""
    model_config = ConfigDict(extra="forbid")

    status: ComparisonStatus = Field(
        ...,
        description="Comparison outcome: 'above_threshold', 'below_threshold', 'indeterminate', etc."
    )
    margin_mm: Optional[float] = Field(
        default=None,
        description="Signed difference in mm: measured_height_mm - required_height_mm."
    )


class Rule7Finding(BaseModel):
    """Complete statutory inspection finding for an individual declaration numeral."""
    model_config = ConfigDict(extra="forbid")

    field: str = Field(
        ...,
        description="Declaration field name (e.g. 'net_quantity', 'mrp', 'dates[0]')."
    )
    declared_text: Optional[str] = Field(
        default=None,
        description="Raw visible declaration text from Module 1."
    )
    declared_numeral: Optional[str] = Field(
        default=None,
        description="Isolated numeral or digit string."
    )
    declared_value: Optional[float] = Field(
        default=None,
        description="Extracted numeric value (for quantity or price)."
    )
    declared_unit: Optional[str] = Field(
        default=None,
        description="Visible unit string (e.g. 'g', 'ml', 'kg')."
    )
    measurement: Optional[Rule7MeasurementSummary] = Field(
        default=None,
        description="Physical measurement details from Module 2A or None if missing."
    )
    requirement: Rule7Requirement = Field(
        ...,
        description="Deterministic Rule 7 requirement."
    )
    comparison: Rule7Comparison = Field(
        ...,
        description="Mathematical comparison against threshold."
    )
    inspection_status: InspectionFindingStatus = Field(
        ...,
        description="Statutory inspection status preserving measurement confidence."
    )
    notes: Optional[str] = Field(
        default=None,
        description="Contextual diagnostic explanation or advisory caveat."
    )


class Rule7Summary(BaseModel):
    """Aggregate statistics for all evaluated declaration numerals on the package."""
    model_config = ConfigDict(extra="forbid")

    rule_source_mode: RuleSourceMode = Field(
        default="sih_ps_26034",
        description="Evaluation mode: 'sih_ps_26034' or 'doca_statutory_2011'."
    )
    total_evaluated: int = Field(default=0, ge=0)
    compliant_count: int = Field(default=0, ge=0)
    non_compliant_count: int = Field(default=0, ge=0)
    indeterminate_count: int = Field(default=0, ge=0)
    unsupported_count: int = Field(default=0, ge=0)
    not_applicable_count: int = Field(default=0, ge=0)


class Rule7ResponseData(BaseModel):
    """Root data payload for Rule 7 compliance evaluation response."""
    model_config = ConfigDict(extra="forbid")

    summary: Rule7Summary
    findings: List[Rule7Finding] = Field(default_factory=list)
    disclaimer: str = Field(
        default=(
            "Automated inspection-support tool only under SIH 2026 PS 26034. "
            "Findings do not constitute final statutory enforcement orders under the Legal Metrology Act, 2009."
        ),
        description="Legal Metrology advisory notice."
    )


class Rule7Response(BaseModel):
    """Standardized API response envelope for Module 3."""
    success: Literal[True] = True
    data: Rule7ResponseData


class Rule7ErrorDetail(BaseModel):
    code: str
    message: str


class Rule7ErrorResponse(BaseModel):
    """Standardized error API response envelope for Module 3."""
    success: Literal[False] = False
    error: Rule7ErrorDetail


class Rule7EvaluationRequest(BaseModel):
    """Request payload for Rule 7 evaluation endpoint."""
    model_config = ConfigDict(extra="allow")

    rule_source_mode: RuleSourceMode = Field(
        default="sih_ps_26034",
        description="Evaluation mode: 'sih_ps_26034' (SIH technical specification) or 'doca_statutory_2011' (official DoCA PCR 2011 Table-I)."
    )
    extraction: Optional[Any] = Field(
        default=None,
        description="Module 1 extraction JSON dict, string, or envelope."
    )
    measurements: Optional[Any] = Field(
        default=None,
        description="Module 2A measurement JSON dict, string, or envelope."
    )
    font_category: FontFormCategory = Field(
        default="normal",
        description="Packaging font/form classification: normal or molded/embossed/perforated/formed/blown."
    )
    pdp_area_cm2: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Principal Display Panel (PDP) area in square centimetres if known."
    )
