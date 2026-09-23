"""Pydantic v2 schemas for Unified Findings / Unified Inspection JSON.

Defines the normalized, machine-readable contract for aggregating results from:
- Module 1: Semantic Extraction
- Module 2A: Calibrated Measurements
- Module 3C: Rule 6 Declaration Visibility
- Module 3A: Rule 7 Numeral-Height Evaluation
- Module 3B: Rule 8 Spatial Clearance Analysis

This schema serves as the single logical input for:
- PDF Inspection Report Generator
- Dashboard / Legal Metrology Officer (LMO) UI
- External API consumers

Design Constraints:
- Pure aggregation & normalization layer: NO recalculation of statutory rules.
- Normalized user-facing statuses: PASS, NON-COMPLIANT, NOT VISIBLE, NOT ASSESSABLE, NOT APPLICABLE.
- 'NOT VISIBLE' strictly signifies non-detection in the supplied image/PDP and
  must never be interpreted as legal non-compliance or package absence.
- No overall legal verdict, score, or percentage is issued.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field
from app.schemas.history import PersistenceInfo

UnifiedStatus = Literal[
    "PASS",
    "NON-COMPLIANT",
    "NOT VISIBLE",
    "NOT ASSESSABLE",
    "NOT APPLICABLE",
]

VisibilityStatus = Literal[
    "DETECTED",
    "NOT VISIBLE",
    "NOT APPLICABLE",
]


class Rule8DirectionClearance(BaseModel):
    """Normalized clearance details for a single cardinal direction under Rule 8."""
    model_config = ConfigDict(extra="forbid")

    measured_mm: Optional[float] = Field(
        default=None,
        description="Measured physical clearance distance in millimetres."
    )
    required_mm: Optional[float] = Field(
        default=None,
        description="Minimum statutory clearance distance in millimetres."
    )
    status: Literal[
        "above_threshold",
        "below_threshold",
        "indeterminate",
        "not_applicable",
    ] = Field(
        ...,
        description="Clearance comparison status."
    )


# Backward-compatible alias
DirectionClearance = Rule8DirectionClearance


class Rule7FindingSummary(BaseModel):
    """Preserved Rule 7 numeral-height compliance details."""
    model_config = ConfigDict(extra="forbid")

    status: UnifiedStatus = Field(
        ...,
        description="Normalized Rule 7 status: PASS, NON-COMPLIANT, or NOT ASSESSABLE."
    )
    rule_reference: str = Field(
        ...,
        description="Statutory legal citation for Rule 7."
    )
    rule_source_mode: Optional[str] = Field(
        default=None,
        description="Evaluation mode: 'sih_ps_26034' or 'doca_statutory_2011'."
    )
    rule_type: Optional[str] = Field(
        default=None,
        description="Classification of applied rule (e.g. 'quantity_weight_volume', 'general_declaration')."
    )
    measured_height_mm: Optional[float] = Field(
        default=None,
        description="Measured physical numeral height in millimetres from Module 2A."
    )
    required_height_mm: Optional[float] = Field(
        default=None,
        description="Statutory required minimum numeral height in millimetres."
    )
    height_margin_mm: Optional[float] = Field(
        default=None,
        description="Signed difference in mm: measured_height_mm - required_height_mm."
    )
    measurement_quality: Optional[str] = Field(
        default=None,
        description="Module 2A measurement quality rating ('good', 'moderate', 'low', 'failed')."
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Measurement confidence score."
    )
    notes: Optional[str] = Field(
        default=None,
        description="Diagnostic notes from Rule 7 evaluation."
    )


class Rule8FindingSummary(BaseModel):
    """Preserved Rule 8 clear-space compliance details."""
    model_config = ConfigDict(extra="forbid")

    status: UnifiedStatus = Field(
        ...,
        description="Normalized Rule 8 status: PASS, NON-COMPLIANT, or NOT ASSESSABLE."
    )
    rule_reference: str = Field(
        ...,
        description="Statutory legal citation for Rule 8."
    )
    clearance_reference: Optional[str] = Field(
        default=None,
        description="Geometric reference bounding box from which clearance extends."
    )
    target_numeral_height_mm: Optional[float] = Field(
        default=None,
        description="Numeral height H used as reference clear-space multiplier."
    )
    top: Rule8DirectionClearance = Field(
        ...,
        description="Clearance above declaration (required: H mm)."
    )
    bottom: Rule8DirectionClearance = Field(
        ...,
        description="Clearance below declaration (required: H mm)."
    )
    left: Rule8DirectionClearance = Field(
        ...,
        description="Clearance left of declaration (required: 2H mm)."
    )
    right: Rule8DirectionClearance = Field(
        ...,
        description="Clearance right of declaration (required: 2H mm)."
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Spatial clearance evaluation confidence score."
    )
    notes: Optional[str] = Field(
        default=None,
        description="Diagnostic notes or boundary exclusions from Rule 8 evaluation."
    )


class DeclarationRules(BaseModel):
    """Structured container for specific rule results applicable to a declaration."""
    model_config = ConfigDict(extra="forbid")

    rule7: Optional[Rule7FindingSummary] = Field(
        default=None,
        description="Rule 7 numeral-height compliance details if evaluated."
    )
    rule8: Optional[Rule8FindingSummary] = Field(
        default=None,
        description="Rule 8 spatial clearance compliance details if evaluated."
    )


class UnifiedDeclarationFinding(BaseModel):
    """Normalized finding representing a single mandatory declaration and its rule outcomes."""
    model_config = ConfigDict(extra="forbid")

    field: str = Field(
        ...,
        description="Standard declaration key (e.g. 'manufacturer_packer_importer', 'mrp', 'net_quantity', 'date_of_manufacture', 'date_use_by', 'consumer_care')."
    )
    display_name: str = Field(
        ...,
        description="Human-readable title (e.g. 'Retail Sale Price (MRP)', 'Net Quantity')."
    )
    visibility: VisibilityStatus = Field(
        ...,
        description="Visual detection status in supplied image: 'DETECTED', 'NOT VISIBLE', or 'NOT APPLICABLE'."
    )
    raw_text: Optional[str] = Field(
        default=None,
        description="Preserved raw declaration text as detected on package."
    )
    detected_value: Optional[str] = Field(
        default=None,
        description="Extracted value (e.g. price string, date string, or entity name)."
    )
    declared_numeral: Optional[str] = Field(
        default=None,
        description="Extracted numeral string (e.g. '10', '42', '26/07/26')."
    )
    declared_unit: Optional[str] = Field(
        default=None,
        description="Extracted physical unit string (e.g. 'g', 'ml', '₹')."
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Extraction or measurement confidence score."
    )
    bbox: Optional[List[float]] = Field(
        default=None,
        description="Normalized bounding box [x_min, y_min, x_max, y_max] (0.0 to 1.0)."
    )
    rule_reference: Optional[str] = Field(
        default=None,
        description="Primary Rule 6 subsection reference (e.g. 'Rule 6(1)(c)')."
    )
    status: UnifiedStatus = Field(
        ...,
        description="Primary summary status for this declaration: PASS, NON-COMPLIANT, NOT VISIBLE, or NOT ASSESSABLE."
    )
    rules: DeclarationRules = Field(
        default_factory=DeclarationRules,
        description="Specific Rule 7 and Rule 8 evaluation findings."
    )
    notes: Optional[str] = Field(
        default=None,
        description="Contextual or explanatory notes (e.g. multi-panel notice)."
    )


class InspectionSummary(BaseModel):
    """Aggregate statistics summarizing inspection findings without legal conclusions."""
    model_config = ConfigDict(extra="forbid")

    total_declarations_evaluated: int = Field(
        ...,
        ge=0,
        description="Total distinct mandatory declarations evaluated."
    )
    detected_declarations_count: int = Field(
        ...,
        ge=0,
        description="Number of mandatory declarations detected in the supplied image(s)."
    )
    not_visible_declarations_count: int = Field(
        ...,
        ge=0,
        description="Number of mandatory declarations not visible in the supplied image(s)."
    )
    rules_evaluated_count: int = Field(
        ...,
        ge=0,
        description="Total count of specific statutory rule checks (Rule 7 and Rule 8) conducted."
    )
    pass_findings_count: int = Field(
        ...,
        ge=0,
        description="Count of rule evaluations meeting statutory requirements."
    )
    non_compliant_findings_count: int = Field(
        ...,
        ge=0,
        description="Count of rule evaluations failing statutory requirements."
    )
    not_assessable_findings_count: int = Field(
        ...,
        ge=0,
        description="Count of rule evaluations where assessment was indeterminate or had low confidence."
    )


class InspectionMetadata(BaseModel):
    """Packaging and image context metadata preserved from upstream modules."""
    model_config = ConfigDict(extra="forbid")

    brand_name: Optional[str] = Field(
        default=None,
        description="Brand name of the packaged commodity."
    )
    generic_name: Optional[str] = Field(
        default=None,
        description="Generic or common name of the commodity."
    )
    package_type: Optional[str] = Field(
        default=None,
        description="Package type (e.g. 'retail', 'wholesale')."
    )
    image_quality: Optional[str] = Field(
        default=None,
        description="Visual quality assessment from Module 1."
    )
    package_dimensions_mm: Optional[Dict[str, float]] = Field(
        default=None,
        description="Calibrated package dimensions in millimetres (width, height)."
    )
    pdp_area_cm2: Optional[float] = Field(
        default=None,
        description="Calculated or provided Principal Display Panel area in square centimetres."
    )
    rule_source_mode: Optional[str] = Field(
        default=None,
        description="Statutory rule interpretation mode applied ('sih_ps_26034' or 'doca_statutory_2011')."
    )
    source_image_sha256: Optional[str] = Field(
        default=None,
        description="SHA-256 cryptographic hash of the exact original uploaded image bytes."
    )
    source_image_filename: Optional[str] = Field(
        default=None,
        description="Original uploaded file name."
    )
    source_image_path: Optional[str] = Field(
        default=None,
        description="Logical storage reference for the original image (e.g. 'inspections/<id>/original_image.jpg')."
    )
    source_image_storage_path: Optional[str] = Field(
        default=None,
        description="Logical storage path for the original image."
    )


class InspectionResponseData(BaseModel):
    """Root data payload for Unified Inspection response."""
    model_config = ConfigDict(extra="forbid")

    inspection_id: str = Field(
        ...,
        description="Unique inspection run identifier or timestamp."
    )
    metadata: InspectionMetadata
    summary: InspectionSummary
    findings: List[UnifiedDeclarationFinding] = Field(default_factory=list)
    inspection_persistence: PersistenceInfo = Field(
        default_factory=lambda: PersistenceInfo(status="disabled"),
        description="MongoDB persistence outcome; separate from compliance findings.",
    )
    disclaimer: str = Field(
        default=(
            "This inspection summarizes findings detected from the supplied package image(s). "
            "'Not visible in supplied image' does not establish legal absence. "
            "Final enforcement and legal determination rests with the competent Legal Metrology Officer."
        ),
        description="Statutory advisory notice."
    )


class InspectionResponse(BaseModel):
    """Standardized API response envelope for Unified Inspection."""
    success: Literal[True] = True
    data: InspectionResponseData


class InspectionErrorDetail(BaseModel):
    code: str
    message: str


class InspectionErrorResponse(BaseModel):
    """Standardized error API response envelope for Unified Inspection."""
    success: Literal[False] = False
    error: InspectionErrorDetail


class InspectionEvaluationRequest(BaseModel):
    """Request payload for Unified Inspection aggregation endpoint."""
    model_config = ConfigDict(extra="allow")

    extraction: Optional[Any] = Field(
        default=None,
        description="Module 1 extraction JSON dict, string, or envelope."
    )
    measurements: Optional[Any] = Field(
        default=None,
        description="Module 2A measurement JSON dict, string, or envelope."
    )
    rule6: Optional[Any] = Field(
        default=None,
        description="Module 3C Rule 6 response JSON dict, string, or envelope."
    )
    rule7: Optional[Any] = Field(
        default=None,
        description="Module 3A Rule 7 response JSON dict, string, or envelope."
    )
    rule8: Optional[Any] = Field(
        default=None,
        description="Module 3B Rule 8 response JSON dict, string, or envelope."
    )
    inspection_id: Optional[str] = Field(
        default=None,
        description="Optional pre-generated inspection identifier."
    )
    source_image_sha256: Optional[str] = Field(
        default=None,
        description="Optional SHA-256 hash of the exact original uploaded image bytes."
    )
    source_image_filename: Optional[str] = Field(
        default=None,
        description="Optional original uploaded filename."
    )
    source_image_path: Optional[str] = Field(
        default=None,
        description="Optional logical storage path of the stored original image."
    )
    source_image_storage_path: Optional[str] = Field(
        default=None,
        description="Optional logical storage path of the stored original image."
    )
