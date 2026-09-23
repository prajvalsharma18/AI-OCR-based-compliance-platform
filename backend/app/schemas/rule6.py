"""Pydantic v2 schemas for Module 3: Rule 6 Declaration Visibility & Completeness.

Defines the contract for evaluating whether mandatory declarations required under
Rule 6 of the Legal Metrology (Packaged Commodities) Rules, 2011 and SIH 2026 PS 26034
are detected in the supplied product image/PDP.

Important Semantic Distinction:
"Not visible in supplied image" is strictly distinguished from "legally missing".
Declarations not visible in the provided image are marked as 'not_visible'
(Display: "Not visible in supplied image"), without making a legal non-compliance conclusion,
because declarations may be located on other packaging panels that were not photographed.
"""

from typing import Any, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

Rule6DeclarationType = Literal[
    "manufacturer_packer_importer",
    "generic_name",
    "mrp",
    "date_of_manufacture_packing_import",
    "net_quantity",
    "consumer_care",
]

Rule6Status = Literal[
    "present",
    "not_visible",
    "not_applicable",
]

Rule6DisplayStatus = Literal[
    "Detected",
    "Not visible in supplied image",
    "Not applicable",
]

STATUS_TO_DISPLAY: dict[Rule6Status, Rule6DisplayStatus] = {
    "present": "Detected",
    "not_visible": "Not visible in supplied image",
    "not_applicable": "Not applicable",
}

DECLARATION_DISPLAY_NAMES: dict[Rule6DeclarationType, str] = {
    "manufacturer_packer_importer": "Manufacturer / Packer / Importer",
    "generic_name": "Generic Name",
    "mrp": "Retail Sale Price (MRP)",
    "date_of_manufacture_packing_import": "Manufacturing / Packing / Import Date",
    "net_quantity": "Net Quantity",
    "consumer_care": "Consumer Care Details",
}

DECLARATION_RULE_SUBSECTIONS: dict[Rule6DeclarationType, str] = {
    "manufacturer_packer_importer": "Rule 6(1)(a)",
    "generic_name": "Rule 6(1)(b)",
    "mrp": "Rule 6(1)(c)",
    "date_of_manufacture_packing_import": "Rule 6(1)(d)",
    "net_quantity": "Rule 6(1)(e)",
    "consumer_care": "Rule 6(1)(f)",
}


class Rule6Finding(BaseModel):
    """Structured finding for an individual Rule 6 mandatory declaration."""
    model_config = ConfigDict(extra="forbid")

    field: Rule6DeclarationType = Field(
        ...,
        description="Standard declaration category key."
    )
    display_name: str = Field(
        ...,
        description="Human-readable declaration name."
    )
    rule_reference: str = Field(
        ...,
        description="Statutory subsection reference (e.g. 'Legal Metrology (Packaged Commodities) Rules, 2011 — Rule 6(1)(a)')."
    )
    status: Rule6Status = Field(
        ...,
        description="Internal structured status: 'present', 'not_visible', or 'not_applicable'."
    )
    display_status: Rule6DisplayStatus = Field(
        ...,
        description="User-facing display string: 'Detected', 'Not visible in supplied image', or 'Not applicable'."
    )
    raw_text: Optional[str] = Field(
        default=None,
        description="Extracted visible text representing this declaration, or null if not detected."
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Extraction confidence score from Module 1."
    )
    bbox: Optional[List[float]] = Field(
        default=None,
        description="Normalized [x_min, y_min, x_max, y_max] bounding box if available."
    )
    notes: Optional[str] = Field(
        default=None,
        description="Context notes, subfield details, or exemption explanations."
    )


class Rule6Summary(BaseModel):
    """Aggregate statistics for Rule 6 declaration visibility evaluation."""
    model_config = ConfigDict(extra="forbid")

    total_declarations: int = Field(default=6, ge=0)
    detected_count: int = Field(default=0, ge=0)
    not_visible_count: int = Field(default=0, ge=0)
    not_applicable_count: int = Field(default=0, ge=0)


class Rule6ResponseData(BaseModel):
    """Payload data envelope for Rule 6 evaluation results."""
    model_config = ConfigDict(extra="forbid")

    rule: str = Field(
        default="Rule 6 — Declaration Visibility",
        description="Rule module title."
    )
    summary: Rule6Summary
    findings: List[Rule6Finding] = Field(default_factory=list)
    disclaimer: str = Field(
        default=(
            "Visibility evaluation in supplied image only. Undetected declarations are reported as "
            "'Not visible in supplied image' and do not constitute a legal conclusion of package non-compliance, "
            "as mandatory declarations may appear on other panels of the package not visible in the provided image."
        ),
        description="Enforcement advisory notice."
    )


class Rule6Response(BaseModel):
    """Standardized API response envelope for Rule 6 evaluation."""
    success: Literal[True] = True
    data: Rule6ResponseData


class Rule6ErrorDetail(BaseModel):
    code: str
    message: str


class Rule6ErrorResponse(BaseModel):
    """Standardized error API response envelope for Rule 6 evaluation."""
    success: Literal[False] = False
    error: Rule6ErrorDetail


class Rule6EvaluationRequest(BaseModel):
    """Request payload for Rule 6 evaluation endpoint."""
    model_config = ConfigDict(extra="allow")

    extraction: Optional[Any] = Field(
        default=None,
        description="Module 1 extraction JSON dict, string, or envelope."
    )
    package_type: Optional[str] = Field(
        default=None,
        description="Optional package type override (retail, wholesale, combination_pack)."
    )
