"""Deterministic Unified Inspection Service.

Aggregates and normalizes outputs from:
- Module 1: Semantic Extraction (app/schemas/extraction.py)
- Module 2A: Calibrated Measurements (app/schemas/measurement.py)
- Module 3C: Rule 6 Declaration Visibility (app/schemas/rule6.py)
- Module 3A: Rule 7 Numeral Height (app/schemas/compliance.py)
- Module 3B: Rule 8 Spatial Clearance (app/schemas/rule8.py)

Strict Constraints:
- Pure aggregation and normalization ONLY.
- ZERO rule recalculation (does not recalculate thresholds, font heights, or clearance distances).
- ZERO LLM/OCR/OpenCV passes.
- Standard user-facing status vocabulary: PASS, NON-COMPLIANT, NOT VISIBLE, NOT ASSESSABLE, NOT APPLICABLE.
- 'NOT VISIBLE' strictly means undetected in the supplied image/PDP; it is NEVER converted
  to NON-COMPLIANT or claimed as legally absent from the physical package.
- No overall legal verdict, score, or percentage is issued.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar

from app.schemas.compliance import (
    InspectionFindingStatus as Rule7InspectionStatus,
    Rule7Finding,
    Rule7ResponseData,
)
from app.schemas.extraction import (
    LabelExtractionResult,
    SemanticExtraction,
)
from app.schemas.inspection import (
    DeclarationRules,
    InspectionMetadata,
    InspectionResponseData,
    InspectionSummary,
    Rule7FindingSummary,
    Rule8DirectionClearance,
    Rule8FindingSummary,
    UnifiedDeclarationFinding,
    UnifiedStatus,
    VisibilityStatus,
)
from app.schemas.measurement import MeasurementResponseData
from app.schemas.rule6 import (
    DECLARATION_DISPLAY_NAMES,
    DECLARATION_RULE_SUBSECTIONS,
    Rule6DeclarationType,
    Rule6Finding,
    Rule6ResponseData,
)
from app.schemas.rule8 import (
    Rule8Finding,
    Rule8InspectionStatus,
    Rule8ResponseData,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

SAFE_NOT_VISIBLE_NOTE = (
    "Not visible in supplied image. Other package panels may contain the declaration."
)

RULE7_STATUS_MAP: Dict[str, UnifiedStatus] = {
    "compliant": "PASS",
    "non_compliant": "NON-COMPLIANT",
    "indeterminate_low_confidence": "NOT ASSESSABLE",
    "indeterminate_missing_measurement": "NOT ASSESSABLE",
    "unsupported": "NOT ASSESSABLE",
    "not_applicable": "NOT APPLICABLE",
}

RULE8_STATUS_MAP: Dict[str, UnifiedStatus] = {
    "compliant": "PASS",
    "non_compliant": "NON-COMPLIANT",
    "indeterminate_low_confidence": "NOT ASSESSABLE",
    "indeterminate_missing_measurement": "NOT ASSESSABLE",
    "not_applicable": "NOT APPLICABLE",
}

RULE6_VISIBILITY_MAP: Dict[str, VisibilityStatus] = {
    "present": "DETECTED",
    "not_visible": "NOT VISIBLE",
    "not_applicable": "NOT APPLICABLE",
}


class InspectionServiceError(Exception):
    """Exception raised for Unified Inspection processing errors."""

    def __init__(self, code: str, message: str, status_code: int = 422):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class InspectionService:
    """Deterministic aggregation service for generating Unified Inspection JSON."""

    def _parse_model(self, raw_input: Any, model_cls: Type[T], name: str) -> Optional[T]:
        """Parses an input dict, string, or existing model instance into model_cls."""
        if raw_input is None:
            return None
        if isinstance(raw_input, model_cls):
            return raw_input

        # Handle JSON strings
        if isinstance(raw_input, str):
            trimmed = raw_input.strip()
            if not trimmed or trimmed == "{}":
                return None
            try:
                raw_input = json.loads(trimmed)
            except Exception as exc:
                raise InspectionServiceError(
                    code=f"INVALID_{name.upper()}_JSON",
                    message=f"Failed to parse {name} JSON string: {exc}",
                    status_code=422,
                ) from exc

        # Handle dictionary payloads (with or without 'data' envelope)
        if isinstance(raw_input, dict):
            if not raw_input:
                return None
            if "data" in raw_input and isinstance(raw_input["data"], dict):
                # If target is not an outer response model, unwrap 'data'
                if not name.endswith("Response") and "data" in model_cls.model_fields:
                    pass
                elif not name.endswith("Response") and "data" not in model_cls.model_fields:
                    raw_input = raw_input["data"]

            try:
                return model_cls.model_validate(raw_input)
            except Exception as exc:
                raise InspectionServiceError(
                    code=f"INVALID_{name.upper()}_SCHEMA",
                    message=f"{name} payload does not conform to expected schema: {exc}",
                    status_code=422,
                ) from exc

        return None

    def aggregate(
        self,
        extraction: Optional[Any] = None,
        measurements: Optional[Any] = None,
        rule6: Optional[Any] = None,
        rule7: Optional[Any] = None,
        rule8: Optional[Any] = None,
        inspection_id: Optional[str] = None,
        source_image_sha256: Optional[str] = None,
        source_image_filename: Optional[str] = None,
        source_image_path: Optional[str] = None,
        source_image_storage_path: Optional[str] = None,
    ) -> InspectionResponseData:
        """Aggregates upstream module outputs into a Unified Inspection JSON.

        Args:
            extraction: Module 1 extraction result dict, envelope, or model.
            measurements: Module 2A calibrated measurements dict, envelope, or model.
            rule6: Module 3C Rule 6 visibility evaluation result.
            rule7: Module 3A Rule 7 numeral-height evaluation result.
            rule8: Module 3B Rule 8 spatial clearance evaluation result.
            inspection_id: Optional pre-generated inspection identifier.
            source_image_sha256: Optional SHA-256 hash of the exact original uploaded image bytes.
            source_image_filename: Optional original uploaded filename.
            source_image_path: Optional logical reference to the stored source image.
            source_image_storage_path: Optional logical reference to the stored source image.

        Returns:
            InspectionResponseData containing metadata, summary counts, and unified findings.
        """
        # Parse inputs into concrete models
        parsed_extraction = self._parse_model(extraction, LabelExtractionResult, "extraction")
        parsed_measurements = self._parse_model(measurements, MeasurementResponseData, "measurements")
        parsed_rule6 = self._parse_model(rule6, Rule6ResponseData, "rule6")
        parsed_rule7 = self._parse_model(rule7, Rule7ResponseData, "rule7")
        parsed_rule8 = self._parse_model(rule8, Rule8ResponseData, "rule8")

        sem: Optional[SemanticExtraction] = (
            parsed_extraction.semantic_extraction if parsed_extraction else None
        )

        # Build Metadata
        metadata = self._build_metadata(
            sem,
            parsed_measurements,
            parsed_rule7,
            source_image_sha256=source_image_sha256,
            source_image_filename=source_image_filename,
            source_image_path=source_image_path,
            source_image_storage_path=source_image_storage_path,
        )

        # Index Rule 6 findings by field
        rule6_by_field: Dict[str, Rule6Finding] = {}
        if parsed_rule6 and parsed_rule6.findings:
            for f in parsed_rule6.findings:
                rule6_by_field[f.field] = f

        # Index Rule 7 findings by field
        rule7_by_field: Dict[str, Rule7Finding] = {}
        if parsed_rule7 and parsed_rule7.findings:
            for f in parsed_rule7.findings:
                rule7_by_field[f.field] = f

        # Index Rule 8 findings by field
        rule8_by_field: Dict[str, Rule8Finding] = {}
        if parsed_rule8 and parsed_rule8.findings:
            for f in parsed_rule8.findings:
                rule8_by_field[f.field] = f

        # Build Unified Findings
        findings: List[UnifiedDeclarationFinding] = []

        # 1. Manufacturer / Packer / Importer
        findings.append(self._build_mfr_finding(sem, rule6_by_field.get("manufacturer_packer_importer")))

        # 2. Generic Name
        findings.append(self._build_generic_finding(sem, rule6_by_field.get("generic_name")))

        # 3. Retail Sale Price (MRP)
        findings.append(
            self._build_mrp_finding(
                sem,
                rule6_by_field.get("mrp"),
                rule7_by_field.get("mrp"),
            )
        )

        # 4. Date Declarations (semantic mapping independent of array order)
        date_findings = self._build_date_findings(
            sem,
            rule6_by_field.get("date_of_manufacture_packing_import"),
            parsed_rule7,
        )
        findings.extend(date_findings)

        # 5. Net Quantity
        findings.append(
            self._build_net_qty_finding(
                sem,
                rule6_by_field.get("net_quantity"),
                rule7_by_field.get("net_quantity"),
                rule8_by_field.get("net_quantity"),
            )
        )

        # 6. Consumer Care Details
        findings.append(self._build_consumer_care_finding(sem, rule6_by_field.get("consumer_care")))

        # Compute Summary
        summary = self._build_summary(findings)

        actual_inspection_id = inspection_id or f"INSP-{int(time.time() * 1000)}"

        return InspectionResponseData(
            inspection_id=actual_inspection_id,
            metadata=metadata,
            summary=summary,
            findings=findings,
        )

    def _build_metadata(
        self,
        sem: Optional[SemanticExtraction],
        measurements: Optional[MeasurementResponseData],
        rule7: Optional[Rule7ResponseData],
        source_image_sha256: Optional[str] = None,
        source_image_filename: Optional[str] = None,
        source_image_path: Optional[str] = None,
        source_image_storage_path: Optional[str] = None,
    ) -> InspectionMetadata:
        """Constructs inspection metadata from extraction, measurements, rule7 context, and image provenance."""
        brand_name: Optional[str] = None
        generic_name: Optional[str] = None
        package_type: Optional[str] = None
        image_quality: Optional[str] = None
        package_dimensions_mm: Optional[Dict[str, float]] = None
        pdp_area_cm2: Optional[float] = None
        rule_source_mode: Optional[str] = None

        if sem:
            if sem.package:
                brand_name = sem.package.brand_name
                generic_name = sem.package.generic_name
                package_type = sem.package.package_type
            if sem.image_quality:
                image_quality = sem.image_quality.overall

        if measurements and measurements.calibration:
            cal = measurements.calibration
            package_dimensions_mm = {
                "width_mm": cal.package_width_mm,
                "height_mm": cal.package_height_mm,
            }
            # Surface area of front PDP face in cm2 = (W_mm * H_mm) / 100.0
            pdp_area_cm2 = round((cal.package_width_mm * cal.package_height_mm) / 100.0, 2)

        if rule7 and rule7.summary:
            rule_source_mode = rule7.summary.rule_source_mode

        logical_path = source_image_storage_path or source_image_path

        return InspectionMetadata(
            brand_name=brand_name,
            generic_name=generic_name,
            package_type=package_type,
            image_quality=image_quality,
            package_dimensions_mm=package_dimensions_mm,
            pdp_area_cm2=pdp_area_cm2,
            rule_source_mode=rule_source_mode,
            source_image_sha256=source_image_sha256,
            source_image_filename=source_image_filename,
            source_image_path=logical_path,
            source_image_storage_path=logical_path,
        )

    def _map_rule7_summary(self, r7: Optional[Rule7Finding]) -> Optional[Rule7FindingSummary]:
        """Maps an existing Rule 7 finding to normalized Rule7FindingSummary."""
        if not r7:
            return None

        stat = RULE7_STATUS_MAP.get(r7.inspection_status, "NOT ASSESSABLE")

        measured_h = r7.measurement.height_mm if r7.measurement else None
        required_h = r7.requirement.required_height_mm
        margin_h = r7.comparison.margin_mm if r7.comparison else None
        quality = r7.measurement.quality if r7.measurement else None
        conf = r7.measurement.confidence if r7.measurement else None

        return Rule7FindingSummary(
            status=stat,
            rule_reference=r7.requirement.rule_reference,
            rule_source_mode=r7.requirement.rule_source_mode,
            rule_type=r7.requirement.rule_type,
            measured_height_mm=measured_h,
            required_height_mm=required_h,
            height_margin_mm=margin_h,
            measurement_quality=quality,
            confidence=conf,
            notes=r7.notes,
        )

    def _map_rule8_summary(self, r8: Optional[Rule8Finding]) -> Optional[Rule8FindingSummary]:
        """Maps an existing Rule 8 finding to normalized Rule8FindingSummary."""
        if not r8:
            return None

        stat = RULE8_STATUS_MAP.get(r8.inspection_status, "NOT ASSESSABLE")
        h_mm = r8.measurement.height_mm if r8.measurement else None

        top_clr = Rule8DirectionClearance(
            measured_mm=r8.actual_clearance_mm.top_mm if r8.actual_clearance_mm else None,
            required_mm=r8.required_clearance_mm.top_mm if r8.required_clearance_mm else None,
            status=r8.direction_results.top,
        )
        bottom_clr = Rule8DirectionClearance(
            measured_mm=r8.actual_clearance_mm.bottom_mm if r8.actual_clearance_mm else None,
            required_mm=r8.required_clearance_mm.bottom_mm if r8.required_clearance_mm else None,
            status=r8.direction_results.bottom,
        )
        left_clr = Rule8DirectionClearance(
            measured_mm=r8.actual_clearance_mm.left_mm if r8.actual_clearance_mm else None,
            required_mm=r8.required_clearance_mm.left_mm if r8.required_clearance_mm else None,
            status=r8.direction_results.left,
        )
        right_clr = Rule8DirectionClearance(
            measured_mm=r8.actual_clearance_mm.right_mm if r8.actual_clearance_mm else None,
            required_mm=r8.required_clearance_mm.right_mm if r8.required_clearance_mm else None,
            status=r8.direction_results.right,
        )

        return Rule8FindingSummary(
            status=stat,
            rule_reference=r8.rule_reference,
            clearance_reference=r8.clearance_reference,
            target_numeral_height_mm=h_mm,
            top=top_clr,
            bottom=bottom_clr,
            left=left_clr,
            right=right_clr,
            confidence=r8.confidence,
            notes=r8.notes,
        )

    def _determine_declaration_status(
        self,
        visibility: VisibilityStatus,
        rules: DeclarationRules,
    ) -> UnifiedStatus:
        """Determines primary declaration status per Step 9 guidelines.

        - If visibility == NOT VISIBLE: status = NOT VISIBLE
        - If visibility == NOT APPLICABLE: status = NOT APPLICABLE
        - If rules exist:
          NON-COMPLIANT takes precedence over NOT ASSESSABLE, which takes precedence over PASS.
        - If detected but NO applicable compliance assessment: status = NOT ASSESSABLE.
        """
        if visibility == "NOT VISIBLE":
            return "NOT VISIBLE"
        if visibility == "NOT APPLICABLE":
            return "NOT APPLICABLE"

        rule_statuses: List[UnifiedStatus] = []
        if rules.rule7:
            rule_statuses.append(rules.rule7.status)
        if rules.rule8:
            rule_statuses.append(rules.rule8.status)

        if not rule_statuses:
            # Per Step 9: "Do not treat Rule 6 detection itself as a PASS.
            # If a declaration is detected but has no applicable compliance assessment, use:
            # status = NOT ASSESSABLE rather than incorrectly declaring PASS."
            return "NOT ASSESSABLE"

        if "NON-COMPLIANT" in rule_statuses:
            return "NON-COMPLIANT"
        if "NOT ASSESSABLE" in rule_statuses:
            return "NOT ASSESSABLE"
        if all(s == "PASS" for s in rule_statuses):
            return "PASS"

        return "PASS"

    def _build_mfr_finding(
        self,
        sem: Optional[SemanticExtraction],
        r6: Optional[Rule6Finding],
    ) -> UnifiedDeclarationFinding:
        """Constructs unified finding for Manufacturer / Packer / Importer."""
        field = "manufacturer_packer_importer"
        display_name = DECLARATION_DISPLAY_NAMES[field]
        rule_ref = DECLARATION_RULE_SUBSECTIONS[field]

        vis: VisibilityStatus = "NOT VISIBLE"
        raw_text = None
        detected_val = None
        conf = None
        bbox = None
        notes = SAFE_NOT_VISIBLE_NOTE

        if r6:
            vis = RULE6_VISIBILITY_MAP.get(r6.status, "NOT VISIBLE")
            raw_text = r6.raw_text
            conf = r6.confidence
            bbox = r6.bbox
            notes = SAFE_NOT_VISIBLE_NOTE if vis == "NOT VISIBLE" else r6.notes
        elif sem and sem.manufacturer and sem.manufacturer.status == "present":
            vis = "DETECTED"
            entities = sem.manufacturer.entities or []
            if entities:
                raw_text = "; ".join([e.raw_text for e in entities if e.raw_text])
                detected_val = entities[0].name
                conf = max([e.confidence for e in entities], default=0.95)
                bbox = entities[0].bbox
            notes = "Manufacturer details detected in extraction."

        rules = DeclarationRules()
        status = self._determine_declaration_status(vis, rules)

        return UnifiedDeclarationFinding(
            field=field,
            display_name=display_name,
            visibility=vis,
            raw_text=raw_text,
            detected_value=detected_val,
            confidence=conf,
            bbox=bbox,
            rule_reference=rule_ref,
            status=status,
            rules=rules,
            notes=notes,
        )

    def _build_generic_finding(
        self,
        sem: Optional[SemanticExtraction],
        r6: Optional[Rule6Finding],
    ) -> UnifiedDeclarationFinding:
        """Constructs unified finding for Generic Name."""
        field = "generic_name"
        display_name = DECLARATION_DISPLAY_NAMES[field]
        rule_ref = DECLARATION_RULE_SUBSECTIONS[field]

        vis: VisibilityStatus = "NOT VISIBLE"
        raw_text = None
        detected_val = None
        conf = None
        bbox = None
        notes = SAFE_NOT_VISIBLE_NOTE

        if r6:
            vis = RULE6_VISIBILITY_MAP.get(r6.status, "NOT VISIBLE")
            raw_text = r6.raw_text
            conf = r6.confidence
            bbox = r6.bbox
            notes = SAFE_NOT_VISIBLE_NOTE if vis == "NOT VISIBLE" else r6.notes
            if sem and sem.generic_name:
                detected_val = sem.generic_name.value
        elif sem and sem.generic_name and sem.generic_name.status == "present":
            vis = "DETECTED"
            raw_text = sem.generic_name.raw_text or sem.generic_name.value
            detected_val = sem.generic_name.value
            conf = sem.generic_name.confidence
            bbox = sem.generic_name.bbox
            notes = f"Generic name declared: '{sem.generic_name.value}'."

        rules = DeclarationRules()
        status = self._determine_declaration_status(vis, rules)

        return UnifiedDeclarationFinding(
            field=field,
            display_name=display_name,
            visibility=vis,
            raw_text=raw_text,
            detected_value=detected_val,
            confidence=conf,
            bbox=bbox,
            rule_reference=rule_ref,
            status=status,
            rules=rules,
            notes=notes,
        )

    def _build_mrp_finding(
        self,
        sem: Optional[SemanticExtraction],
        r6: Optional[Rule6Finding],
        r7: Optional[Rule7Finding],
    ) -> UnifiedDeclarationFinding:
        """Constructs unified finding for Retail Sale Price (MRP)."""
        field = "mrp"
        display_name = DECLARATION_DISPLAY_NAMES[field]
        rule_ref = DECLARATION_RULE_SUBSECTIONS[field]

        vis: VisibilityStatus = "NOT VISIBLE"
        raw_text = None
        detected_val = None
        declared_num = None
        declared_unit = None
        conf = None
        bbox = None
        notes = SAFE_NOT_VISIBLE_NOTE

        if r6:
            vis = RULE6_VISIBILITY_MAP.get(r6.status, "NOT VISIBLE")
            raw_text = r6.raw_text
            conf = r6.confidence
            bbox = r6.bbox
            notes = SAFE_NOT_VISIBLE_NOTE if vis == "NOT VISIBLE" else r6.notes
        elif sem and sem.mrp and sem.mrp.status == "present":
            vis = "DETECTED"
            raw_text = sem.mrp.raw_text
            conf = sem.mrp.confidence
            bbox = sem.mrp.bbox
            notes = "MRP detected."

        if sem and sem.mrp:
            if sem.mrp.value is not None:
                detected_val = f"{sem.mrp.currency or '₹'} {sem.mrp.value:.2f}"
                declared_num = str(int(sem.mrp.value)) if sem.mrp.value.is_integer() else str(sem.mrp.value)
            declared_unit = sem.mrp.currency or "₹"

        # Merge Rule 7 details
        r7_summary = self._map_rule7_summary(r7)
        if r7:
            if r7.declared_text and not raw_text:
                raw_text = r7.declared_text
            if r7.declared_numeral and not declared_num:
                declared_num = r7.declared_numeral
            if r7.declared_unit and not declared_unit:
                declared_unit = r7.declared_unit

        rules = DeclarationRules(rule7=r7_summary)
        status = self._determine_declaration_status(vis, rules)

        return UnifiedDeclarationFinding(
            field=field,
            display_name=display_name,
            visibility=vis,
            raw_text=raw_text,
            detected_value=detected_val,
            declared_numeral=declared_num,
            declared_unit=declared_unit,
            confidence=conf,
            bbox=bbox,
            rule_reference=rule_ref,
            status=status,
            rules=rules,
            notes=notes,
        )

    def _find_r7_for_date(
        self,
        date_item_type: str,
        date_item_raw: str,
        rule7_data: Optional[Rule7ResponseData],
        index: int,
    ) -> Optional[Rule7Finding]:
        """Finds matching Rule 7 finding for a semantic date item without assuming array order."""
        if not rule7_data or not rule7_data.findings:
            return None

        # 1. Match by field name
        target_field = "date_of_manufacture" if date_item_type in ("manufacture", "packing", "import") else "date_use_by"
        for f in rule7_data.findings:
            if f.field == target_field:
                return f

        # 2. Match by exact text substring
        for f in rule7_data.findings:
            if "date" in f.field or "dates[" in f.field:
                if f.declared_text and f.declared_text.strip() == date_item_raw.strip():
                    return f

        # 3. Match by semantic keywords in declared_text
        is_mfd = date_item_type in ("manufacture", "packing", "import")
        for f in rule7_data.findings:
            if "date" in f.field or "dates[" in f.field:
                txt = (f.declared_text or "").lower()
                if is_mfd and any(k in txt for k in ("mfd", "mfg", "pkd", "pkg", "pack")):
                    return f
                if not is_mfd and any(k in txt for k in ("use", "exp", "best", "bb")):
                    return f

        # 4. Fallback to array index match if dates[index] matches
        for f in rule7_data.findings:
            if f.field == f"dates[{index}]":
                return f

        return None

    def _build_date_findings(
        self,
        sem: Optional[SemanticExtraction],
        r6: Optional[Rule6Finding],
        rule7_data: Optional[Rule7ResponseData],
    ) -> List[UnifiedDeclarationFinding]:
        """Constructs unified findings for date declarations, preserving semantic separation."""
        rule_ref = DECLARATION_RULE_SUBSECTIONS["date_of_manufacture_packing_import"]

        dates = sem.dates if (sem and sem.dates) else []
        present_dates = [d for d in dates if d.status == "present"]

        # Case 1: Multiple dates detected in Module 1 (e.g. manufacture and use_by)
        if len(present_dates) > 1:
            findings: List[UnifiedDeclarationFinding] = []
            for idx, date_item in enumerate(present_dates):
                is_mfd = date_item.date_type in ("manufacture", "packing", "import")
                field_name = "date_of_manufacture" if is_mfd else "date_use_by"
                disp_name = (
                    "Manufacturing / Packing Date" if is_mfd else "Best Before / Use By Date"
                )

                # Match Rule 7 finding semantically
                r7 = self._find_r7_for_date(date_item.date_type, date_item.raw_text, rule7_data, idx)
                r7_summary = self._map_rule7_summary(r7)

                declared_num = None
                if r7 and r7.declared_numeral:
                    declared_num = r7.declared_numeral
                elif date_item.month and date_item.year:
                    declared_num = f"{date_item.month}/{date_item.year}"
                elif date_item.raw_text:
                    declared_num = date_item.raw_text

                detected_val = (
                    f"{date_item.month}/{date_item.year}"
                    if (date_item.month and date_item.year)
                    else date_item.raw_text
                )

                rules = DeclarationRules(rule7=r7_summary)
                status = self._determine_declaration_status("DETECTED", rules)

                findings.append(
                    UnifiedDeclarationFinding(
                        field=field_name,
                        display_name=disp_name,
                        visibility="DETECTED",
                        raw_text=date_item.raw_text,
                        detected_value=detected_val,
                        declared_numeral=declared_num,
                        confidence=date_item.confidence,
                        bbox=date_item.bbox,
                        rule_reference=rule_ref,
                        status=status,
                        rules=rules,
                        notes=f"Declared {disp_name.lower()} detected.",
                    )
                )
            return findings

        # Case 2: Exactly 1 date detected
        if len(present_dates) == 1:
            date_item = present_dates[0]
            is_mfd = date_item.date_type in ("manufacture", "packing", "import", "unknown")
            field_name = "date_of_manufacture" if is_mfd else "date_use_by"
            disp_name = (
                "Manufacturing / Packing Date" if is_mfd else "Best Before / Use By Date"
            )

            r7 = self._find_r7_for_date(date_item.date_type, date_item.raw_text, rule7_data, 0)
            r7_summary = self._map_rule7_summary(r7)

            declared_num = None
            if r7 and r7.declared_numeral:
                declared_num = r7.declared_numeral
            elif date_item.month and date_item.year:
                declared_num = f"{date_item.month}/{date_item.year}"
            elif date_item.raw_text:
                declared_num = date_item.raw_text

            detected_val = (
                f"{date_item.month}/{date_item.year}"
                if (date_item.month and date_item.year)
                else date_item.raw_text
            )

            rules = DeclarationRules(rule7=r7_summary)
            status = self._determine_declaration_status("DETECTED", rules)

            return [
                UnifiedDeclarationFinding(
                    field=field_name,
                    display_name=disp_name,
                    visibility="DETECTED",
                    raw_text=date_item.raw_text,
                    detected_value=detected_val,
                    declared_numeral=declared_num,
                    confidence=date_item.confidence,
                    bbox=date_item.bbox,
                    rule_reference=rule_ref,
                    status=status,
                    rules=rules,
                    notes=f"Declared date ({date_item.date_type}) detected.",
                )
            ]

        # Case 3: No dates detected in extraction (or extraction missing)
        vis: VisibilityStatus = "NOT VISIBLE"
        raw_text = None
        conf = None
        bbox = None
        notes = SAFE_NOT_VISIBLE_NOTE

        if r6:
            vis = RULE6_VISIBILITY_MAP.get(r6.status, "NOT VISIBLE")
            raw_text = r6.raw_text
            conf = r6.confidence
            bbox = r6.bbox
            notes = SAFE_NOT_VISIBLE_NOTE if vis == "NOT VISIBLE" else r6.notes

        r7 = None
        if rule7_data and rule7_data.findings:
            for f in rule7_data.findings:
                if "date" in f.field:
                    r7 = f
                    break
        r7_summary = self._map_rule7_summary(r7)

        rules = DeclarationRules(rule7=r7_summary)
        status = self._determine_declaration_status(vis, rules)

        return [
            UnifiedDeclarationFinding(
                field="date_of_manufacture",
                display_name="Manufacturing / Packing Date",
                visibility=vis,
                raw_text=raw_text,
                confidence=conf,
                bbox=bbox,
                rule_reference=rule_ref,
                status=status,
                rules=rules,
                notes=notes,
            )
        ]

    def _build_net_qty_finding(
        self,
        sem: Optional[SemanticExtraction],
        r6: Optional[Rule6Finding],
        r7: Optional[Rule7Finding],
        r8: Optional[Rule8Finding],
    ) -> UnifiedDeclarationFinding:
        """Constructs unified finding for Net Quantity, preserving Rule 7 and Rule 8 results."""
        field = "net_quantity"
        display_name = DECLARATION_DISPLAY_NAMES[field]
        rule_ref = DECLARATION_RULE_SUBSECTIONS[field]

        vis: VisibilityStatus = "NOT VISIBLE"
        raw_text = None
        detected_val = None
        declared_num = None
        declared_unit = None
        conf = None
        bbox = None
        notes = SAFE_NOT_VISIBLE_NOTE

        if r6:
            vis = RULE6_VISIBILITY_MAP.get(r6.status, "NOT VISIBLE")
            raw_text = r6.raw_text
            conf = r6.confidence
            bbox = r6.bbox
            notes = SAFE_NOT_VISIBLE_NOTE if vis == "NOT VISIBLE" else r6.notes
        elif sem and sem.net_quantity and sem.net_quantity.status == "present":
            vis = "DETECTED"
            raw_text = sem.net_quantity.raw_text
            conf = sem.net_quantity.confidence
            bbox = sem.net_quantity.bbox
            notes = "Net quantity detected."

        if sem and sem.net_quantity:
            if sem.net_quantity.value is not None:
                detected_val = f"{sem.net_quantity.value} {sem.net_quantity.unit or ''}".strip()
                val = sem.net_quantity.value
                declared_num = str(int(val)) if isinstance(val, float) and val.is_integer() else str(val)
            declared_unit = sem.net_quantity.unit

        # Merge Rule 7 details
        r7_summary = self._map_rule7_summary(r7)
        if r7:
            if r7.declared_text and not raw_text:
                raw_text = r7.declared_text
            if r7.declared_numeral and not declared_num:
                declared_num = r7.declared_numeral
            if r7.declared_unit and not declared_unit:
                declared_unit = r7.declared_unit

        # Merge Rule 8 details
        r8_summary = self._map_rule8_summary(r8)

        rules = DeclarationRules(rule7=r7_summary, rule8=r8_summary)
        status = self._determine_declaration_status(vis, rules)

        return UnifiedDeclarationFinding(
            field=field,
            display_name=display_name,
            visibility=vis,
            raw_text=raw_text,
            detected_value=detected_val,
            declared_numeral=declared_num,
            declared_unit=declared_unit,
            confidence=conf,
            bbox=bbox,
            rule_reference=rule_ref,
            status=status,
            rules=rules,
            notes=notes,
        )

    def _build_consumer_care_finding(
        self,
        sem: Optional[SemanticExtraction],
        r6: Optional[Rule6Finding],
    ) -> UnifiedDeclarationFinding:
        """Constructs unified finding for Consumer Care Details."""
        field = "consumer_care"
        display_name = DECLARATION_DISPLAY_NAMES[field]
        rule_ref = DECLARATION_RULE_SUBSECTIONS[field]

        vis: VisibilityStatus = "NOT VISIBLE"
        raw_text = None
        conf = None
        bbox = None
        notes = SAFE_NOT_VISIBLE_NOTE

        if r6:
            vis = RULE6_VISIBILITY_MAP.get(r6.status, "NOT VISIBLE")
            raw_text = r6.raw_text
            conf = r6.confidence
            bbox = r6.bbox
            notes = SAFE_NOT_VISIBLE_NOTE if vis == "NOT VISIBLE" else r6.notes
        elif sem and sem.consumer_care and sem.consumer_care.status == "present":
            vis = "DETECTED"
            raw_text = sem.consumer_care.raw_text
            conf = sem.consumer_care.confidence
            bbox = sem.consumer_care.bbox
            notes = "Consumer care channels detected."

        rules = DeclarationRules()
        status = self._determine_declaration_status(vis, rules)

        return UnifiedDeclarationFinding(
            field=field,
            display_name=display_name,
            visibility=vis,
            raw_text=raw_text,
            confidence=conf,
            bbox=bbox,
            rule_reference=rule_ref,
            status=status,
            rules=rules,
            notes=notes,
        )

    def _build_summary(
        self,
        findings: List[UnifiedDeclarationFinding],
    ) -> InspectionSummary:
        """Computes aggregate counts across all evaluated findings and rules per Step 11."""
        total_decls = len(findings)
        detected_count = sum(1 for f in findings if f.visibility == "DETECTED")
        not_visible_count = sum(1 for f in findings if f.visibility == "NOT VISIBLE")

        rules_evaluated = 0
        pass_count = 0
        non_compliant_count = 0
        not_assessable_count = 0

        for f in findings:
            if f.rules.rule7:
                rules_evaluated += 1
                if f.rules.rule7.status == "PASS":
                    pass_count += 1
                elif f.rules.rule7.status == "NON-COMPLIANT":
                    non_compliant_count += 1
                elif f.rules.rule7.status == "NOT ASSESSABLE":
                    not_assessable_count += 1

            if f.rules.rule8:
                rules_evaluated += 1
                if f.rules.rule8.status == "PASS":
                    pass_count += 1
                elif f.rules.rule8.status == "NON-COMPLIANT":
                    non_compliant_count += 1
                elif f.rules.rule8.status == "NOT ASSESSABLE":
                    not_assessable_count += 1

        return InspectionSummary(
            total_declarations_evaluated=total_decls,
            detected_declarations_count=detected_count,
            not_visible_declarations_count=not_visible_count,
            rules_evaluated_count=rules_evaluated,
            pass_findings_count=pass_count,
            non_compliant_findings_count=non_compliant_count,
            not_assessable_findings_count=not_assessable_count,
        )
