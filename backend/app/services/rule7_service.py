"""Deterministic Rule 7 Numeral-Height Evaluation Service.

Evaluates statutory package numeral declarations against the Legal Metrology
(Packaged Commodities) Rules, 2011, Rule 7 and Second Schedule Table 1.

Characteristics:
- Completely deterministic: No LLMs, No OpenAI calls, No OpenCV.
- Handles unit normalization (e.g. kg -> g, l -> ml).
- Distinguishes net quantity sliding scales from general declaration numerals (MRP, Dates).
- Strictly excludes non-arithmetic / non-Rule 7 numbers (phone, PIN, barcode, etc.).
- Preserves measurement uncertainty (low quality -> indeterminate_low_confidence).
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from app.schemas.compliance import (
    ComparisonStatus,
    FontFormCategory,
    InspectionFindingStatus,
    Rule7Comparison,
    Rule7EvaluationRequest,
    Rule7Finding,
    Rule7MeasurementSummary,
    Rule7Requirement,
    Rule7ResponseData,
    Rule7Summary,
    RuleSourceMode,
    RuleType,
)
from app.schemas.extraction import LabelExtractionResult
from app.schemas.measurement import (
    MeasurementQuality,
    MeasurementResponseData,
    NumeralMeasurement,
)

logger = logging.getLogger(__name__)

# Mass units normalized to grams ('g')
WEIGHT_UNITS: Dict[str, float] = {
    "g": 1.0,
    "gm": 1.0,
    "gms": 1.0,
    "gram": 1.0,
    "grams": 1.0,
    "kg": 1000.0,
    "kgs": 1000.0,
    "kilogram": 1000.0,
    "kilograms": 1000.0,
    "mg": 0.001,
    "mgs": 0.001,
    "milligram": 0.001,
    "milligrams": 0.001,
}

# Volume units normalized to millilitres ('ml')
VOLUME_UNITS: Dict[str, float] = {
    "ml": 1.0,
    "mls": 1.0,
    "millilitre": 1.0,
    "millilitres": 1.0,
    "milliliter": 1.0,
    "milliliters": 1.0,
    "l": 1000.0,
    "lt": 1000.0,
    "ltr": 1000.0,
    "litre": 1000.0,
    "litres": 1000.0,
    "liter": 1000.0,
    "liters": 1000.0,
    "cl": 10.0,
    "centilitre": 10.0,
}

# Length / Area / Number units
LENGTH_UNITS = {"m", "metre", "metres", "meter", "meters", "cm", "centimetre", "mm"}
AREA_UNITS = {"sq m", "m²", "sq cm", "cm²"}
NUMBER_UNITS = {"n", "u", "unit", "units", "item", "items", "piece", "pieces", "count"}

NON_RULE7_SEMANTIC_ROLES = {"phone", "pin_code", "survey_number", "barcode", "license", "other"}


class Rule7ServiceError(Exception):
    """Exception raised for Rule 7 evaluation failures."""

    def __init__(self, code: str, message: str, status_code: int = 422):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class Rule7Service:
    """Deterministic Rule 7 numeral-height compliance evaluation engine."""

    def is_molded_category(self, font_category: FontFormCategory) -> bool:
        """Checks if font category is molded, perforated, embossed, formed, or blown."""
        return font_category.lower() in ("molded", "perforated", "embossed", "formed", "blown")

    def normalize_quantity(
        self,
        value: Optional[float],
        unit: Optional[str],
        quantity_type: Optional[str] = None,
    ) -> Tuple[Optional[float], Optional[str], Optional[str]]:
        """Normalizes quantity value and unit for range comparison.

        Returns:
            (normalized_value, normalized_unit, error_message)
            - normalized_unit is 'g' for weight or 'ml' for volume.
            - error_message is populated if unit or quantity is invalid or unsupported.
        """
        if value is None or value <= 0:
            return None, None, f"Invalid or non-positive declared quantity value: {value}"

        if not unit or not str(unit).strip():
            return None, None, "Declared quantity unit is missing"

        clean_unit = str(unit).strip().lower()

        # Check weight units
        if clean_unit in WEIGHT_UNITS:
            factor = WEIGHT_UNITS[clean_unit]
            norm_val = round(value * factor, 4)
            return norm_val, "g", None

        # Check volume units
        if clean_unit in VOLUME_UNITS:
            factor = VOLUME_UNITS[clean_unit]
            norm_val = round(value * factor, 4)
            return norm_val, "ml", None

        # Check length, area, number units
        if clean_unit in LENGTH_UNITS:
            return value, clean_unit, None
        if clean_unit in AREA_UNITS:
            return value, clean_unit, None
        if clean_unit in NUMBER_UNITS:
            return value, clean_unit, None

        return None, None, f"Unsupported unit '{unit}' for statutory quantity evaluation"

    def determine_weight_volume_threshold(
        self,
        normalized_value: float,
        font_category: FontFormCategory,
    ) -> Tuple[str, float]:
        """Calculates Rule 7 minimum numeral height for weight / volume declarations.

        Sliding scale:
        - Normal:
          - Below 200 g/ml -> 1.0 mm
          - 200–500 g/ml   -> 2.0 mm
          - Above 500 g/ml -> 4.0 mm
        - Molded / perforated / embossed / formed / blown:
          - Below 200 g/ml -> 2.0 mm
          - 200–500 g/ml   -> 4.0 mm
          - Above 500 g/ml -> 6.0 mm
        """
        molded = self.is_molded_category(font_category)

        if normalized_value < 200.0:
            condition = "below_200_g_ml"
            req_mm = 2.0 if molded else 1.0
        elif normalized_value <= 500.0:
            condition = "between_200_and_500_g_ml"
            req_mm = 4.0 if molded else 2.0
        else:
            condition = "above_500_g_ml"
            req_mm = 6.0 if molded else 4.0

        return condition, req_mm

    def determine_pdp_area_threshold(
        self,
        pdp_area_cm2: Optional[float],
        font_category: FontFormCategory,
        rule_prefix: str = "",
        rule_source_mode: RuleSourceMode = "sih_ps_26034",
    ) -> Tuple[str, float]:
        """Calculates minimum numeral height based on Principal Display Panel (PDP) area.

        Applicable for:
        - Declarations by length, area, or number
        - General statutory declarations (MRP, dates of mfg/pkd) under Second Schedule Table 1

        Scales:
        Mode: sih_ps_26034 (SIH Technical Specification mode):
        - Normal:
          - Below 100 cm²       -> 1.0 mm
          - 100–500 cm²         -> 2.0 mm
          - 500–2500 cm²        -> 4.0 mm
          - Above 2500 cm²      -> 6.0 mm
        - Molded / embossed / etc.:
          - Below 100 cm²       -> 2.0 mm
          - 100–500 cm²         -> 4.0 mm
          - 500–2500 cm²        -> 6.0 mm
          - Above 2500 cm²      -> 6.0 mm

        Mode: doca_statutory_2011 (Official Legal Metrology PCR 2011 Table-I mode):
        - Normal:
          - A <= 50 cm²         -> 1.0 mm
          - 50 < A <= 100 cm²   -> 1.5 mm
          - 100 < A <= 500 cm²  -> 2.5 mm
          - 500 < A <= 2500 cm² -> 4.0 mm
          - A > 2500 cm²        -> 6.0 mm
        - Blown / formed / molded / embossed / perforated:
          - A <= 50 cm²         -> 2.0 mm
          - 50 < A <= 100 cm²   -> 3.0 mm
          - 100 < A <= 500 cm²  -> 4.0 mm
          - 500 < A <= 2500 cm² -> 6.0 mm
          - A > 2500 cm²        -> 6.0 mm
        """
        molded = self.is_molded_category(font_category)
        prefix = f"{rule_prefix}_" if rule_prefix else ""

        if rule_source_mode == "doca_statutory_2011":
            if pdp_area_cm2 is None:
                condition = f"{prefix}baseline_pdp_lte_50_cm2"
                req_mm = 2.0 if molded else 1.0
                return condition, req_mm

            if pdp_area_cm2 <= 50.0:
                condition = f"{prefix}pdp_lte_50_cm2"
                req_mm = 2.0 if molded else 1.0
            elif pdp_area_cm2 <= 100.0:
                condition = f"{prefix}pdp_between_50_and_100_cm2"
                req_mm = 3.0 if molded else 1.5
            elif pdp_area_cm2 <= 500.0:
                condition = f"{prefix}pdp_between_100_and_500_cm2"
                req_mm = 4.0 if molded else 2.5
            elif pdp_area_cm2 <= 2500.0:
                condition = f"{prefix}pdp_between_500_and_2500_cm2"
                req_mm = 6.0 if molded else 4.0
            else:
                condition = f"{prefix}pdp_above_2500_cm2"
                req_mm = 6.0 if molded else 6.0

            return condition, req_mm

        # Default mode: sih_ps_26034
        if pdp_area_cm2 is None:
            # Baseline default when PDP area is not provided (< 100 cm²)
            condition = f"{prefix}baseline_pdp_below_100_cm2"
            req_mm = 2.0 if molded else 1.0
            return condition, req_mm

        if pdp_area_cm2 < 100.0:
            condition = f"{prefix}pdp_below_100_cm2"
            req_mm = 2.0 if molded else 1.0
        elif pdp_area_cm2 <= 500.0:
            condition = f"{prefix}pdp_between_100_and_500_cm2"
            req_mm = 4.0 if molded else 2.0
        elif pdp_area_cm2 <= 2500.0:
            condition = f"{prefix}pdp_between_500_and_2500_cm2"
            req_mm = 6.0 if molded else 4.0
        else:
            condition = f"{prefix}pdp_above_2500_cm2"
            req_mm = 6.0 if molded else 6.0

        return condition, req_mm

    def evaluate_declaration(
        self,
        field: str,
        raw_text: Optional[str],
        numeral_str: Optional[str],
        declared_value: Optional[float],
        declared_unit: Optional[str],
        quantity_type: Optional[str],
        measurement: Optional[NumeralMeasurement],
        font_category: FontFormCategory = "normal",
        pdp_area_cm2: Optional[float] = None,
        is_non_rule7: bool = False,
        rule_source_mode: RuleSourceMode = "sih_ps_26034",
    ) -> Rule7Finding:
        """Evaluates an individual declaration candidate against Rule 7."""
        rule_ref = (
            "Legal Metrology (Packaged Commodities) Rules, 2011 — Rule 7 / Second Schedule Table-I"
            if rule_source_mode == "doca_statutory_2011"
            else "SIH 2026 PS 26034 Technical Specification (Rule 7 Numeral-Height Engine)"
        )

        # 1. Non-Rule 7 exclusions (phone, PIN, barcode, license, lot numbers)
        if is_non_rule7 or any(role in field.lower() for role in ("phone", "pin", "barcode", "license", "survey")):
            return Rule7Finding(
                field=field,
                declared_text=raw_text,
                declared_numeral=numeral_str,
                declared_value=declared_value,
                declared_unit=declared_unit,
                measurement=None,
                requirement=Rule7Requirement(
                    rule_reference=rule_ref,
                    rule_source_mode=rule_source_mode,
                    rule_type="not_applicable",
                    condition="non_rule_7_exclusion",
                    required_height_mm=None,
                    font_category=font_category,
                ),
                comparison=Rule7Comparison(
                    status="not_applicable",
                    margin_mm=None,
                ),
                inspection_status="not_applicable",
                notes="Non-Rule 7 numeral: phone/PIN/barcode/license excluded from statutory numeral height requirements.",
            )

        # 2. Determine applicable rule requirement based on declaration field
        field_lower = field.lower()
        rule_type: RuleType
        condition: str
        req_height_mm: Optional[float] = None
        norm_val: Optional[float] = None
        norm_unit: Optional[str] = None
        rule_notes: Optional[str] = None

        if "net_quantity" in field_lower:
            # Net Quantity: check if weight/volume or other
            norm_val, norm_unit, err = self.normalize_quantity(declared_value, declared_unit, quantity_type)
            if err:
                rule_type = "unsupported"
                condition = "unsupported_quantity_specification"
                rule_notes = err
            elif norm_unit in ("g", "ml"):
                rule_type = "quantity_weight_volume"
                condition, req_height_mm = self.determine_weight_volume_threshold(norm_val, font_category)  # type: ignore
            else:
                # Length / Area / Number
                rule_type = "quantity_length_area_number"
                condition, req_height_mm = self.determine_pdp_area_threshold(
                    pdp_area_cm2, font_category, rule_prefix="quantity", rule_source_mode=rule_source_mode
                )
                rule_notes = f"Quantity declared by {quantity_type or 'count/dimension'}; evaluated against PDP area."
        elif "mrp" in field_lower or "price" in field_lower:
            # MRP / Retail Sale Price: non-quantity statutory numeral
            rule_type = "general_declaration"
            condition, req_height_mm = self.determine_pdp_area_threshold(
                pdp_area_cm2, font_category, rule_prefix="mrp_general", rule_source_mode=rule_source_mode
            )
            rule_notes = (
                f"Retail Sale Price (MRP) numeral evaluated under general declaration text minimum "
                f"({rule_ref})."
            )
        elif "date" in field_lower:
            # Date of manufacture / packing / import: non-quantity statutory numeral
            rule_type = "general_declaration"
            condition, req_height_mm = self.determine_pdp_area_threshold(
                pdp_area_cm2, font_category, rule_prefix="date_general", rule_source_mode=rule_source_mode
            )
            rule_notes = (
                f"Date numeral (month/year) evaluated under general declaration text minimum "
                f"({rule_ref})."
            )
        else:
            # Unknown / general statutory field
            rule_type = "general_declaration"
            condition, req_height_mm = self.determine_pdp_area_threshold(
                pdp_area_cm2, font_category, rule_prefix="general", rule_source_mode=rule_source_mode
            )
            rule_notes = f"Statutory field '{field}' evaluated under general declaration text minimum."

        requirement = Rule7Requirement(
            rule_reference=rule_ref,
            rule_source_mode=rule_source_mode,
            rule_type=rule_type,
            condition=condition,
            required_height_mm=req_height_mm,
            font_category=font_category,
            normalized_value=norm_val,
            normalized_unit=norm_unit,
        )

        # 3. Handle measurement summary
        meas_summary: Optional[Rule7MeasurementSummary] = None
        if measurement:
            meas_summary = Rule7MeasurementSummary(
                height_mm=measurement.numeral_height_mm,
                height_px=measurement.numeral_height_px,
                confidence=measurement.confidence,
                quality=measurement.measurement_quality,
            )

        # 4. Handle unsupported rule requirement
        if rule_type == "unsupported" or req_height_mm is None:
            return Rule7Finding(
                field=field,
                declared_text=raw_text,
                declared_numeral=numeral_str,
                declared_value=declared_value,
                declared_unit=declared_unit,
                measurement=meas_summary,
                requirement=requirement,
                comparison=Rule7Comparison(status="unsupported", margin_mm=None),
                inspection_status="unsupported",
                notes=rule_notes,
            )

        # 5. Handle missing or failed measurement
        if not meas_summary or meas_summary.height_mm is None or meas_summary.quality == "failed":
            notes_str = "No physical numeral measurement provided."
            if meas_summary and meas_summary.quality == "failed":
                notes_str = "Physical measurement failed during glyph segmentation."
            if rule_notes:
                notes_str = f"{rule_notes} {notes_str}".strip()

            return Rule7Finding(
                field=field,
                declared_text=raw_text,
                declared_numeral=numeral_str,
                declared_value=declared_value,
                declared_unit=declared_unit,
                measurement=meas_summary,
                requirement=requirement,
                comparison=Rule7Comparison(status="indeterminate", margin_mm=None),
                inspection_status="indeterminate_missing_measurement",
                notes=notes_str,
            )

        # 6. Mathematical comparison
        measured_h = meas_summary.height_mm
        margin_mm = round(measured_h - req_height_mm, 2)
        comp_status: ComparisonStatus = "above_threshold" if measured_h >= req_height_mm else "below_threshold"

        # 7. Uncertainty & Inspection finding status
        # Low measurement quality or low confidence prevents definitive statutory order
        diag_notes = []
        if rule_notes:
            diag_notes.append(rule_notes)

        if meas_summary.quality == "low" or (meas_summary.confidence is not None and meas_summary.confidence < 0.5):
            insp_status: InspectionFindingStatus = "indeterminate_low_confidence"
            diag_notes.append(
                f"Measurement quality is low (confidence: {meas_summary.confidence or 0.0:.2f}). "
                "Result cannot be treated as definitive legal conclusion without manual/higher-confidence recalibration."
            )
        else:
            if comp_status == "above_threshold":
                insp_status = "compliant"
            else:
                insp_status = "non_compliant"

        if measurement and measurement.notes:
            diag_notes.append(f"CV notes: {measurement.notes}")

        return Rule7Finding(
            field=field,
            declared_text=raw_text,
            declared_numeral=numeral_str,
            declared_value=declared_value,
            declared_unit=declared_unit,
            measurement=meas_summary,
            requirement=requirement,
            comparison=Rule7Comparison(
                status=comp_status,
                margin_mm=margin_mm,
            ),
            inspection_status=insp_status,
            notes=" ".join(diag_notes).strip() or None,
        )

    def parse_extraction_input(self, extraction_input: Any) -> Optional[LabelExtractionResult]:
        """Safely parses Module 1 extraction data from JSON string, dict, or model."""
        if extraction_input is None:
            return None
        if isinstance(extraction_input, LabelExtractionResult):
            return extraction_input

        if isinstance(extraction_input, str):
            trimmed = extraction_input.strip()
            if not trimmed:
                return None
            try:
                extraction_input = json.loads(trimmed)
            except Exception as exc:
                raise Rule7ServiceError(
                    code="MALFORMED_EXTRACTION_JSON",
                    message=f"Failed to parse extraction_json: {str(exc)}",
                    status_code=400,
                ) from exc

        if isinstance(extraction_input, dict):
            # Unwrap envelope if present
            if "data" in extraction_input and isinstance(extraction_input["data"], dict):
                extraction_input = extraction_input["data"]
            try:
                return LabelExtractionResult.model_validate(extraction_input)
            except Exception as exc:
                raise Rule7ServiceError(
                    code="INVALID_EXTRACTION_SCHEMA",
                    message=f"extraction data does not conform to Module 1 schema: {str(exc)}",
                    status_code=422,
                ) from exc

        return None

    def parse_measurements_input(self, measurements_input: Any) -> List[NumeralMeasurement]:
        """Safely parses Module 2A measurements from JSON string, dict, model, or list."""
        if measurements_input is None:
            return []

        if isinstance(measurements_input, MeasurementResponseData):
            return measurements_input.measurements

        if isinstance(measurements_input, str):
            trimmed = measurements_input.strip()
            if not trimmed:
                return []
            try:
                measurements_input = json.loads(trimmed)
            except Exception as exc:
                raise Rule7ServiceError(
                    code="MALFORMED_MEASUREMENT_JSON",
                    message=f"Failed to parse measurements JSON: {str(exc)}",
                    status_code=400,
                ) from exc

        if isinstance(measurements_input, dict):
            # Unwrap envelope if present
            if "data" in measurements_input and isinstance(measurements_input["data"], dict):
                measurements_input = measurements_input["data"]
            if "measurements" in measurements_input and isinstance(measurements_input["measurements"], list):
                measurements_input = measurements_input["measurements"]

        if isinstance(measurements_input, list):
            parsed_measurements = []
            for item in measurements_input:
                if isinstance(item, NumeralMeasurement):
                    parsed_measurements.append(item)
                elif isinstance(item, dict):
                    try:
                        parsed_measurements.append(NumeralMeasurement.model_validate(item))
                    except Exception as exc:
                        logger.warning("Skipping invalid NumeralMeasurement item: %s", exc)
            return parsed_measurements

        return []

    def evaluate(
        self,
        extraction: Optional[Union[LabelExtractionResult, Dict[str, Any], str]] = None,
        measurements: Optional[Union[MeasurementResponseData, List[NumeralMeasurement], Dict[str, Any], str]] = None,
        font_category: FontFormCategory = "normal",
        pdp_area_cm2: Optional[float] = None,
        rule_source_mode: RuleSourceMode = "sih_ps_26034",
    ) -> Rule7ResponseData:
        """Executes complete Rule 7 compliance evaluation across all statutory declarations."""
        parsed_extraction = self.parse_extraction_input(extraction)
        parsed_measurements = self.parse_measurements_input(measurements)

        # Index measurements by field name for fast lookup
        # e.g. 'net_quantity', 'mrp', 'dates[0]', 'dates[1]'
        meas_by_field: Dict[str, NumeralMeasurement] = {}
        for m in parsed_measurements:
            meas_by_field[m.field.strip().lower()] = m

        # If pdp_area_cm2 was not supplied, see if calibration dimensions exist in measurements
        if pdp_area_cm2 is None and isinstance(measurements, dict):
            calib = measurements.get("calibration", {})
            if isinstance(calib, dict):
                w_mm = calib.get("package_width_mm")
                h_mm = calib.get("package_height_mm")
                if w_mm and h_mm and w_mm > 0 and h_mm > 0:
                    pdp_area_cm2 = round((w_mm * h_mm) / 100.0, 2)

        findings: List[Rule7Finding] = []

        if parsed_extraction:
            # 1. Net Quantity
            net_qty = parsed_extraction.net_quantity
            if net_qty and net_qty.status == "present":
                meas = meas_by_field.get("net_quantity")
                val_str = str(int(net_qty.value)) if (net_qty.value is not None and net_qty.value.is_integer()) else str(net_qty.value or "")
                finding = self.evaluate_declaration(
                    field="net_quantity",
                    raw_text=net_qty.raw_text,
                    numeral_str=meas.numeral if meas else val_str,
                    declared_value=net_qty.value,
                    declared_unit=net_qty.unit,
                    quantity_type=net_qty.quantity_type,
                    measurement=meas,
                    font_category=font_category,
                    pdp_area_cm2=pdp_area_cm2,
                    rule_source_mode=rule_source_mode,
                )
                findings.append(finding)

            # 2. MRP / Retail Sale Price
            mrp = parsed_extraction.mrp
            if mrp and mrp.status == "present":
                meas = meas_by_field.get("mrp")
                val_str = f"{mrp.value:.2f}" if (mrp.value is not None and mrp.value > 0) else str(mrp.value or "")
                finding = self.evaluate_declaration(
                    field="mrp",
                    raw_text=mrp.raw_text,
                    numeral_str=meas.numeral if meas else val_str,
                    declared_value=mrp.value,
                    declared_unit=mrp.currency,
                    quantity_type=None,
                    measurement=meas,
                    font_category=font_category,
                    pdp_area_cm2=pdp_area_cm2,
                    rule_source_mode=rule_source_mode,
                )
                findings.append(finding)

            # 3. Dates (Manufacture, Packing, Import)
            for idx, dt in enumerate(parsed_extraction.dates):
                if dt and dt.status == "present":
                    field_key = f"dates[{idx}]"
                    meas = meas_by_field.get(field_key) or meas_by_field.get(f"date[{idx}]") or meas_by_field.get("dates")
                    date_numeral = f"{dt.month or ''}/{dt.year or ''}".strip("/") or dt.raw_text
                    finding = self.evaluate_declaration(
                        field=field_key,
                        raw_text=dt.raw_text,
                        numeral_str=meas.numeral if meas else date_numeral,
                        declared_value=None,
                        declared_unit=None,
                        quantity_type=None,
                        measurement=meas,
                        font_category=font_category,
                        pdp_area_cm2=pdp_area_cm2,
                        rule_source_mode=rule_source_mode,
                    )
                    findings.append(finding)

            # 4. Explicitly mark non-Rule 7 numeric information if present
            for idx, num_info in enumerate(parsed_extraction.additional_numeric_information):
                field_key = f"additional_numeric_information[{idx}]"
                finding = self.evaluate_declaration(
                    field=f"{field_key}_{num_info.semantic_role}",
                    raw_text=num_info.raw_text,
                    numeral_str=num_info.value,
                    declared_value=None,
                    declared_unit=None,
                    quantity_type=None,
                    measurement=None,
                    font_category=font_category,
                    pdp_area_cm2=pdp_area_cm2,
                    is_non_rule7=True,
                    rule_source_mode=rule_source_mode,
                )
                findings.append(finding)

        else:
            # Fallback when only measurements are provided without full Module 1 extraction
            for m in parsed_measurements:
                f_name = m.field.strip().lower()
                is_excluded = any(role in f_name for role in ("phone", "pin", "barcode", "license", "survey"))
                finding = self.evaluate_declaration(
                    field=m.field,
                    raw_text=m.raw_text,
                    numeral_str=m.numeral,
                    declared_value=self._extract_first_number(m.numeral),
                    declared_unit=self._guess_unit(m.raw_text),
                    quantity_type="weight" if "g" in m.raw_text.lower() else None,
                    measurement=m,
                    font_category=font_category,
                    pdp_area_cm2=pdp_area_cm2,
                    is_non_rule7=is_excluded,
                    rule_source_mode=rule_source_mode,
                )
                findings.append(finding)

        # Aggregate summary statistics
        summary = Rule7Summary(
            rule_source_mode=rule_source_mode,
            total_evaluated=len(findings),
            compliant_count=sum(1 for f in findings if f.inspection_status == "compliant"),
            non_compliant_count=sum(1 for f in findings if f.inspection_status == "non_compliant"),
            indeterminate_count=sum(1 for f in findings if f.inspection_status.startswith("indeterminate")),
            unsupported_count=sum(1 for f in findings if f.inspection_status == "unsupported"),
            not_applicable_count=sum(1 for f in findings if f.inspection_status == "not_applicable"),
        )

        return Rule7ResponseData(
            summary=summary,
            findings=findings,
        )

    def _extract_first_number(self, text: str) -> Optional[float]:
        """Extracts first numeric float from string."""
        if not text:
            return None
        match = re.search(r"[-+]?\d*\.?\d+", text)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
        return None

    def _guess_unit(self, text: str) -> Optional[str]:
        """Heuristically extracts unit from raw text."""
        if not text:
            return None
        for u in ("kg", "g", "gm", "ml", "l", "ltr"):
            if re.search(rf"\b{u}\b", text, re.IGNORECASE):
                return u
        return None
