"""Deterministic Rule 6 Declaration Visibility & Completeness Service.

Evaluates whether mandatory retail packaging declarations required under
Rule 6 of the Legal Metrology (Packaged Commodities) Rules, 2011 and SIH 2026 PS 26034
are detected in the supplied product image/PDP.

Mandatory Declarations Checked:
1. Manufacturer / Packer / Importer (Rule 6(1)(a))
2. Generic Name of the Commodity (Rule 6(1)(b))
3. Retail Sale Price (MRP) (Rule 6(1)(c))
4. Date of Manufacture / Packing / Import (Rule 6(1)(d))
5. Net Quantity (Rule 6(1)(e))
6. Consumer Care Details (Rule 6(1)(f))

Semantic Clarity:
"Not visible in supplied image" is strictly distinguished from "legally missing".
If a declaration is not detected in the provided image, its status is 'not_visible'
(Display: "Not visible in supplied image"). It does NOT issue a legal non-compliance conclusion,
because the declaration may exist on another packaging face that was not submitted in the scan.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from app.schemas.extraction import LabelExtractionResult, SemanticExtraction
from app.schemas.rule6 import (
    DECLARATION_DISPLAY_NAMES,
    DECLARATION_RULE_SUBSECTIONS,
    STATUS_TO_DISPLAY,
    Rule6DeclarationType,
    Rule6Finding,
    Rule6ResponseData,
    Rule6Status,
    Rule6Summary,
)

logger = logging.getLogger(__name__)


class Rule6ServiceError(Exception):
    """Exception raised for Rule 6 service processing errors."""

    def __init__(self, code: str, message: str, status_code: int = 422):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class Rule6Service:
    """Deterministic Rule 6 declaration visibility evaluation engine."""

    def _parse_extraction_payload(self, raw_input: Any) -> Optional[LabelExtractionResult]:
        """Parses and validates Module 1 extraction payload."""
        if raw_input is None or raw_input == {}:
            return None
        if isinstance(raw_input, LabelExtractionResult):
            return raw_input
        if isinstance(raw_input, str):
            trimmed = raw_input.strip()
            if not trimmed or trimmed == "{}":
                return None
            try:
                raw_input = json.loads(trimmed)
            except Exception as exc:
                raise Rule6ServiceError(
                    code="INVALID_EXTRACTION_JSON",
                    message=f"Failed to parse extraction JSON string: {exc}",
                    status_code=422,
                ) from exc

        if isinstance(raw_input, dict):
            if not raw_input:
                return None
            if "data" in raw_input and isinstance(raw_input["data"], dict):
                raw_input = raw_input["data"]
            try:
                return LabelExtractionResult.model_validate(raw_input)
            except Exception as exc:
                raise Rule6ServiceError(
                    code="INVALID_EXTRACTION_SCHEMA",
                    message=f"Extraction data does not conform to Module 1 schema: {exc}",
                    status_code=422,
                ) from exc

        return None

    def evaluate(
        self,
        extraction: Optional[Any],
        package_type: Optional[str] = None,
    ) -> Rule6ResponseData:
        """Evaluates visibility of all six Rule 6 mandatory declarations.

        Args:
            extraction: Module 1 extraction result dict, envelope, string, or model.
            package_type: Optional package classification override.

        Returns:
            Rule6ResponseData with structured findings and summary.
        """
        parsed_extraction = self._parse_extraction_payload(extraction)
        sem: Optional[SemanticExtraction] = (
            parsed_extraction.semantic_extraction if parsed_extraction else None
        )

        findings: List[Rule6Finding] = []

        # 1. Manufacturer / Packer / Importer (Rule 6(1)(a))
        mfr_finding = self._evaluate_manufacturer(sem)
        findings.append(mfr_finding)

        # 2. Generic Name (Rule 6(1)(b))
        gen_finding = self._evaluate_generic_name(sem)
        findings.append(gen_finding)

        # 3. Retail Sale Price (MRP) (Rule 6(1)(c))
        mrp_finding = self._evaluate_mrp(sem)
        findings.append(mrp_finding)

        # 4. Date of Manufacture / Packing / Import (Rule 6(1)(d))
        date_finding = self._evaluate_date(sem)
        findings.append(date_finding)

        # 5. Net Quantity (Rule 6(1)(e))
        qty_finding = self._evaluate_net_quantity(sem)
        findings.append(qty_finding)

        # 6. Consumer Care Details (Rule 6(1)(f))
        care_finding = self._evaluate_consumer_care(sem)
        findings.append(care_finding)

        detected_count = sum(1 for f in findings if f.status == "present")
        not_visible_count = sum(1 for f in findings if f.status == "not_visible")
        not_applicable_count = sum(1 for f in findings if f.status == "not_applicable")

        summary = Rule6Summary(
            total_declarations=len(findings),
            detected_count=detected_count,
            not_visible_count=not_visible_count,
            not_applicable_count=not_applicable_count,
        )

        return Rule6ResponseData(
            rule="Rule 6 — Declaration Visibility",
            summary=summary,
            findings=findings,
        )

    def _build_finding(
        self,
        field: Rule6DeclarationType,
        status: Rule6Status,
        raw_text: Optional[str] = None,
        confidence: Optional[float] = None,
        bbox: Optional[List[float]] = None,
        notes: Optional[str] = None,
    ) -> Rule6Finding:
        """Helper to construct a standardized Rule6Finding."""
        rule_sub = DECLARATION_RULE_SUBSECTIONS[field]
        rule_ref = f"Legal Metrology (Packaged Commodities) Rules, 2011 — {rule_sub}"

        return Rule6Finding(
            field=field,
            display_name=DECLARATION_DISPLAY_NAMES[field],
            rule_reference=rule_ref,
            status=status,
            display_status=STATUS_TO_DISPLAY[status],
            raw_text=raw_text,
            confidence=round(confidence, 2) if confidence is not None else None,
            bbox=bbox,
            notes=notes,
        )

    def _evaluate_manufacturer(self, sem: Optional[SemanticExtraction]) -> Rule6Finding:
        """Evaluates declaration of manufacturer, packer, or importer (Rule 6(1)(a))."""
        field: Rule6DeclarationType = "manufacturer_packer_importer"
        if not sem or not sem.manufacturer:
            return self._build_finding(
                field=field,
                status="not_visible",
                notes="Manufacturer / Packer / Importer details not detected in supplied image.",
            )

        mfr = sem.manufacturer
        entities = mfr.entities or []

        # Check if visibly present with entities
        if mfr.status == "present" and len(entities) > 0:
            first_entity = entities[0]
            raw_text = "; ".join([e.raw_text for e in entities if e.raw_text])
            if not raw_text:
                raw_text = f"{first_entity.name}, {first_entity.address}" if first_entity.address else first_entity.name

            conf = max([e.confidence for e in entities], default=0.95)
            bbox = first_entity.bbox if first_entity.bbox else None

            roles = list({e.role for e in entities if e.role and e.role != "unknown"})
            roles_desc = f" ({', '.join(roles)})" if roles else ""
            notes = f"Declared entity name and address detected{roles_desc}."

            return self._build_finding(
                field=field,
                status="present",
                raw_text=raw_text,
                confidence=conf,
                bbox=bbox,
                notes=notes,
            )

        return self._build_finding(
            field=field,
            status="not_visible",
            notes="Manufacturer / Packer / Importer details not detected in supplied image.",
        )

    def _evaluate_generic_name(self, sem: Optional[SemanticExtraction]) -> Rule6Finding:
        """Evaluates generic / common name of the commodity (Rule 6(1)(b))."""
        field: Rule6DeclarationType = "generic_name"
        if not sem or not sem.generic_name:
            return self._build_finding(
                field=field,
                status="not_visible",
                notes="Generic commodity name not detected in supplied image.",
            )

        gn = sem.generic_name
        if gn.status == "present" and (gn.value or gn.raw_text):
            raw_text = gn.raw_text or gn.value
            return self._build_finding(
                field=field,
                status="present",
                raw_text=raw_text,
                confidence=gn.confidence,
                bbox=gn.bbox,
                notes=f"Generic name declared: '{gn.value or gn.raw_text}'.",
            )

        return self._build_finding(
            field=field,
            status="not_visible",
            notes="Generic commodity name not detected in supplied image.",
        )

    def _evaluate_mrp(self, sem: Optional[SemanticExtraction]) -> Rule6Finding:
        """Evaluates Retail Sale Price (MRP) declaration (Rule 6(1)(c))."""
        field: Rule6DeclarationType = "mrp"
        if not sem or not sem.mrp:
            return self._build_finding(
                field=field,
                status="not_visible",
                notes="Retail Sale Price (MRP) declaration not detected in supplied image.",
            )

        mrp = sem.mrp
        if mrp.status == "present" and (mrp.value is not None or mrp.raw_text):
            curr = mrp.currency or "₹"
            formatted_val = f"{curr} {mrp.value:.2f}" if mrp.value is not None else None
            notes = f"Declared MRP: {formatted_val} (Inclusive of all taxes)." if formatted_val else "MRP detected."

            return self._build_finding(
                field=field,
                status="present",
                raw_text=mrp.raw_text or formatted_val,
                confidence=mrp.confidence,
                bbox=mrp.bbox,
                notes=notes,
            )

        return self._build_finding(
            field=field,
            status="not_visible",
            notes="Retail Sale Price (MRP) declaration not detected in supplied image.",
        )

    def _evaluate_date(self, sem: Optional[SemanticExtraction]) -> Rule6Finding:
        """Evaluates Date of Manufacture / Packing / Import declaration (Rule 6(1)(d))."""
        field: Rule6DeclarationType = "date_of_manufacture_packing_import"
        if not sem or not sem.dates:
            return self._build_finding(
                field=field,
                status="not_visible",
                notes="Date of manufacture, packing, or import not detected in supplied image.",
            )

        applicable_dates = [
            d for d in sem.dates
            if d.status == "present" and d.date_type in ("manufacture", "packing", "import")
        ]
        # Fallback to any date if date_type was general/unknown
        if not applicable_dates:
            applicable_dates = [d for d in sem.dates if d.status == "present"]

        if applicable_dates:
            first_date = applicable_dates[0]
            raw_text = "; ".join([d.raw_text for d in applicable_dates if d.raw_text])
            conf = max([d.confidence for d in applicable_dates], default=0.90)
            bbox = first_date.bbox

            types = list({d.date_type for d in applicable_dates if d.date_type})
            type_str = f" ({', '.join(types)})" if types else ""
            notes = f"Declared date detected{type_str}."

            return self._build_finding(
                field=field,
                status="present",
                raw_text=raw_text,
                confidence=conf,
                bbox=bbox,
                notes=notes,
            )

        return self._build_finding(
            field=field,
            status="not_visible",
            notes="Date of manufacture, packing, or import not detected in supplied image.",
        )

    def _evaluate_net_quantity(self, sem: Optional[SemanticExtraction]) -> Rule6Finding:
        """Evaluates Net Quantity declaration (Rule 6(1)(e))."""
        field: Rule6DeclarationType = "net_quantity"
        if not sem or not sem.net_quantity:
            return self._build_finding(
                field=field,
                status="not_visible",
                notes="Net Quantity declaration not detected in supplied image.",
            )

        nq = sem.net_quantity
        if nq.status == "present" and (nq.value is not None or nq.raw_text):
            qty_str = f"{nq.value} {nq.unit or ''}".strip() if nq.value is not None else ""
            notes = f"Declared net quantity: {qty_str}." if qty_str else "Net quantity detected."

            return self._build_finding(
                field=field,
                status="present",
                raw_text=nq.raw_text or qty_str,
                confidence=nq.confidence,
                bbox=nq.bbox,
                notes=notes,
            )

        return self._build_finding(
            field=field,
            status="not_visible",
            notes="Net Quantity declaration not detected in supplied image.",
        )

    def _evaluate_consumer_care(self, sem: Optional[SemanticExtraction]) -> Rule6Finding:
        """Evaluates Consumer Care Details declaration (Rule 6(1)(f))."""
        field: Rule6DeclarationType = "consumer_care"
        if not sem or not sem.consumer_care:
            return self._build_finding(
                field=field,
                status="not_visible",
                notes="Consumer Care Details not detected in supplied image.",
            )

        cc = sem.consumer_care
        subfields_present = []
        if cc.phone and cc.phone.status == "present" and cc.phone.value:
            subfields_present.append(f"phone: {cc.phone.value}")
        if cc.email and cc.email.status == "present" and cc.email.value:
            subfields_present.append(f"email: {cc.email.value}")
        if cc.address and cc.address.status == "present" and cc.address.value:
            subfields_present.append(f"address: {cc.address.value}")
        if cc.name and cc.name.status == "present" and cc.name.value:
            subfields_present.append(f"name: {cc.name.value}")

        if cc.status == "present" or len(subfields_present) > 0 or cc.raw_text:
            raw_text = cc.raw_text or "; ".join(subfields_present)
            conf = cc.confidence if cc.confidence > 0 else 0.90
            notes = (
                f"Consumer care contact details detected: {', '.join(subfields_present)}."
                if subfields_present else "Consumer care details detected."
            )

            return self._build_finding(
                field=field,
                status="present",
                raw_text=raw_text,
                confidence=conf,
                bbox=cc.bbox,
                notes=notes,
            )

        return self._build_finding(
            field=field,
            status="not_visible",
            notes="Consumer Care Details not detected in supplied image.",
        )
