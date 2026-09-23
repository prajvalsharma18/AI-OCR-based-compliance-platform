"""Unit and integration tests for Unified Findings / Unified Inspection JSON.

SIH 2026 PS 26034: Automated Compliance Checker for Packaged Commodities

Covers all 34 required test conditions from Step 14:
1. Pydantic model input
2. dictionary input
3. JSON string input
4. data-envelope input
5. missing optional module input
6. Rule 6 detected declaration
7. Rule 6 NOT VISIBLE declaration
8. Consumer Care NOT VISIBLE
9. NOT VISIBLE never becomes NON-COMPLIANT
10. MRP + Rule 7 NON-COMPLIANT
11. Net Quantity + Rule 7 PASS
12. Net Quantity + Rule 8 PASS
13. Rule 7 NOT ASSESSABLE
14. Rule 8 NOT ASSESSABLE
15. measured height preserved
16. required height preserved
17. height margin preserved
18. measurement quality preserved
19. confidence preserved
20. Rule 8 directional measurements preserved
21. MRP Rule 6 + Rule 7 merge
22. Net Quantity Rule 6 + Rule 7 + Rule 8 merge
23. no duplicate declarations
24. manufacture/use_by semantic separation
25. date mapping independent of array position
26. evidence preservation
27. correct detected count
28. correct not-visible count
29. correct PASS rule count
30. correct NON-COMPLIANT rule count
31. correct NOT ASSESSABLE rule count
32. POST /api/v1/compliance/inspection integration test
33. response matches Pydantic schema
34. upstream rule results are not recalculated
"""

import json
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.compliance import Rule7ResponseData
from app.schemas.extraction import LabelExtractionResult
from app.schemas.inspection import (
    InspectionResponse,
    InspectionResponseData,
    UnifiedDeclarationFinding,
)
from app.schemas.measurement import MeasurementResponseData
from app.schemas.rule6 import Rule6ResponseData
from app.schemas.rule8 import Rule8ResponseData
from app.services.inspection_service import InspectionService, InspectionServiceError


@pytest.fixture
def client():
    """Test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def service():
    """Instance of InspectionService."""
    return InspectionService()


@pytest.fixture
def sample_extraction():
    """Realistic Module 1 extraction payload (Lay's pattern) strictly conforming to schema."""
    return {
        "image": {"width": 1200, "height": 1600, "coordinate_system": "normalized_0_1"},
        "raw_text_blocks": [
            {"text": "Lay's Classic Salted", "bbox": [0.1, 0.1, 0.9, 0.2]}
        ],
        "semantic_extraction": {
            "image_quality": {"overall": "good", "notes": "Sharp packaging image"},
            "package": {
                "package_type": "retail",
                "brand_name": "Lay's",
                "generic_name": "Potato Chips",
                "food_status": "food",
                "retail_unit_count": None,
                "evidence": ["Front label"],
            },
            "manufacturer": {
                "status": "present",
                "entities": [
                    {
                        "role": "manufacturer",
                        "name": "PepsiCo India Holdings Pvt. Ltd.",
                        "address": "Village Channo, Patiala 147001",
                        "raw_text": "Mfg by PepsiCo India Holdings Pvt. Ltd., Village Channo, Patiala 147001",
                        "confidence": 0.98,
                        "bbox": [0.10, 0.70, 0.90, 0.75],
                    }
                ],
            },
            "generic_name": {
                "status": "present",
                "value": "Potato Chips",
                "raw_text": "POTATO CHIPS",
                "confidence": 0.95,
                "bbox": [0.25, 0.30, 0.75, 0.35],
            },
            "mrp": {
                "status": "present",
                "raw_text": "MRP Rs. 10.00 (Incl. of all taxes)",
                "value": 10.0,
                "currency": "₹",
                "confidence": 0.96,
                "bbox": [0.60, 0.80, 0.85, 0.85],
                "numeral_region": [0.65, 0.81, 0.72, 0.84],
            },
            "dates": [
                {
                    "date_type": "manufacture",
                    "status": "present",
                    "raw_text": "MFD: 26/07/26",
                    "month": "07",
                    "year": 2026,
                    "confidence": 0.94,
                    "bbox": [0.10, 0.80, 0.40, 0.83],
                    "numeral_region": [0.18, 0.81, 0.38, 0.83],
                },
                {
                    "date_type": "use_by",
                    "status": "present",
                    "raw_text": "USE BY: 08/12/26",
                    "month": "12",
                    "year": 2026,
                    "confidence": 0.93,
                    "bbox": [0.10, 0.84, 0.40, 0.87],
                    "numeral_region": [0.20, 0.85, 0.38, 0.87],
                },
            ],
            "net_quantity": {
                "status": "present",
                "raw_text": "Net Wt. 42 g",
                "value": 42.0,
                "unit": "g",
                "confidence": 0.97,
                "bbox": [0.15, 0.90, 0.40, 0.95],
                "numeral_region": [0.22, 0.91, 0.30, 0.94],
            },
            "consumer_care": {
                "status": "missing",
                "raw_text": None,
                "confidence": 0.0,
                "bbox": None,
            },
            "stickers": [],
            "additional_numeric_information": [],
            "other_visible_text": [],
        },
    }


@pytest.fixture
def sample_measurements():
    """Module 2A calibrated measurements payload."""
    return {
        "image": {"width": 1200, "height": 1600},
        "calibration": {
            "method": "package_dimensions",
            "package_width_mm": 150.0,
            "package_height_mm": 200.0,
            "package_bbox_px": [100, 100, 1100, 1500],
            "package_width_px": 1000.0,
            "package_height_px": 1400.0,
            "width_mm_per_px": 0.15,
            "height_mm_per_px": 0.1428,
            "confidence": 0.98,
            "quality": "good",
        },
        "measurements": [
            {
                "field": "mrp",
                "raw_text": "MRP Rs. 10.00",
                "numeral": "10",
                "semantic_region_normalized": [0.65, 0.81, 0.72, 0.84],
                "numeral_bbox_px": [780, 1296, 864, 1344],
                "numeral_height_px": 12,
                "numeral_height_mm": 1.65,
                "confidence": 0.92,
                "measurement_quality": "good",
                "notes": "Clear numeral",
            },
            {
                "field": "net_quantity",
                "raw_text": "Net Wt. 42 g",
                "numeral": "42",
                "semantic_region_normalized": [0.22, 0.91, 0.30, 0.94],
                "numeral_bbox_px": [264, 1456, 360, 1504],
                "numeral_height_px": 18,
                "numeral_height_mm": 2.50,
                "confidence": 0.95,
                "measurement_quality": "good",
                "notes": "Sharp quantity numeral",
            },
            {
                "field": "dates[0]",
                "raw_text": "MFD: 26/07/26",
                "numeral": "26/07/26",
                "semantic_region_normalized": [0.18, 0.81, 0.38, 0.83],
                "numeral_bbox_px": [216, 1296, 456, 1328],
                "numeral_height_px": 14,
                "numeral_height_mm": 2.00,
                "confidence": 0.90,
                "measurement_quality": "good",
                "notes": "Clear mfd date",
            },
            {
                "field": "dates[1]",
                "raw_text": "USE BY: 08/12/26",
                "numeral": "08/12/26",
                "semantic_region_normalized": [0.20, 0.85, 0.38, 0.87],
                "numeral_bbox_px": [240, 1360, 456, 1392],
                "numeral_height_px": 14,
                "numeral_height_mm": 2.00,
                "confidence": 0.89,
                "measurement_quality": "good",
                "notes": "Clear use-by date",
            },
        ],
    }


@pytest.fixture
def sample_rule6():
    """Module 3C Rule 6 visibility evaluation result."""
    return {
        "rule": "Rule 6 — Declaration Visibility",
        "summary": {
            "total_declarations": 6,
            "detected_count": 5,
            "not_visible_count": 1,
            "not_applicable_count": 0,
        },
        "findings": [
            {
                "field": "manufacturer_packer_importer",
                "display_name": "Manufacturer / Packer / Importer",
                "rule_reference": "Legal Metrology (Packaged Commodities) Rules, 2011 — Rule 6(1)(a)",
                "status": "present",
                "display_status": "Detected",
                "raw_text": "Mfg by PepsiCo India Holdings Pvt. Ltd., Village Channo, Patiala 147001",
                "confidence": 0.98,
                "bbox": [0.10, 0.70, 0.90, 0.75],
                "notes": "Declared entity name and address detected.",
            },
            {
                "field": "generic_name",
                "display_name": "Generic Name",
                "rule_reference": "Legal Metrology (Packaged Commodities) Rules, 2011 — Rule 6(1)(b)",
                "status": "present",
                "display_status": "Detected",
                "raw_text": "POTATO CHIPS",
                "confidence": 0.95,
                "bbox": [0.25, 0.30, 0.75, 0.35],
                "notes": "Generic name declared: 'Potato Chips'.",
            },
            {
                "field": "mrp",
                "display_name": "Retail Sale Price (MRP)",
                "rule_reference": "Legal Metrology (Packaged Commodities) Rules, 2011 — Rule 6(1)(c)",
                "status": "present",
                "display_status": "Detected",
                "raw_text": "MRP Rs. 10.00 (Incl. of all taxes)",
                "confidence": 0.96,
                "bbox": [0.60, 0.80, 0.85, 0.85],
                "notes": "Declared MRP: ₹ 10.00 (Inclusive of all taxes).",
            },
            {
                "field": "date_of_manufacture_packing_import",
                "display_name": "Manufacturing / Packing / Import Date",
                "rule_reference": "Legal Metrology (Packaged Commodities) Rules, 2011 — Rule 6(1)(d)",
                "status": "present",
                "display_status": "Detected",
                "raw_text": "MFD: 26/07/26; USE BY: 08/12/26",
                "confidence": 0.94,
                "bbox": [0.10, 0.80, 0.40, 0.83],
                "notes": "Declared date detected (manufacture, use_by).",
            },
            {
                "field": "net_quantity",
                "display_name": "Net Quantity",
                "rule_reference": "Legal Metrology (Packaged Commodities) Rules, 2011 — Rule 6(1)(e)",
                "status": "present",
                "display_status": "Detected",
                "raw_text": "Net Wt. 42 g",
                "confidence": 0.97,
                "bbox": [0.15, 0.90, 0.40, 0.95],
                "notes": "Declared net quantity: 42.0 g.",
            },
            {
                "field": "consumer_care",
                "display_name": "Consumer Care Details",
                "rule_reference": "Legal Metrology (Packaged Commodities) Rules, 2011 — Rule 6(1)(f)",
                "status": "not_visible",
                "display_status": "Not visible in supplied image",
                "raw_text": None,
                "confidence": None,
                "bbox": None,
                "notes": "Consumer care details not detected in supplied image.",
            },
        ],
        "disclaimer": "Visibility evaluation in supplied image only.",
    }


@pytest.fixture
def sample_rule7():
    """Module 3A Rule 7 numeral-height evaluation result."""
    return {
        "summary": {
            "rule_source_mode": "sih_ps_26034",
            "total_evaluated": 4,
            "compliant_count": 3,
            "non_compliant_count": 1,
            "indeterminate_count": 0,
            "unsupported_count": 0,
            "not_applicable_count": 0,
        },
        "findings": [
            {
                "field": "mrp",
                "declared_text": "MRP Rs. 10.00",
                "declared_numeral": "10",
                "declared_value": 10.0,
                "declared_unit": "₹",
                "measurement": {
                    "height_mm": 1.65,
                    "height_px": 12,
                    "confidence": 0.92,
                    "quality": "good",
                },
                "requirement": {
                    "rule_reference": "Legal Metrology (Packaged Commodities) Rules, 2011 — Rule 7 (Second Schedule Table 1)",
                    "rule_source_mode": "sih_ps_26034",
                    "rule_type": "general_declaration",
                    "condition": "pdp_area_200_to_500_cm2",
                    "required_height_mm": 2.0,
                    "font_category": "normal",
                    "normalized_value": None,
                    "normalized_unit": None,
                },
                "comparison": {
                    "status": "below_threshold",
                    "margin_mm": -0.35,
                },
                "inspection_status": "non_compliant",
                "notes": "Measured height (1.65 mm) is below required height (2.00 mm).",
            },
            {
                "field": "net_quantity",
                "declared_text": "Net Wt. 42 g",
                "declared_numeral": "42",
                "declared_value": 42.0,
                "declared_unit": "g",
                "measurement": {
                    "height_mm": 2.50,
                    "height_px": 18,
                    "confidence": 0.95,
                    "quality": "good",
                },
                "requirement": {
                    "rule_reference": "Legal Metrology (Packaged Commodities) Rules, 2011 — Rule 7 (Second Schedule Table 1)",
                    "rule_source_mode": "sih_ps_26034",
                    "rule_type": "quantity_weight_volume",
                    "condition": "below_200_g_ml",
                    "required_height_mm": 1.0,
                    "font_category": "normal",
                    "normalized_value": 42.0,
                    "normalized_unit": "g",
                },
                "comparison": {
                    "status": "above_threshold",
                    "margin_mm": 1.50,
                },
                "inspection_status": "compliant",
                "notes": "Measured height (2.50 mm) meets minimum required height (1.00 mm).",
            },
            {
                "field": "dates[0]",
                "declared_text": "MFD: 26/07/26",
                "declared_numeral": "26/07/26",
                "declared_value": None,
                "declared_unit": None,
                "measurement": {
                    "height_mm": 2.00,
                    "height_px": 14,
                    "confidence": 0.90,
                    "quality": "good",
                },
                "requirement": {
                    "rule_reference": "Legal Metrology (Packaged Commodities) Rules, 2011 — Rule 7 (Second Schedule Table 1)",
                    "rule_source_mode": "sih_ps_26034",
                    "rule_type": "general_declaration",
                    "condition": "pdp_area_200_to_500_cm2",
                    "required_height_mm": 2.0,
                    "font_category": "normal",
                    "normalized_value": None,
                    "normalized_unit": None,
                },
                "comparison": {
                    "status": "above_threshold",
                    "margin_mm": 0.0,
                },
                "inspection_status": "compliant",
                "notes": "Measured height (2.00 mm) meets minimum required height (2.00 mm).",
            },
            {
                "field": "dates[1]",
                "declared_text": "USE BY: 08/12/26",
                "declared_numeral": "08/12/26",
                "declared_value": None,
                "declared_unit": None,
                "measurement": {
                    "height_mm": 2.00,
                    "height_px": 14,
                    "confidence": 0.89,
                    "quality": "good",
                },
                "requirement": {
                    "rule_reference": "Legal Metrology (Packaged Commodities) Rules, 2011 — Rule 7 (Second Schedule Table 1)",
                    "rule_source_mode": "sih_ps_26034",
                    "rule_type": "general_declaration",
                    "condition": "pdp_area_200_to_500_cm2",
                    "required_height_mm": 2.0,
                    "font_category": "normal",
                    "normalized_value": None,
                    "normalized_unit": None,
                },
                "comparison": {
                    "status": "above_threshold",
                    "margin_mm": 0.0,
                },
                "inspection_status": "compliant",
                "notes": "Measured height (2.00 mm) meets minimum required height (2.00 mm).",
            },
        ],
        "disclaimer": "Automated inspection-support tool only.",
    }


@pytest.fixture
def sample_rule8():
    """Module 3B Rule 8 spatial clearance evaluation result."""
    return {
        "summary": {
            "total_evaluated": 1,
            "compliant_count": 1,
            "non_compliant_count": 0,
            "indeterminate_count": 0,
            "not_applicable_count": 0,
        },
        "findings": [
            {
                "field": "net_quantity",
                "rule_reference": "Legal Metrology (Packaged Commodities) Rules, 2011 — Rule 8(1) proviso",
                "clearance_reference": "quantity_declaration_bbox",
                "declaration_bbox_px": [180, 1440, 480, 1520],
                "numeral_bbox_px": [264, 1456, 360, 1504],
                "measurement": {
                    "height_px": 18,
                    "height_mm": 2.50,
                    "confidence": 0.95,
                    "quality": "good",
                },
                "required_clearance_mm": {
                    "top_mm": 2.50,
                    "bottom_mm": 2.50,
                    "left_mm": 5.00,
                    "right_mm": 5.00,
                },
                "actual_clearance_px": {
                    "top_px": 70.0,
                    "bottom_px": 36.0,
                    "left_px": 109.0,
                    "right_px": 109.0,
                },
                "actual_clearance_mm": {
                    "top_mm": 10.00,
                    "bottom_mm": 5.14,
                    "left_mm": 16.36,
                    "right_mm": 16.36,
                },
                "direction_results": {
                    "top": "above_threshold",
                    "bottom": "above_threshold",
                    "left": "above_threshold",
                    "right": "above_threshold",
                },
                "inspection_status": "compliant",
                "confidence": 0.95,
                "notes": "All cardinal clearances exceed statutory minimums.",
            }
        ],
        "disclaimer": "Automated inspection-support tool only.",
    }


# ==============================================================================
# 34 REQUIRED TEST CASES
# ==============================================================================

# 1. Pydantic model input
def test_1_pydantic_model_input(service, sample_extraction, sample_measurements, sample_rule6, sample_rule7, sample_rule8):
    ext_model = LabelExtractionResult.model_validate(sample_extraction)
    meas_model = MeasurementResponseData.model_validate(sample_measurements)
    r6_model = Rule6ResponseData.model_validate(sample_rule6)
    r7_model = Rule7ResponseData.model_validate(sample_rule7)
    r8_model = Rule8ResponseData.model_validate(sample_rule8)

    result = service.aggregate(
        extraction=ext_model,
        measurements=meas_model,
        rule6=r6_model,
        rule7=r7_model,
        rule8=r8_model,
    )
    assert isinstance(result, InspectionResponseData)
    assert result.metadata.brand_name == "Lay's"


# 2. Dictionary input
def test_2_dictionary_input(service, sample_extraction, sample_measurements, sample_rule6, sample_rule7, sample_rule8):
    result = service.aggregate(
        extraction=sample_extraction,
        measurements=sample_measurements,
        rule6=sample_rule6,
        rule7=sample_rule7,
        rule8=sample_rule8,
    )
    assert isinstance(result, InspectionResponseData)


# 3. JSON string input
def test_3_json_string_input(service, sample_extraction, sample_measurements, sample_rule6, sample_rule7, sample_rule8):
    result = service.aggregate(
        extraction=json.dumps(sample_extraction),
        measurements=json.dumps(sample_measurements),
        rule6=json.dumps(sample_rule6),
        rule7=json.dumps(sample_rule7),
        rule8=json.dumps(sample_rule8),
    )
    assert isinstance(result, InspectionResponseData)


# 4. Data envelope input
def test_4_data_envelope_input(service, sample_extraction, sample_measurements, sample_rule6, sample_rule7, sample_rule8):
    result = service.aggregate(
        extraction={"data": sample_extraction},
        measurements={"data": sample_measurements},
        rule6={"data": sample_rule6},
        rule7={"data": sample_rule7},
        rule8={"data": sample_rule8},
    )
    assert isinstance(result, InspectionResponseData)


# 5. Missing optional module input
def test_5_missing_optional_module_input(service, sample_rule6):
    result = service.aggregate(
        rule6=sample_rule6,
        rule7=None,
        rule8=None,
    )
    assert isinstance(result, InspectionResponseData)
    assert len(result.findings) >= 6


# 6. Rule 6 detected declaration
def test_6_rule6_detected_declaration(service, sample_extraction, sample_rule6):
    result = service.aggregate(extraction=sample_extraction, rule6=sample_rule6)
    gn = [f for f in result.findings if f.field == "generic_name"][0]
    assert gn.visibility == "DETECTED"
    # Per Step 9: detected with no specific compliance assessment -> NOT ASSESSABLE
    assert gn.status == "NOT ASSESSABLE"


# 7. Rule 6 NOT VISIBLE declaration
def test_7_rule6_not_visible_declaration(service, sample_extraction, sample_rule6):
    result = service.aggregate(extraction=sample_extraction, rule6=sample_rule6)
    cc = [f for f in result.findings if f.field == "consumer_care"][0]
    assert cc.visibility == "NOT VISIBLE"
    assert cc.status == "NOT VISIBLE"


# 8. Consumer Care NOT VISIBLE
def test_8_consumer_care_not_visible(service, sample_extraction, sample_rule6):
    result = service.aggregate(extraction=sample_extraction, rule6=sample_rule6)
    cc = [f for f in result.findings if f.field == "consumer_care"][0]
    assert cc.status == "NOT VISIBLE"
    assert "Not visible in supplied image" in cc.notes
    assert "Other package panels may contain the declaration" in cc.notes


# 9. NOT VISIBLE never becomes NON-COMPLIANT
def test_9_not_visible_never_becomes_non_compliant(service, sample_extraction, sample_rule6):
    result = service.aggregate(extraction=sample_extraction, rule6=sample_rule6)
    for f in result.findings:
        if f.visibility == "NOT VISIBLE":
            assert f.status == "NOT VISIBLE"
            assert f.status != "NON-COMPLIANT"


# 10. MRP + Rule 7 NON-COMPLIANT
def test_10_mrp_rule7_non_compliant(service, sample_extraction, sample_rule6, sample_rule7):
    result = service.aggregate(extraction=sample_extraction, rule6=sample_rule6, rule7=sample_rule7)
    mrp = [f for f in result.findings if f.field == "mrp"][0]
    assert mrp.visibility == "DETECTED"
    assert mrp.rules.rule7.status == "NON-COMPLIANT"
    assert mrp.status == "NON-COMPLIANT"


# 11. Net Quantity + Rule 7 PASS
def test_11_net_quantity_rule7_pass(service, sample_extraction, sample_rule6, sample_rule7):
    result = service.aggregate(extraction=sample_extraction, rule6=sample_rule6, rule7=sample_rule7)
    net_f = [f for f in result.findings if f.field == "net_quantity"][0]
    assert net_f.rules.rule7.status == "PASS"


# 12. Net Quantity + Rule 8 PASS
def test_12_net_quantity_rule8_pass(service, sample_extraction, sample_rule6, sample_rule7, sample_rule8):
    result = service.aggregate(
        extraction=sample_extraction, rule6=sample_rule6, rule7=sample_rule7, rule8=sample_rule8
    )
    net_f = [f for f in result.findings if f.field == "net_quantity"][0]
    assert net_f.rules.rule8.status == "PASS"
    assert net_f.status == "PASS"


# 13. Rule 7 NOT ASSESSABLE
def test_13_rule7_not_assessable(service, sample_extraction, sample_rule6, sample_rule7):
    r7_mod = dict(sample_rule7)
    r7_mod["findings"] = [dict(f) for f in sample_rule7["findings"]]
    for f in r7_mod["findings"]:
        if f["field"] == "mrp":
            f["inspection_status"] = "indeterminate_low_confidence"

    result = service.aggregate(extraction=sample_extraction, rule6=sample_rule6, rule7=r7_mod)
    mrp = [f for f in result.findings if f.field == "mrp"][0]
    assert mrp.rules.rule7.status == "NOT ASSESSABLE"
    assert mrp.status == "NOT ASSESSABLE"


# 14. Rule 8 NOT ASSESSABLE
def test_14_rule8_not_assessable(service, sample_extraction, sample_rule6, sample_rule7, sample_rule8):
    r8_mod = dict(sample_rule8)
    r8_mod["findings"] = [dict(f) for f in sample_rule8["findings"]]
    r8_mod["findings"][0]["inspection_status"] = "indeterminate_low_confidence"

    result = service.aggregate(
        extraction=sample_extraction, rule6=sample_rule6, rule7=sample_rule7, rule8=r8_mod
    )
    net_f = [f for f in result.findings if f.field == "net_quantity"][0]
    assert net_f.rules.rule8.status == "NOT ASSESSABLE"
    assert net_f.status == "NOT ASSESSABLE"


# 15. Measured height preserved
def test_15_measured_height_preserved(service, sample_extraction, sample_rule6, sample_rule7):
    result = service.aggregate(extraction=sample_extraction, rule6=sample_rule6, rule7=sample_rule7)
    mrp = [f for f in result.findings if f.field == "mrp"][0]
    assert mrp.rules.rule7.measured_height_mm == 1.65


# 16. Required height preserved
def test_16_required_height_preserved(service, sample_extraction, sample_rule6, sample_rule7):
    result = service.aggregate(extraction=sample_extraction, rule6=sample_rule6, rule7=sample_rule7)
    mrp = [f for f in result.findings if f.field == "mrp"][0]
    assert mrp.rules.rule7.required_height_mm == 2.00


# 17. Height margin preserved
def test_17_height_margin_preserved(service, sample_extraction, sample_rule6, sample_rule7):
    result = service.aggregate(extraction=sample_extraction, rule6=sample_rule6, rule7=sample_rule7)
    mrp = [f for f in result.findings if f.field == "mrp"][0]
    assert mrp.rules.rule7.height_margin_mm == -0.35


# 18. Measurement quality preserved
def test_18_measurement_quality_preserved(service, sample_extraction, sample_rule6, sample_rule7):
    result = service.aggregate(extraction=sample_extraction, rule6=sample_rule6, rule7=sample_rule7)
    mrp = [f for f in result.findings if f.field == "mrp"][0]
    assert mrp.rules.rule7.measurement_quality == "good"


# 19. Confidence preserved
def test_19_confidence_preserved(service, sample_extraction, sample_rule6, sample_rule7, sample_rule8):
    result = service.aggregate(
        extraction=sample_extraction, rule6=sample_rule6, rule7=sample_rule7, rule8=sample_rule8
    )
    mfr = [f for f in result.findings if f.field == "manufacturer_packer_importer"][0]
    assert mfr.confidence == 0.98

    mrp = [f for f in result.findings if f.field == "mrp"][0]
    assert mrp.rules.rule7.confidence == 0.92

    net_f = [f for f in result.findings if f.field == "net_quantity"][0]
    assert net_f.rules.rule8.confidence == 0.95


# 20. Rule 8 directional measurements preserved
def test_20_rule8_directional_measurements_preserved(service, sample_extraction, sample_rule6, sample_rule8):
    result = service.aggregate(extraction=sample_extraction, rule6=sample_rule6, rule8=sample_rule8)
    net_f = [f for f in result.findings if f.field == "net_quantity"][0]
    r8 = net_f.rules.rule8
    assert r8.top.measured_mm == 10.00
    assert r8.top.required_mm == 2.50
    assert r8.top.status == "above_threshold"
    assert r8.bottom.measured_mm == 5.14
    assert r8.bottom.required_mm == 2.50
    assert r8.left.measured_mm == 16.36
    assert r8.left.required_mm == 5.00
    assert r8.right.measured_mm == 16.36
    assert r8.right.required_mm == 5.00


# 21. MRP Rule 6 + Rule 7 merge
def test_21_mrp_merge(service, sample_extraction, sample_rule6, sample_rule7):
    result = service.aggregate(extraction=sample_extraction, rule6=sample_rule6, rule7=sample_rule7)
    mrp = [f for f in result.findings if f.field == "mrp"][0]
    assert mrp.visibility == "DETECTED"
    assert mrp.rules.rule7 is not None
    assert mrp.rules.rule8 is None
    assert mrp.status == "NON-COMPLIANT"


# 22. Net Quantity Rule 6 + Rule 7 + Rule 8 merge
def test_22_net_quantity_merge(service, sample_extraction, sample_rule6, sample_rule7, sample_rule8):
    result = service.aggregate(
        extraction=sample_extraction, rule6=sample_rule6, rule7=sample_rule7, rule8=sample_rule8
    )
    net_f = [f for f in result.findings if f.field == "net_quantity"][0]
    assert net_f.visibility == "DETECTED"
    assert net_f.rules.rule7 is not None
    assert net_f.rules.rule8 is not None
    assert net_f.status == "PASS"


# 23. No duplicate declarations
def test_23_no_duplicate_declarations(service, sample_extraction, sample_rule6, sample_rule7, sample_rule8):
    result = service.aggregate(
        extraction=sample_extraction, rule6=sample_rule6, rule7=sample_rule7, rule8=sample_rule8
    )
    fields = [f.field for f in result.findings]
    assert len(fields) == len(set(fields))


# 24. Manufacture / use_by semantic separation
def test_24_date_semantic_separation(service, sample_extraction, sample_rule6, sample_rule7):
    result = service.aggregate(extraction=sample_extraction, rule6=sample_rule6, rule7=sample_rule7)
    fields = [f.field for f in result.findings]
    assert "date_of_manufacture" in fields
    assert "date_use_by" in fields


# 25. Date mapping independent of array position
def test_25_date_mapping_independent_of_order(service, sample_extraction, sample_rule6, sample_rule7):
    # Reverse dates in extraction
    rev_ext = json.loads(json.dumps(sample_extraction))
    rev_ext["semantic_extraction"]["dates"].reverse()

    result = service.aggregate(extraction=rev_ext, rule6=sample_rule6, rule7=sample_rule7)
    d_mfd = [f for f in result.findings if f.field == "date_of_manufacture"][0]
    d_use = [f for f in result.findings if f.field == "date_use_by"][0]

    assert "MFD" in d_mfd.raw_text
    assert "USE BY" in d_use.raw_text


# 26. Evidence preservation
def test_26_evidence_preservation(service, sample_extraction, sample_measurements, sample_rule6, sample_rule7):
    result = service.aggregate(
        extraction=sample_extraction,
        measurements=sample_measurements,
        rule6=sample_rule6,
        rule7=sample_rule7,
    )
    mrp = [f for f in result.findings if f.field == "mrp"][0]
    assert mrp.bbox == [0.60, 0.80, 0.85, 0.85]
    assert mrp.declared_numeral == "10"
    assert mrp.declared_unit == "₹"


# 27. Correct detected count
def test_27_correct_detected_count(service, sample_extraction, sample_rule6, sample_rule7, sample_rule8):
    result = service.aggregate(
        extraction=sample_extraction, rule6=sample_rule6, rule7=sample_rule7, rule8=sample_rule8
    )
    assert result.summary.detected_declarations_count == 6


# 28. Correct not-visible count
def test_28_correct_not_visible_count(service, sample_extraction, sample_rule6, sample_rule7, sample_rule8):
    result = service.aggregate(
        extraction=sample_extraction, rule6=sample_rule6, rule7=sample_rule7, rule8=sample_rule8
    )
    assert result.summary.not_visible_declarations_count == 1


# 29. Correct PASS rule count
def test_29_correct_pass_rule_count(service, sample_extraction, sample_rule6, sample_rule7, sample_rule8):
    result = service.aggregate(
        extraction=sample_extraction, rule6=sample_rule6, rule7=sample_rule7, rule8=sample_rule8
    )
    # Net quantity R7 (PASS), Net quantity R8 (PASS), MFD R7 (PASS), USE BY R7 (PASS) = 4
    assert result.summary.pass_findings_count == 4


# 30. Correct NON-COMPLIANT rule count
def test_30_correct_non_compliant_rule_count(service, sample_extraction, sample_rule6, sample_rule7, sample_rule8):
    result = service.aggregate(
        extraction=sample_extraction, rule6=sample_rule6, rule7=sample_rule7, rule8=sample_rule8
    )
    # MRP R7 (NON-COMPLIANT) = 1
    assert result.summary.non_compliant_findings_count == 1


# 31. Correct NOT ASSESSABLE rule count
def test_31_correct_not_assessable_rule_count(service, sample_extraction, sample_rule6, sample_rule7, sample_rule8):
    result = service.aggregate(
        extraction=sample_extraction, rule6=sample_rule6, rule7=sample_rule7, rule8=sample_rule8
    )
    assert result.summary.not_assessable_findings_count == 0


# 32. POST /api/v1/compliance/inspection integration test
def test_32_api_integration_test(client, sample_extraction, sample_measurements, sample_rule6, sample_rule7, sample_rule8):
    payload = {
        "extraction": sample_extraction,
        "measurements": sample_measurements,
        "rule6": sample_rule6,
        "rule7": sample_rule7,
        "rule8": sample_rule8,
    }
    response = client.post("/api/v1/compliance/inspection", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "findings" in data["data"]
    assert len(data["data"]["findings"]) == 7


# 33. Response matches Pydantic schema
def test_33_response_matches_pydantic_schema(client, sample_extraction, sample_measurements, sample_rule6, sample_rule7, sample_rule8):
    payload = {
        "extraction": sample_extraction,
        "measurements": sample_measurements,
        "rule6": sample_rule6,
        "rule7": sample_rule7,
        "rule8": sample_rule8,
    }
    response = client.post("/api/v1/compliance/inspection", json=payload)
    assert response.status_code == 200
    # Validate against schema
    resp_obj = InspectionResponse.model_validate(response.json())
    assert resp_obj.success is True
    assert resp_obj.data.metadata.brand_name == "Lay's"


# 34. Upstream rule results are not recalculated
def test_34_upstream_rule_results_not_recalculated(service, sample_extraction, sample_measurements, sample_rule6, sample_rule7, sample_rule8):
    # Tamper with Rule 7: set measured 999.0 mm but keep status as "non_compliant"
    tampered_r7 = json.loads(json.dumps(sample_rule7))
    tampered_r7["findings"][0]["measurement"]["height_mm"] = 999.0
    tampered_r7["findings"][0]["requirement"]["required_height_mm"] = 2.0
    tampered_r7["findings"][0]["inspection_status"] = "non_compliant"

    result = service.aggregate(
        extraction=sample_extraction,
        measurements=sample_measurements,
        rule6=sample_rule6,
        rule7=tampered_r7,
        rule8=sample_rule8,
    )
    mrp = [f for f in result.findings if f.field == "mrp"][0]
    # It must faithfully preserve "NON-COMPLIANT" from Rule 7 and NOT recalculate 999.0 >= 2.0
    assert mrp.rules.rule7.status == "NON-COMPLIANT"
    assert mrp.rules.rule7.measured_height_mm == 999.0
