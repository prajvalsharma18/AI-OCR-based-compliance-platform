"""Deterministic Unit & Integration Tests for Module 3: Rule 7 Numeral-Height Evaluation Engine.

Covers:
- All 15 required unit cases from the SIH specification
- Lay's product full integration scenario
- FastAPI endpoint integration via TestClient
"""

import json
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.compliance import (
    Rule7EvaluationRequest,
    Rule7Finding,
    Rule7ResponseData,
)
from app.schemas.measurement import NumeralMeasurement
from app.services.rule7_service import Rule7Service


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture
def service():
    """Rule7Service instance fixture."""
    return Rule7Service()


def make_measurement(
    field: str,
    raw_text: str,
    numeral: str,
    height_mm: float,
    height_px: int = 24,
    confidence: float = 1.0,
    quality: str = "good",
) -> NumeralMeasurement:
    """Helper to construct a NumeralMeasurement instance."""
    return NumeralMeasurement(
        field=field,
        raw_text=raw_text,
        numeral=numeral,
        semantic_region_normalized=[0.1, 0.1, 0.2, 0.2],
        numeral_bbox_px=[10, 10, 30, 34],
        numeral_height_px=height_px,
        numeral_height_mm=height_mm,
        confidence=confidence,
        measurement_quality=quality,  # type: ignore
        notes="Synthetic test measurement",
    )


# ==============================================================================
# 15 MANDATORY SIH SPECIFICATION TESTS
# ==============================================================================

def test_1_net_quantity_28_5g_normal(service):
    """TEST 1: 28.5 g, normal font, measured = 1.87 mm -> required = 1.0 mm, comparison = above_threshold."""
    meas = make_measurement(
        field="net_quantity",
        raw_text="Net Qty: 28.5 g",
        numeral="28.5",
        height_mm=1.87,
        quality="good",
    )
    finding = service.evaluate_declaration(
        field="net_quantity",
        raw_text="Net Qty: 28.5 g",
        numeral_str="28.5",
        declared_value=28.5,
        declared_unit="g",
        quantity_type="weight",
        measurement=meas,
        font_category="normal",
    )

    assert finding.requirement.rule_type == "quantity_weight_volume"
    assert finding.requirement.condition == "below_200_g_ml"
    assert finding.requirement.required_height_mm == 1.0
    assert finding.comparison.status == "above_threshold"
    assert finding.comparison.margin_mm == 0.87
    assert finding.inspection_status == "compliant"


def test_2_net_quantity_150g_normal(service):
    """TEST 2: 150 g, normal font, measured = 0.8 mm -> required = 1.0 mm, comparison = below_threshold."""
    meas = make_measurement(
        field="net_quantity",
        raw_text="Net Weight 150 g",
        numeral="150",
        height_mm=0.8,
        quality="good",
    )
    finding = service.evaluate_declaration(
        field="net_quantity",
        raw_text="Net Weight 150 g",
        numeral_str="150",
        declared_value=150.0,
        declared_unit="g",
        quantity_type="weight",
        measurement=meas,
        font_category="normal",
    )

    assert finding.requirement.condition == "below_200_g_ml"
    assert finding.requirement.required_height_mm == 1.0
    assert finding.comparison.status == "below_threshold"
    assert finding.comparison.margin_mm == -0.2
    assert finding.inspection_status == "non_compliant"


def test_3_boundary_500g_normal(service):
    """TEST 3: 500 g, normal font, measured = 2.0 mm -> boundary handling for 200–500 g range."""
    meas = make_measurement(
        field="net_quantity",
        raw_text="500 g",
        numeral="500",
        height_mm=2.0,
        quality="good",
    )
    finding = service.evaluate_declaration(
        field="net_quantity",
        raw_text="500 g",
        numeral_str="500",
        declared_value=500.0,
        declared_unit="g",
        quantity_type="weight",
        measurement=meas,
        font_category="normal",
    )

    assert finding.requirement.condition == "between_200_and_500_g_ml"
    assert finding.requirement.required_height_mm == 2.0
    assert finding.comparison.status == "above_threshold"
    assert finding.comparison.margin_mm == 0.0
    assert finding.inspection_status == "compliant"


def test_4_boundary_501g_normal(service):
    """TEST 4: 501 g, normal font, measured = 4.0 mm -> above-500-g rule."""
    meas = make_measurement(
        field="net_quantity",
        raw_text="501 g",
        numeral="501",
        height_mm=4.0,
        quality="good",
    )
    finding = service.evaluate_declaration(
        field="net_quantity",
        raw_text="501 g",
        numeral_str="501",
        declared_value=501.0,
        declared_unit="g",
        quantity_type="weight",
        measurement=meas,
        font_category="normal",
    )

    assert finding.requirement.condition == "above_500_g_ml"
    assert finding.requirement.required_height_mm == 4.0
    assert finding.comparison.status == "above_threshold"
    assert finding.comparison.margin_mm == 0.0
    assert finding.inspection_status == "compliant"


def test_5_molded_28_5g(service):
    """TEST 5: 28.5 g, molded/embossed, measured = 1.5 mm -> required = 2.0 mm, below_threshold."""
    meas = make_measurement(
        field="net_quantity",
        raw_text="28.5 g",
        numeral="28.5",
        height_mm=1.5,
        quality="good",
    )
    finding = service.evaluate_declaration(
        field="net_quantity",
        raw_text="28.5 g",
        numeral_str="28.5",
        declared_value=28.5,
        declared_unit="g",
        quantity_type="weight",
        measurement=meas,
        font_category="embossed",
    )

    assert finding.requirement.condition == "below_200_g_ml"
    assert finding.requirement.required_height_mm == 2.0  # Molded requirement for <200g is 2mm
    assert finding.comparison.status == "below_threshold"
    assert finding.comparison.margin_mm == -0.5
    assert finding.inspection_status == "non_compliant"


def test_6_unit_normalization_half_kg(service):
    """TEST 6: 0.5 kg -> normalized comparison value = 500 g -> between_200_and_500_g_ml."""
    norm_val, norm_unit, err = service.normalize_quantity(0.5, "kg", "weight")
    assert err is None
    assert norm_val == 500.0
    assert norm_unit == "g"

    cond, req = service.determine_weight_volume_threshold(norm_val, "normal")
    assert cond == "between_200_and_500_g_ml"
    assert req == 2.0


def test_7_unit_normalization_one_kg(service):
    """TEST 7: 1 kg -> normalized comparison value = 1000 g -> above-500-g range."""
    norm_val, norm_unit, err = service.normalize_quantity(1.0, "kg", "weight")
    assert err is None
    assert norm_val == 1000.0
    assert norm_unit == "g"

    cond, req = service.determine_weight_volume_threshold(norm_val, "normal")
    assert cond == "above_500_g_ml"
    assert req == 4.0


def test_8_phone_number_exclusion(service):
    """TEST 8: Phone number -> not evaluated by Rule 7 numeral-height engine."""
    finding = service.evaluate_declaration(
        field="consumer_care_phone",
        raw_text="1800-222-444",
        numeral_str="1800222444",
        declared_value=None,
        declared_unit=None,
        quantity_type=None,
        measurement=None,
        is_non_rule7=True,
    )

    assert finding.requirement.rule_type == "not_applicable"
    assert finding.comparison.status == "not_applicable"
    assert finding.inspection_status == "not_applicable"
    assert "excluded" in finding.notes.lower()


def test_9_pin_code_exclusion(service):
    """TEST 9: PIN code -> not evaluated by Rule 7 numeral-height engine."""
    finding = service.evaluate_declaration(
        field="additional_numeric_information[0]_pin_code",
        raw_text="PIN 122002",
        numeral_str="122002",
        declared_value=None,
        declared_unit=None,
        quantity_type=None,
        measurement=None,
        is_non_rule7=True,
    )

    assert finding.requirement.rule_type == "not_applicable"
    assert finding.comparison.status == "not_applicable"
    assert finding.inspection_status == "not_applicable"


def test_10_barcode_exclusion(service):
    """TEST 10: Barcode number -> not evaluated by Rule 7 numeral-height engine."""
    finding = service.evaluate_declaration(
        field="barcode_numeral",
        raw_text="8901030383829",
        numeral_str="8901030383829",
        declared_value=None,
        declared_unit=None,
        quantity_type=None,
        measurement=None,
        is_non_rule7=True,
    )

    assert finding.requirement.rule_type == "not_applicable"
    assert finding.comparison.status == "not_applicable"
    assert finding.inspection_status == "not_applicable"


def test_11_mrp_numeral_handling(service):
    """TEST 11: MRP numeral -> general/non-quantity handling, NOT weight/volume quantity threshold."""
    meas = make_measurement(
        field="mrp",
        raw_text="MRP Rs. 10.00",
        numeral="10.00",
        height_mm=1.95,
        quality="good",
    )
    finding = service.evaluate_declaration(
        field="mrp",
        raw_text="MRP Rs. 10.00",
        numeral_str="10.00",
        declared_value=10.0,
        declared_unit="₹",
        quantity_type=None,
        measurement=meas,
        font_category="normal",
        pdp_area_cm2=150.0,  # 100-500 cm2 range -> general minimum 2.0 mm
    )

    assert finding.requirement.rule_type == "general_declaration"
    assert "general" in finding.requirement.condition
    assert finding.requirement.required_height_mm == 2.0
    # Measured 1.95 mm < 2.0 mm
    assert finding.comparison.status == "below_threshold"
    assert finding.comparison.margin_mm == -0.05
    assert finding.inspection_status == "non_compliant"


def test_12_numeric_date_handling(service):
    """TEST 12: numeric month/year -> general/non-quantity handling, NOT quantity threshold."""
    meas = make_measurement(
        field="dates[0]",
        raw_text="Mfg: 07/2026",
        numeral="07/2026",
        height_mm=2.76,
        quality="good",
    )
    finding = service.evaluate_declaration(
        field="dates[0]",
        raw_text="Mfg: 07/2026",
        numeral_str="07/2026",
        declared_value=None,
        declared_unit=None,
        quantity_type=None,
        measurement=meas,
        font_category="normal",
        pdp_area_cm2=80.0,  # Below 100 cm2 -> required = 1.0 mm
    )

    assert finding.requirement.rule_type == "general_declaration"
    assert finding.requirement.required_height_mm == 1.0
    assert finding.comparison.status == "above_threshold"
    assert finding.inspection_status == "compliant"


def test_13_low_measurement_quality_indeterminacy(service):
    """TEST 13: measurement_quality = low -> comparison computed, but status is indeterminate_low_confidence."""
    meas = make_measurement(
        field="net_quantity",
        raw_text="Net Qty: 28.5 g",
        numeral="28.5",
        height_mm=1.87,
        confidence=0.35,
        quality="low",
    )
    finding = service.evaluate_declaration(
        field="net_quantity",
        raw_text="Net Qty: 28.5 g",
        numeral_str="28.5",
        declared_value=28.5,
        declared_unit="g",
        quantity_type="weight",
        measurement=meas,
        font_category="normal",
    )

    # Mathematical comparison is computed (1.87 >= 1.0)
    assert finding.comparison.status == "above_threshold"
    assert finding.comparison.margin_mm == 0.87
    # But inspection status MUST be indeterminate_low_confidence due to low quality
    assert finding.inspection_status == "indeterminate_low_confidence"
    assert "cannot be treated as definitive legal conclusion" in finding.notes


def test_14_missing_numeral_measurement(service):
    """TEST 14: missing numeral measurement -> indeterminate/unsupported result, no fabricated measurement."""
    finding = service.evaluate_declaration(
        field="net_quantity",
        raw_text="Net Qty: 28.5 g",
        numeral_str="28.5",
        declared_value=28.5,
        declared_unit="g",
        quantity_type="weight",
        measurement=None,
        font_category="normal",
    )

    assert finding.measurement is None
    assert finding.comparison.status == "indeterminate"
    assert finding.comparison.margin_mm is None
    assert finding.inspection_status == "indeterminate_missing_measurement"
    assert "no physical numeral measurement" in finding.notes.lower()


def test_15_unsupported_unit(service):
    """TEST 15: unsupported unit -> structured unsupported result, no crash."""
    finding = service.evaluate_declaration(
        field="net_quantity",
        raw_text="Net Qty: 28.5 cubits",
        numeral_str="28.5",
        declared_value=28.5,
        declared_unit="cubits",
        quantity_type="length",
        measurement=None,
        font_category="normal",
    )

    assert finding.requirement.rule_type == "unsupported"
    assert finding.comparison.status == "unsupported"
    assert finding.inspection_status == "unsupported"
    assert "unsupported unit" in finding.notes.lower()


# ==============================================================================
# LAY'S FULL INTEGRATION TEST
# ==============================================================================

def test_lays_integration_module1_and_module2a(service):
    """Evaluates the full Lay's package scenario with Module 1 & Module 2A outputs.

    Candidates:
    - MRP: 10, height_px=24, height_mm=1.95, quality="low", confidence=0.35
    - date[0] (Mfg): 07/2026, height_px=34, height_mm=2.76, quality="low", confidence=0.35
    - date[1] (Use-by): 12/2026, height_px=34, height_mm=2.76, quality="low", confidence=0.35
    - net_quantity: 28.5 g, height_px=23, height_mm=1.87, quality="low", confidence=0.35
    - non-Rule 7 numbers (PIN code) should be excluded/not_applicable
    """
    lays_extraction = {
        "image": {"width": 1000, "height": 1000, "coordinate_system": "normalized_0_1"},
        "raw_text_blocks": [],
        "semantic_extraction": {
            "image_quality": {"overall": "good"},
            "package": {"package_type": "retail", "brand_name": "Lay's"},
            "manufacturer": {"status": "present", "entities": []},
            "generic_name": {"status": "present", "value": "Potato Chips"},
            "mrp": {
                "status": "present",
                "raw_text": "MRP Rs. 10.00",
                "value": 10.0,
                "currency": "₹",
                "numeral_region": [0.2, 0.4, 0.3, 0.5],
            },
            "dates": [
                {
                    "status": "present",
                    "date_type": "manufacture",
                    "raw_text": "Mfg: 07/2026",
                    "month": "07",
                    "year": 2026,
                    "numeral_region": [0.5, 0.4, 0.7, 0.5],
                },
                {
                    "status": "present",
                    "date_type": "packing",
                    "raw_text": "Use by: 12/2026",
                    "month": "12",
                    "year": 2026,
                    "numeral_region": [0.5, 0.5, 0.7, 0.6],
                },
            ],
            "net_quantity": {
                "status": "present",
                "raw_text": "Net Qty: 28.5 g",
                "value": 28.5,
                "unit": "g",
                "quantity_type": "weight",
                "numeral_region": [0.2, 0.5, 0.3, 0.6],
            },
            "consumer_care": {"status": "missing"},
            "additional_numeric_information": [
                {
                    "semantic_role": "pin_code",
                    "raw_text": "PIN 122002",
                    "value": "122002",
                }
            ],
        },
    }

    lays_measurements = {
        "calibration": {
            "package_width_mm": 150.0,
            "package_height_mm": 200.0,
        },
        "measurements": [
            {
                "field": "net_quantity",
                "raw_text": "Net Qty: 28.5 g",
                "numeral": "28.5",
                "semantic_region_normalized": [0.2, 0.5, 0.3, 0.6],
                "numeral_bbox_px": [200, 500, 300, 523],
                "numeral_height_px": 23,
                "numeral_height_mm": 1.87,
                "confidence": 0.35,
                "measurement_quality": "low",
                "notes": "Low contrast on flexible foil background",
            },
            {
                "field": "mrp",
                "raw_text": "MRP Rs. 10.00",
                "numeral": "10",
                "semantic_region_normalized": [0.2, 0.4, 0.3, 0.5],
                "numeral_bbox_px": [200, 400, 300, 424],
                "numeral_height_px": 24,
                "numeral_height_mm": 1.95,
                "confidence": 0.35,
                "measurement_quality": "low",
                "notes": "Dot matrix print on crinkled foil",
            },
            {
                "field": "dates[0]",
                "raw_text": "Mfg: 07/2026",
                "numeral": "07/2026",
                "semantic_region_normalized": [0.5, 0.4, 0.7, 0.5],
                "numeral_bbox_px": [500, 400, 700, 434],
                "numeral_height_px": 34,
                "numeral_height_mm": 2.76,
                "confidence": 0.35,
                "measurement_quality": "low",
                "notes": "Inkjet stamp",
            },
            {
                "field": "dates[1]",
                "raw_text": "Use by: 12/2026",
                "numeral": "12/2026",
                "semantic_region_normalized": [0.5, 0.5, 0.7, 0.6],
                "numeral_bbox_px": [500, 500, 700, 534],
                "numeral_height_px": 34,
                "numeral_height_mm": 2.76,
                "confidence": 0.35,
                "measurement_quality": "low",
                "notes": "Inkjet stamp",
            },
        ],
    }

    result: Rule7ResponseData = service.evaluate(
        extraction=lays_extraction,
        measurements=lays_measurements,
        font_category="normal",
    )

    assert result.summary.total_evaluated == 5  # 4 statutory + 1 non-rule 7 pin_code
    assert result.summary.not_applicable_count == 1
    assert result.summary.indeterminate_count == 4  # All 4 statutory have low quality

    findings_by_field = {f.field: f for f in result.findings}

    # Verify Net Quantity
    net_f = findings_by_field["net_quantity"]
    assert net_f.requirement.condition == "below_200_g_ml"
    assert net_f.requirement.required_height_mm == 1.0
    assert net_f.comparison.status == "above_threshold"
    assert net_f.comparison.margin_mm == 0.87
    assert net_f.inspection_status == "indeterminate_low_confidence"

    # Verify MRP
    mrp_f = findings_by_field["mrp"]
    assert mrp_f.requirement.rule_type == "general_declaration"
    assert mrp_f.inspection_status == "indeterminate_low_confidence"

    # Verify Dates
    d0_f = findings_by_field["dates[0]"]
    assert d0_f.requirement.rule_type == "general_declaration"
    assert d0_f.inspection_status == "indeterminate_low_confidence"

    # Verify PIN code exclusion
    pin_f = [f for f in result.findings if "pin_code" in f.field][0]
    assert pin_f.requirement.rule_type == "not_applicable"
    assert pin_f.inspection_status == "not_applicable"


# ==============================================================================
# FASTAPI ENDPOINT INTEGRATION TESTS
# ==============================================================================

def test_api_rule7_endpoint_success(client):
    """Tests POST /api/v1/compliance/rule7 endpoint with valid payload."""
    payload = {
        "extraction": {
            "image": {"width": 1000, "height": 1000, "coordinate_system": "normalized_0_1"},
            "semantic_extraction": {
                "image_quality": {"overall": "good"},
                "package": {"package_type": "retail"},
                "manufacturer": {"status": "missing"},
                "generic_name": {"status": "missing"},
                "mrp": {"status": "missing"},
                "net_quantity": {
                    "status": "present",
                    "raw_text": "Net Qty: 250 g",
                    "value": 250.0,
                    "unit": "g",
                    "quantity_type": "weight",
                },
                "consumer_care": {"status": "missing"},
            },
        },
        "measurements": [
            {
                "field": "net_quantity",
                "raw_text": "Net Qty: 250 g",
                "numeral": "250",
                "semantic_region_normalized": [0.1, 0.1, 0.3, 0.3],
                "numeral_bbox_px": [100, 100, 300, 320],
                "numeral_height_px": 20,
                "numeral_height_mm": 2.2,
                "confidence": 0.95,
                "measurement_quality": "good",
            }
        ],
        "font_category": "normal",
    }

    response = client.post("/api/v1/compliance/rule7", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["success"] is True
    data = res_data["data"]
    assert data["summary"]["compliant_count"] == 1
    assert data["findings"][0]["requirement"]["required_height_mm"] == 2.0  # 200-500g -> 2.0 mm
    assert data["findings"][0]["inspection_status"] == "compliant"


def test_api_rule7_endpoint_malformed_extraction(client):
    """Tests POST /api/v1/compliance/rule7 with invalid JSON string."""
    payload = {
        "extraction": "{ malformed_json ",
        "measurements": [],
    }
    response = client.post("/api/v1/compliance/rule7", json=payload)
    assert response.status_code == 400
    res_data = response.json()
    assert res_data["success"] is False
    assert res_data["error"]["code"] == "MALFORMED_EXTRACTION_JSON"


def test_health_check_includes_module3(client):
    """Verifies that /health endpoint reflects Module 3."""
    response = client.get("/health")
    assert response.status_code == 200
    res_data = response.json()
    assert "3: Rule 7 Numeral-Height Evaluation Engine" in res_data["modules"]


# ==============================================================================
# RULE SOURCE MODE TESTS: SIH PS 26034 vs. DoCA STATUTORY PCR 2011 TABLE-I
# ==============================================================================

def test_sih_specification_pdp_thresholds(service):
    """Verifies the 4-tier PDP area thresholds in SIH specification mode (1/2/4/6 mm)."""
    # 1. Below 100 cm2 -> 1.0 mm (molded 2.0 mm)
    cond, req = service.determine_pdp_area_threshold(50.0, "normal", rule_source_mode="sih_ps_26034")
    assert req == 1.0
    cond_m, req_m = service.determine_pdp_area_threshold(50.0, "molded", rule_source_mode="sih_ps_26034")
    assert req_m == 2.0

    # 2. 100-500 cm2 -> 2.0 mm (molded 4.0 mm)
    cond, req = service.determine_pdp_area_threshold(250.0, "normal", rule_source_mode="sih_ps_26034")
    assert req == 2.0
    cond_m, req_m = service.determine_pdp_area_threshold(250.0, "molded", rule_source_mode="sih_ps_26034")
    assert req_m == 4.0

    # 3. 500-2500 cm2 -> 4.0 mm (molded 6.0 mm)
    cond, req = service.determine_pdp_area_threshold(1000.0, "normal", rule_source_mode="sih_ps_26034")
    assert req == 4.0
    cond_m, req_m = service.determine_pdp_area_threshold(1000.0, "molded", rule_source_mode="sih_ps_26034")
    assert req_m == 6.0

    # 4. Above 2500 cm2 -> 6.0 mm (molded 6.0 mm)
    cond, req = service.determine_pdp_area_threshold(3000.0, "normal", rule_source_mode="sih_ps_26034")
    assert req == 6.0
    cond_m, req_m = service.determine_pdp_area_threshold(3000.0, "molded", rule_source_mode="sih_ps_26034")
    assert req_m == 6.0


def test_doca_statutory_table1_pdp_thresholds(service):
    """Verifies the 5-tier official DoCA PCR 2011 Table-I thresholds (1/1.5/2.5/4/6 mm)."""
    # 1. A <= 50 cm2 -> 1.0 mm (molded 2.0 mm)
    cond, req = service.determine_pdp_area_threshold(40.0, "normal", rule_source_mode="doca_statutory_2011")
    assert req == 1.0
    assert "lte_50_cm2" in cond
    cond_m, req_m = service.determine_pdp_area_threshold(40.0, "molded", rule_source_mode="doca_statutory_2011")
    assert req_m == 2.0

    # 2. 50 < A <= 100 cm2 -> 1.5 mm (molded 3.0 mm)
    cond, req = service.determine_pdp_area_threshold(75.0, "normal", rule_source_mode="doca_statutory_2011")
    assert req == 1.5
    assert "between_50_and_100_cm2" in cond
    cond_m, req_m = service.determine_pdp_area_threshold(75.0, "molded", rule_source_mode="doca_statutory_2011")
    assert req_m == 3.0

    # 3. 100 < A <= 500 cm2 -> 2.5 mm (molded 4.0 mm) - NOTE: SIH uses 2.0 mm here!
    cond, req = service.determine_pdp_area_threshold(300.0, "normal", rule_source_mode="doca_statutory_2011")
    assert req == 2.5
    assert "between_100_and_500_cm2" in cond
    cond_m, req_m = service.determine_pdp_area_threshold(300.0, "molded", rule_source_mode="doca_statutory_2011")
    assert req_m == 4.0

    # 4. 500 < A <= 2500 cm2 -> 4.0 mm (molded 6.0 mm)
    cond, req = service.determine_pdp_area_threshold(1000.0, "normal", rule_source_mode="doca_statutory_2011")
    assert req == 4.0
    cond_m, req_m = service.determine_pdp_area_threshold(1000.0, "molded", rule_source_mode="doca_statutory_2011")
    assert req_m == 6.0

    # 5. A > 2500 cm2 -> 6.0 mm (molded 6.0 mm)
    cond, req = service.determine_pdp_area_threshold(3000.0, "normal", rule_source_mode="doca_statutory_2011")
    assert req == 6.0
    cond_m, req_m = service.determine_pdp_area_threshold(3000.0, "molded", rule_source_mode="doca_statutory_2011")
    assert req_m == 6.0


def test_lays_mrp_and_date_thresholds_both_modes(service):
    """Demonstrates and documents the exact difference in Lay's thresholds between SIH and DoCA modes.

    Lay's package dimensions: 150 mm x 200 mm -> PDP face area = 300 cm2.
    - Under SIH PS 26034 (100-500 cm2 tier):
      MRP & Dates required height = 2.0 mm
    - Under DoCA Statutory PCR 2011 Table-I (100 < A <= 500 cm2 tier):
      MRP & Dates required height = 2.5 mm
    - Net Quantity (28.5 g) uses weight/volume rule (< 200 g):
      Required height = 1.0 mm in BOTH modes.
    """
    mrp_meas = make_measurement("mrp", "MRP Rs. 10.00", "10", height_mm=1.95, quality="low")
    date_meas = make_measurement("dates[0]", "Mfg: 07/2026", "07/2026", height_mm=2.76, quality="low")
    net_meas = make_measurement("net_quantity", "Net Qty: 28.5 g", "28.5", height_mm=1.87, quality="low")

    pdp_area = 300.0  # (150 mm * 200 mm) / 100 = 300 cm2

    # --- Mode 1: SIH Technical Specification Mode ---
    sih_mrp = service.evaluate_declaration(
        field="mrp",
        raw_text="MRP Rs. 10.00",
        numeral_str="10",
        declared_value=10.0,
        declared_unit="₹",
        quantity_type=None,
        measurement=mrp_meas,
        pdp_area_cm2=pdp_area,
        rule_source_mode="sih_ps_26034",
    )
    assert sih_mrp.requirement.required_height_mm == 2.0
    assert sih_mrp.comparison.margin_mm == -0.05  # 1.95 - 2.00
    assert sih_mrp.comparison.status == "below_threshold"

    sih_date = service.evaluate_declaration(
        field="dates[0]",
        raw_text="Mfg: 07/2026",
        numeral_str="07/2026",
        declared_value=None,
        declared_unit=None,
        quantity_type=None,
        measurement=date_meas,
        pdp_area_cm2=pdp_area,
        rule_source_mode="sih_ps_26034",
    )
    assert sih_date.requirement.required_height_mm == 2.0
    assert sih_date.comparison.margin_mm == 0.76  # 2.76 - 2.00
    assert sih_date.comparison.status == "above_threshold"

    sih_net = service.evaluate_declaration(
        field="net_quantity",
        raw_text="Net Qty: 28.5 g",
        numeral_str="28.5",
        declared_value=28.5,
        declared_unit="g",
        quantity_type="weight",
        measurement=net_meas,
        pdp_area_cm2=pdp_area,
        rule_source_mode="sih_ps_26034",
    )
    assert sih_net.requirement.required_height_mm == 1.0  # Weight rule: < 200g -> 1.0 mm

    # --- Mode 2: Official DoCA PCR 2011 Table-I Mode ---
    doca_mrp = service.evaluate_declaration(
        field="mrp",
        raw_text="MRP Rs. 10.00",
        numeral_str="10",
        declared_value=10.0,
        declared_unit="₹",
        quantity_type=None,
        measurement=mrp_meas,
        pdp_area_cm2=pdp_area,
        rule_source_mode="doca_statutory_2011",
    )
    assert doca_mrp.requirement.required_height_mm == 2.5  # Table-I for 100 < A <= 500 cm2 is 2.5 mm!
    assert doca_mrp.comparison.margin_mm == -0.55  # 1.95 - 2.50
    assert doca_mrp.comparison.status == "below_threshold"

    doca_date = service.evaluate_declaration(
        field="dates[0]",
        raw_text="Mfg: 07/2026",
        numeral_str="07/2026",
        declared_value=None,
        declared_unit=None,
        quantity_type=None,
        measurement=date_meas,
        pdp_area_cm2=pdp_area,
        rule_source_mode="doca_statutory_2011",
    )
    assert doca_date.requirement.required_height_mm == 2.5
    assert doca_date.comparison.margin_mm == 0.26  # 2.76 - 2.50
    assert doca_date.comparison.status == "above_threshold"

    doca_net = service.evaluate_declaration(
        field="net_quantity",
        raw_text="Net Qty: 28.5 g",
        numeral_str="28.5",
        declared_value=28.5,
        declared_unit="g",
        quantity_type="weight",
        measurement=net_meas,
        pdp_area_cm2=pdp_area,
        rule_source_mode="doca_statutory_2011",
    )
    assert doca_net.requirement.required_height_mm == 1.0  # Net quantity remains 1.0 mm in both modes


def test_api_rule7_endpoint_with_doca_mode(client):
    """Tests POST /api/v1/compliance/rule7 with rule_source_mode='doca_statutory_2011'."""
    payload = {
        "rule_source_mode": "doca_statutory_2011",
        "pdp_area_cm2": 300.0,
        "extraction": {
            "image": {"width": 1000, "height": 1000, "coordinate_system": "normalized_0_1"},
            "semantic_extraction": {
                "image_quality": {"overall": "good"},
                "package": {"package_type": "retail"},
                "manufacturer": {"status": "missing"},
                "generic_name": {"status": "missing"},
                "mrp": {
                    "status": "present",
                    "raw_text": "MRP Rs. 10.00",
                    "value": 10.0,
                    "currency": "₹",
                },
                "net_quantity": {"status": "missing"},
                "consumer_care": {"status": "missing"},
            },
        },
        "measurements": [
            {
                "field": "mrp",
                "raw_text": "MRP Rs. 10.00",
                "numeral": "10",
                "semantic_region_normalized": [0.2, 0.4, 0.3, 0.5],
                "numeral_bbox_px": [200, 400, 300, 424],
                "numeral_height_px": 24,
                "numeral_height_mm": 2.6,
                "confidence": 0.95,
                "measurement_quality": "good",
            }
        ],
    }

    response = client.post("/api/v1/compliance/rule7", json=payload)
    assert response.status_code == 200
    res_data = response.json()
    assert res_data["success"] is True
    data = res_data["data"]
    assert data["summary"]["rule_source_mode"] == "doca_statutory_2011"
    # Required height under DoCA Table-I for 300 cm2 PDP is 2.5 mm
    mrp_finding = data["findings"][0]
    assert mrp_finding["requirement"]["required_height_mm"] == 2.5
    assert mrp_finding["requirement"]["rule_source_mode"] == "doca_statutory_2011"
    # 2.6 mm >= 2.5 mm -> compliant
    assert mrp_finding["inspection_status"] == "compliant"

