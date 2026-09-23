"""Unit and integration tests for Module 3: Rule 6 Declaration Visibility & Completeness.

Tests statutory compliance and visibility evaluation under Rule 6 of the
Legal Metrology (Packaged Commodities) Rules, 2011 and SIH 2026 PS 26034:
1. Manufacturer / Packer / Importer (Rule 6(1)(a))
2. Generic Name (Rule 6(1)(b))
3. Retail Sale Price (MRP) (Rule 6(1)(c))
4. Date of Manufacture / Packing / Import (Rule 6(1)(d))
5. Net Quantity (Rule 6(1)(e))
6. Consumer Care Details (Rule 6(1)(f))

Semantic Requirement:
"Not visible in supplied image" is strictly distinguished from "legally missing".
Undetected declarations MUST be reported as 'not_visible' (Display: "Not visible in supplied image")
and NEVER as "Missing" or "Non-compliant".
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.rule6_service import Rule6Service, Rule6ServiceError


def make_full_extraction_payload(
    mfr_present: bool = True,
    generic_present: bool = True,
    mrp_present: bool = True,
    dates_present: bool = True,
    net_qty_present: bool = True,
    care_present: bool = True,
) -> dict:
    """Helper to generate extraction payload with selectively present/absent Rule 6 fields."""
    return {
        "image": {"width": 1200, "height": 1600, "coordinate_system": "normalized_0_1"},
        "raw_text_blocks": [
            {"text": "Sample Pack", "bbox": [0.1, 0.1, 0.9, 0.2]}
        ],
        "semantic_extraction": {
            "image_quality": {"overall": "good", "notes": "Clear shot"},
            "package": {
                "package_type": "retail",
                "brand_name": "TestBrand",
                "generic_name": "Potato Chips",
                "food_status": "food",
                "retail_unit_count": None,
                "evidence": ["Front label"],
            },
            "manufacturer": {
                "status": "present" if mfr_present else "missing",
                "entities": [
                    {
                        "role": "manufacturer",
                        "name": "ABC Foods Pvt. Ltd.",
                        "address": "14 MG Road, Pune 411001",
                        "raw_text": "Mfg by ABC Foods Pvt. Ltd., 14 MG Road, Pune 411001",
                        "confidence": 0.96,
                        "bbox": [0.10, 0.70, 0.90, 0.75],
                    }
                ] if mfr_present else [],
            },
            "generic_name": {
                "status": "present" if generic_present else "missing",
                "raw_text": "Potato Chips" if generic_present else None,
                "value": "Potato Chips" if generic_present else None,
                "confidence": 0.98 if generic_present else 0.0,
                "bbox": [0.20, 0.20, 0.80, 0.25] if generic_present else None,
            },
            "mrp": {
                "status": "present" if mrp_present else "missing",
                "raw_text": "MRP ₹20.00 (Incl. of all taxes)" if mrp_present else None,
                "value": 20.0 if mrp_present else None,
                "currency": "₹" if mrp_present else None,
                "confidence": 0.99 if mrp_present else 0.0,
                "bbox": [0.10, 0.45, 0.45, 0.50] if mrp_present else None,
            },
            "dates": [
                {
                    "status": "present",
                    "date_type": "packing",
                    "raw_text": "Pkd: 08/2026",
                    "month": "08",
                    "year": 2026,
                    "confidence": 0.95,
                    "bbox": [0.50, 0.45, 0.85, 0.50],
                }
            ] if dates_present else [],
            "net_quantity": {
                "status": "present" if net_qty_present else "missing",
                "raw_text": "Net Qty: 50 g" if net_qty_present else None,
                "value": 50.0 if net_qty_present else None,
                "unit": "g" if net_qty_present else None,
                "quantity_type": "weight",
                "when_packed": False,
                "confidence": 0.98 if net_qty_present else 0.0,
                "bbox": [0.10, 0.55, 0.45, 0.60] if net_qty_present else None,
            },
            "consumer_care": {
                "status": "present" if care_present else "missing",
                "name": {"status": "present", "value": "Customer Care Officer", "confidence": 0.90} if care_present else {"status": "missing"},
                "address": {"status": "present", "value": "14 MG Road, Pune", "confidence": 0.90} if care_present else {"status": "missing"},
                "phone": {"status": "present", "value": "1800-123-4567", "confidence": 0.95} if care_present else {"status": "missing"},
                "email": {"status": "present", "value": "care@abcfoods.in", "confidence": 0.95} if care_present else {"status": "missing"},
                "raw_text": "Customer Care: care@abcfoods.in, 1800-123-4567" if care_present else None,
                "confidence": 0.95 if care_present else 0.0,
                "bbox": [0.10, 0.80, 0.90, 0.88] if care_present else None,
            },
            "stickers": [],
            "additional_numeric_information": [],
            "other_visible_text": [],
        },
    }


# ==============================================================================
# TEST SUITE: RULE 6 DECLARATION VISIBILITY
# ==============================================================================

def test_all_six_declarations_detected():
    """All 6 declarations present in image -> all return 'present' / 'Detected'."""
    service = Rule6Service()
    payload = make_full_extraction_payload()

    res = service.evaluate(payload)
    assert res.rule == "Rule 6 — Declaration Visibility"
    assert res.summary.total_declarations == 6
    assert res.summary.detected_count == 6
    assert res.summary.not_visible_count == 0
    assert res.summary.not_applicable_count == 0

    f_map = {f.field: f for f in res.findings}
    for field in [
        "manufacturer_packer_importer",
        "generic_name",
        "mrp",
        "date_of_manufacture_packing_import",
        "net_quantity",
        "consumer_care",
    ]:
        assert f_map[field].status == "present"
        assert f_map[field].display_status == "Detected"
        assert f_map[field].raw_text is not None


def test_one_declaration_not_visible():
    """Consumer care absent from image -> marked 'not_visible' / 'Not visible in supplied image'."""
    service = Rule6Service()
    payload = make_full_extraction_payload(care_present=False)

    res = service.evaluate(payload)
    assert res.summary.detected_count == 5
    assert res.summary.not_visible_count == 1

    f_map = {f.field: f for f in res.findings}
    assert f_map["consumer_care"].status == "not_visible"
    assert f_map["consumer_care"].display_status == "Not visible in supplied image"
    assert f_map["consumer_care"].raw_text is None

    # Other 5 remain detected
    assert f_map["manufacturer_packer_importer"].status == "present"
    assert f_map["generic_name"].status == "present"
    assert f_map["mrp"].status == "present"
    assert f_map["date_of_manufacture_packing_import"].status == "present"
    assert f_map["net_quantity"].status == "present"


def test_multiple_declarations_not_visible():
    """MRP and Dates not visible -> both marked 'not_visible', others detected."""
    service = Rule6Service()
    payload = make_full_extraction_payload(mrp_present=False, dates_present=False)

    res = service.evaluate(payload)
    assert res.summary.detected_count == 4
    assert res.summary.not_visible_count == 2

    f_map = {f.field: f for f in res.findings}
    assert f_map["mrp"].status == "not_visible"
    assert f_map["mrp"].display_status == "Not visible in supplied image"
    assert f_map["date_of_manufacture_packing_import"].status == "not_visible"
    assert f_map["date_of_manufacture_packing_import"].display_status == "Not visible in supplied image"


def test_all_declarations_not_visible_on_empty_payload():
    """Empty extraction -> all 6 declarations return 'not_visible' / 'Not visible in supplied image'."""
    service = Rule6Service()
    res = service.evaluate({})

    assert res.summary.total_declarations == 6
    assert res.summary.detected_count == 0
    assert res.summary.not_visible_count == 6

    for finding in res.findings:
        assert finding.status == "not_visible"
        assert finding.display_status == "Not visible in supplied image"
        assert finding.raw_text is None


def test_manufacturer_packer_importer_variants():
    """Entity with packer role or importer role is properly detected."""
    service = Rule6Service()
    payload = make_full_extraction_payload()

    # Modify manufacturer entity to 'packer'
    payload["semantic_extraction"]["manufacturer"]["entities"][0]["role"] = "packer"
    payload["semantic_extraction"]["manufacturer"]["entities"][0]["name"] = "Packer Logistics Ltd."
    res = service.evaluate(payload)

    f_map = {f.field: f for f in res.findings}
    mfr_finding = f_map["manufacturer_packer_importer"]
    assert mfr_finding.status == "present"
    assert mfr_finding.display_status == "Detected"
    assert "packer" in mfr_finding.notes.lower()


def test_date_variants_manufacture_packing_import():
    """Dates with manufacture, packing, or import date_type count as applicable date declaration."""
    service = Rule6Service()
    payload = make_full_extraction_payload(dates_present=False)

    # 1. Manufacture date
    payload["semantic_extraction"]["dates"] = [
        {
            "status": "present",
            "date_type": "manufacture",
            "raw_text": "Mfg: 01/2026",
            "month": "01",
            "year": 2026,
            "confidence": 0.94,
        }
    ]
    res1 = service.evaluate(payload)
    f_map1 = {f.field: f for f in res1.findings}
    assert f_map1["date_of_manufacture_packing_import"].status == "present"
    assert "01/2026" in f_map1["date_of_manufacture_packing_import"].raw_text

    # 2. Combined manufacture & packing dates
    payload["semantic_extraction"]["dates"] = [
        {
            "status": "present",
            "date_type": "manufacture",
            "raw_text": "Mfg: 01/2026",
            "month": "01",
            "year": 2026,
            "confidence": 0.94,
        },
        {
            "status": "present",
            "date_type": "packing",
            "raw_text": "Pkd: 02/2026",
            "month": "02",
            "year": 2026,
            "confidence": 0.92,
        },
    ]
    res2 = service.evaluate(payload)
    f_map2 = {f.field: f for f in res2.findings}
    assert f_map2["date_of_manufacture_packing_import"].status == "present"
    assert "Mfg: 01/2026" in f_map2["date_of_manufacture_packing_import"].raw_text
    assert "Pkd: 02/2026" in f_map2["date_of_manufacture_packing_import"].raw_text


def test_consumer_care_partial_subfields():
    """Consumer care with only phone or email is detected, preserving subfield notes."""
    service = Rule6Service()
    payload = make_full_extraction_payload(care_present=False)

    # Supply only phone number
    payload["semantic_extraction"]["consumer_care"] = {
        "status": "present",
        "phone": {"status": "present", "value": "1800-999-888", "confidence": 0.97},
        "email": {"status": "missing"},
        "address": {"status": "missing"},
        "name": {"status": "missing"},
        "raw_text": "Call Toll Free: 1800-999-888",
    }
    res = service.evaluate(payload)
    f_map = {f.field: f for f in res.findings}
    care_finding = f_map["consumer_care"]
    assert care_finding.status == "present"
    assert care_finding.display_status == "Detected"
    assert "phone" in care_finding.notes.lower()


def test_no_false_non_compliant_wording():
    """CRITICAL TEST: Verify user-facing display never uses 'Missing' or 'Non-compliant'."""
    service = Rule6Service()
    # Payload with all absent declarations
    res = service.evaluate({})

    for finding in res.findings:
        # Status must be 'not_visible', display must be 'Not visible in supplied image'
        assert finding.display_status == "Not visible in supplied image"
        assert finding.display_status != "Missing"
        assert finding.display_status != "Non-compliant"
        assert finding.status != "missing"
        assert finding.status != "non_compliant"


def test_lays_extraction_regression_compatibility():
    """Verify real-world Lay's extraction pattern evaluates cleanly against Rule 6."""
    service = Rule6Service()
    lays_payload = {
        "image": {"width": 1200, "height": 1600, "coordinate_system": "normalized_0_1"},
        "raw_text_blocks": [{"text": "Lay's Classic", "bbox": [0.1, 0.1, 0.9, 0.2]}],
        "semantic_extraction": {
            "image_quality": {"overall": "good"},
            "package": {"package_type": "retail", "brand_name": "Lay's", "generic_name": "Potato Chips"},
            "manufacturer": {
                "status": "present",
                "entities": [
                    {
                        "role": "manufacturer",
                        "name": "PepsiCo India Holdings Pvt. Ltd.",
                        "address": "Gurugram, Haryana",
                        "raw_text": "Mfg by PepsiCo India Holdings Pvt. Ltd.",
                        "confidence": 0.97,
                        "bbox": [0.10, 0.70, 0.90, 0.80],
                    }
                ],
            },
            "generic_name": {
                "status": "present",
                "raw_text": "Potato Chips",
                "value": "Potato Chips",
                "confidence": 0.98,
                "bbox": [0.20, 0.20, 0.80, 0.25],
            },
            "mrp": {
                "status": "present",
                "raw_text": "MRP ₹20.00",
                "value": 20.0,
                "currency": "₹",
                "confidence": 0.99,
                "bbox": [0.10, 0.45, 0.45, 0.50],
            },
            "dates": [
                {
                    "status": "present",
                    "date_type": "packing",
                    "raw_text": "Pkd: 15/08/2026",
                    "month": "08",
                    "year": 2026,
                    "confidence": 0.96,
                    "bbox": [0.50, 0.45, 0.85, 0.50],
                }
            ],
            "net_quantity": {
                "status": "present",
                "raw_text": "Net Qty: 52 g",
                "value": 52.0,
                "unit": "g",
                "quantity_type": "weight",
                "confidence": 0.99,
                "bbox": [0.10, 0.55, 0.45, 0.60],
            },
            "consumer_care": {
                "status": "present",
                "phone": {"status": "present", "value": "1800-222-444", "confidence": 0.95},
                "email": {"status": "present", "value": "feedback@pepsico.com", "confidence": 0.95},
                "address": {"status": "present", "value": "PO Box 27, New Delhi", "confidence": 0.95},
                "name": {"status": "present", "value": "Feedback Manager", "confidence": 0.95},
                "raw_text": "Contact Feedback Manager 1800-222-444",
                "confidence": 0.95,
                "bbox": [0.10, 0.82, 0.90, 0.90],
            },
            "stickers": [],
            "additional_numeric_information": [],
            "other_visible_text": [],
        },
    }

    res = service.evaluate(lays_payload)
    assert res.summary.detected_count == 6
    assert res.summary.not_visible_count == 0


# ==============================================================================
# API ENDPOINT TESTS: POST /api/v1/compliance/rule6
# ==============================================================================

def test_api_rule6_endpoint_success():
    """Test POST /api/v1/compliance/rule6 returns successful 200 response."""
    client = TestClient(app)
    payload = make_full_extraction_payload()

    response = client.post("/api/v1/compliance/rule6", json={"extraction": payload})
    assert response.status_code == 200

    data = response.json()
    assert data["success"] is True
    assert "data" in data
    assert data["data"]["summary"]["total_declarations"] == 6
    assert data["data"]["summary"]["detected_count"] == 6

    findings = data["data"]["findings"]
    f_map = {f["field"]: f for f in findings}
    assert f_map["net_quantity"]["display_status"] == "Detected"
    assert f_map["generic_name"]["display_status"] == "Detected"


def test_api_rule6_endpoint_with_partial_visibility():
    """Test POST /api/v1/compliance/rule6 with partially visible declarations."""
    client = TestClient(app)
    payload = make_full_extraction_payload(care_present=False, mfr_present=False)

    response = client.post("/api/v1/compliance/rule6", json={"extraction": payload})
    assert response.status_code == 200

    data = response.json()
    assert data["data"]["summary"]["detected_count"] == 4
    assert data["data"]["summary"]["not_visible_count"] == 2

    findings = data["data"]["findings"]
    f_map = {f["field"]: f for f in findings}
    assert f_map["consumer_care"]["display_status"] == "Not visible in supplied image"
    assert f_map["manufacturer_packer_importer"]["display_status"] == "Not visible in supplied image"


def test_api_rule6_endpoint_invalid_json_string():
    """Test POST /api/v1/compliance/rule6 with malformed JSON string returns 422."""
    client = TestClient(app)
    response = client.post("/api/v1/compliance/rule6", json={"extraction": "{malformed json"})
    assert response.status_code == 422
    data = response.json()
    assert data["success"] is False
    assert data["error"]["code"] == "INVALID_EXTRACTION_JSON"
