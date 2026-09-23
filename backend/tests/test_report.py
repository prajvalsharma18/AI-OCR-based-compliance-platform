"""Unit and integration tests for PDF Inspection Report Generator (ReportLab).

SIH 2026 PS 26034: Automated Compliance Checker for Packaged Commodities

Covers all 26 test conditions:
1. PDF generation from InspectionResponse
2. PDF generation from InspectionResponseData
3. PDF generation from dictionary
4. PDF generation from JSON string
5. PDF generation without image
6. PDF generation with PNG image
7. PDF generation with JPEG image
8. NOT VISIBLE rendering
9. NOT ASSESSABLE rendering
10. NON-COMPLIANT rendering
11. PASS rendering
12. Rule 7 table rendering
13. Rule 8 table rendering
14. Summary rendering
15. Disclaimer rendering
16. Multi-page report generation
17. Long notes wrap safely
18. Missing metadata uses N/A
19. Missing optional evidence does not crash
20. Corrupted image does not crash the whole report
21. Input inspection object is not mutated
22. Rule results are rendered exactly as supplied
23. No status recalculation
24. No summary recalculation
25. No OpenAI/LLM calls
26. No OCR/CV calls
"""

import base64
import copy
import io
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage
import pypdf

from app.main import app
from app.schemas.inspection import (
    InspectionResponse,
    InspectionResponseData,
)
from app.services.image_storage_service import ImageStorageService
from app.services.inspection_service import InspectionService
from app.services.report_service import ReportService, ReportServiceError


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture
def report_service():
    """ReportService instance."""
    return ReportService()


@pytest.fixture
def sample_inspection_data():
    """Representative Unified Inspection JSON model data (Lay's pattern)."""
    return {
        "inspection_id": "INSP-TEST-12345",
        "metadata": {
            "brand_name": "Lay's",
            "generic_name": "Potato Chips",
            "package_type": "retail",
            "image_quality": "good",
            "package_dimensions_mm": {
                "width_mm": 150.0,
                "height_mm": 200.0,
            },
            "pdp_area_cm2": 300.0,
            "rule_source_mode": "sih_ps_26034",
        },
        "summary": {
            "total_declarations_evaluated": 7,
            "detected_declarations_count": 6,
            "not_visible_declarations_count": 1,
            "rules_evaluated_count": 5,
            "pass_findings_count": 3,
            "non_compliant_findings_count": 2,
            "not_assessable_findings_count": 0,
        },
        "findings": [
            {
                "field": "manufacturer_packer_importer",
                "display_name": "Manufacturer / Packer / Importer",
                "visibility": "DETECTED",
                "raw_text": "Mfg by PepsiCo India Holdings Pvt. Ltd., Village Channo, Patiala 147001",
                "detected_value": None,
                "declared_numeral": None,
                "declared_unit": None,
                "confidence": 0.98,
                "bbox": [0.10, 0.70, 0.90, 0.75],
                "rule_reference": "Rule 6(1)(a)",
                "status": "NOT ASSESSABLE",
                "rules": {"rule7": None, "rule8": None},
                "notes": "Declared entity name and address detected.",
            },
            {
                "field": "generic_name",
                "display_name": "Generic Name",
                "visibility": "DETECTED",
                "raw_text": "POTATO CHIPS",
                "detected_value": "Potato Chips",
                "declared_numeral": None,
                "declared_unit": None,
                "confidence": 0.95,
                "bbox": [0.25, 0.30, 0.75, 0.35],
                "rule_reference": "Rule 6(1)(b)",
                "status": "NOT ASSESSABLE",
                "rules": {"rule7": None, "rule8": None},
                "notes": "Generic name declared: 'Potato Chips'.",
            },
            {
                "field": "mrp",
                "display_name": "Retail Sale Price (MRP)",
                "visibility": "DETECTED",
                "raw_text": "MRP Rs. 10.00 (Incl. of all taxes)",
                "detected_value": "₹ 10.00",
                "declared_numeral": "10",
                "declared_unit": "₹",
                "confidence": 0.96,
                "bbox": [0.60, 0.80, 0.85, 0.85],
                "rule_reference": "Rule 6(1)(c)",
                "status": "NON-COMPLIANT",
                "rules": {
                    "rule7": {
                        "status": "NON-COMPLIANT",
                        "rule_reference": "SIH 2026 PS 26034 Technical Specification",
                        "rule_source_mode": "sih_ps_26034",
                        "rule_type": "general_declaration",
                        "measured_height_mm": 1.65,
                        "required_height_mm": 2.00,
                        "height_margin_mm": -0.35,
                        "measurement_quality": "good",
                        "confidence": 0.92,
                        "notes": "MRP numeral below required height.",
                    },
                    "rule8": None,
                },
                "notes": "Declared MRP: ₹ 10.00 (Inclusive of all taxes).",
            },
            {
                "field": "date_of_manufacture",
                "display_name": "Manufacturing / Packing Date",
                "visibility": "DETECTED",
                "raw_text": "MFD: 26/07/26",
                "detected_value": "07/2026",
                "declared_numeral": "26/07/26",
                "declared_unit": None,
                "confidence": 0.94,
                "bbox": [0.10, 0.80, 0.40, 0.83],
                "rule_reference": "Rule 6(1)(d)",
                "status": "PASS",
                "rules": {
                    "rule7": {
                        "status": "PASS",
                        "rule_reference": "SIH 2026 PS 26034 Technical Specification",
                        "rule_source_mode": "sih_ps_26034",
                        "rule_type": "general_declaration",
                        "measured_height_mm": 2.00,
                        "required_height_mm": 2.00,
                        "height_margin_mm": 0.00,
                        "measurement_quality": "good",
                        "confidence": 0.90,
                        "notes": "Manufacturing date numeral meets requirement.",
                    },
                    "rule8": None,
                },
                "notes": "Declared manufacturing date detected.",
            },
            {
                "field": "date_use_by",
                "display_name": "Best Before / Use By Date",
                "visibility": "DETECTED",
                "raw_text": "USE BY: 08/12/26",
                "detected_value": "12/2026",
                "declared_numeral": "08/12/26",
                "declared_unit": None,
                "confidence": 0.93,
                "bbox": [0.10, 0.84, 0.40, 0.87],
                "rule_reference": "Rule 6(1)(d)",
                "status": "PASS",
                "rules": {
                    "rule7": {
                        "status": "PASS",
                        "rule_reference": "SIH 2026 PS 26034 Technical Specification",
                        "rule_source_mode": "sih_ps_26034",
                        "rule_type": "general_declaration",
                        "measured_height_mm": 2.00,
                        "required_height_mm": 2.00,
                        "height_margin_mm": 0.00,
                        "measurement_quality": "good",
                        "confidence": 0.89,
                        "notes": "Use by date numeral meets requirement.",
                    },
                    "rule8": None,
                },
                "notes": "Declared best before / use by date detected.",
            },
            {
                "field": "net_quantity",
                "display_name": "Net Quantity",
                "visibility": "DETECTED",
                "raw_text": "Net Wt. 42 g",
                "detected_value": "42.0 g",
                "declared_numeral": "42",
                "declared_unit": "g",
                "confidence": 0.97,
                "bbox": [0.15, 0.90, 0.40, 0.95],
                "rule_reference": "Rule 6(1)(e)",
                "status": "NON-COMPLIANT",
                "rules": {
                    "rule7": {
                        "status": "PASS",
                        "rule_reference": "SIH 2026 PS 26034 Technical Specification",
                        "rule_source_mode": "sih_ps_26034",
                        "rule_type": "quantity_weight_volume",
                        "measured_height_mm": 2.50,
                        "required_height_mm": 1.00,
                        "height_margin_mm": 1.50,
                        "measurement_quality": "good",
                        "confidence": 0.95,
                        "notes": "Quantity numeral exceeds requirement.",
                    },
                    "rule8": {
                        "status": "NON-COMPLIANT",
                        "rule_reference": "Legal Metrology Rules, 2011 — Rule 8(1) proviso",
                        "clearance_reference": "quantity_declaration_bbox",
                        "target_numeral_height_mm": 2.50,
                        "top": {"measured_mm": 6.85, "required_mm": 2.50, "status": "above_threshold"},
                        "bottom": {"measured_mm": 0.00, "required_mm": 2.50, "status": "below_threshold"},
                        "left": {"measured_mm": 12.00, "required_mm": 5.00, "status": "above_threshold"},
                        "right": {"measured_mm": 93.00, "required_mm": 5.00, "status": "above_threshold"},
                        "confidence": 0.90,
                        "notes": "Insufficient clear space detected in bottom direction.",
                    },
                },
                "notes": "Declared net quantity: 42.0 g.",
            },
            {
                "field": "consumer_care",
                "display_name": "Consumer Care Details",
                "visibility": "NOT VISIBLE",
                "raw_text": None,
                "detected_value": None,
                "declared_numeral": None,
                "declared_unit": None,
                "confidence": None,
                "bbox": None,
                "rule_reference": "Rule 6(1)(f)",
                "status": "NOT VISIBLE",
                "rules": {"rule7": None, "rule8": None},
                "notes": "Not visible in supplied image. Other package panels may contain the declaration.",
            },
        ],
        "disclaimer": "This inspection summarizes findings detected from the supplied package image(s). 'Not visible in supplied image' does not establish legal absence. Final enforcement and legal determination rests with the competent Legal Metrology Officer.",
    }


def make_test_image_base64(format="PNG", color=(255, 200, 0)) -> str:
    """Helper to create small valid in-memory test image in base64."""
    img = PILImage.new("RGB", (200, 150), color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def read_pdf_text(pdf_path: str) -> str:
    """Extracts all text across all pages using pypdf, normalizing whitespace."""
    reader = pypdf.PdfReader(pdf_path)
    full_text = []
    for page in reader.pages:
        full_text.append(page.extract_text() or "")
    return " ".join("\n".join(full_text).split())


# ==============================================================================
# TEST CASES
# ==============================================================================

# 1. PDF generation from InspectionResponse
def test_1_pdf_from_inspection_response(report_service, sample_inspection_data, tmp_path):
    resp = InspectionResponse(success=True, data=InspectionResponseData.model_validate(sample_inspection_data))
    out = str(tmp_path / "test1.pdf")
    path = report_service.generate_report(resp, output_path=out)
    assert Path(path).exists()
    reader = pypdf.PdfReader(path)
    assert len(reader.pages) >= 1


# 2. PDF generation from InspectionResponseData
def test_2_pdf_from_inspection_response_data(report_service, sample_inspection_data, tmp_path):
    data_obj = InspectionResponseData.model_validate(sample_inspection_data)
    out = str(tmp_path / "test2.pdf")
    path = report_service.generate_report(data_obj, output_path=out)
    assert Path(path).exists()


# 3. PDF generation from dictionary
def test_3_pdf_from_dict(report_service, sample_inspection_data, tmp_path):
    out = str(tmp_path / "test3.pdf")
    path = report_service.generate_report(sample_inspection_data, output_path=out)
    assert Path(path).exists()


# 4. PDF generation from JSON string
def test_4_pdf_from_json_string(report_service, sample_inspection_data, tmp_path):
    json_str = json.dumps(sample_inspection_data)
    out = str(tmp_path / "test4.pdf")
    path = report_service.generate_report(json_str, output_path=out)
    assert Path(path).exists()


# 5. PDF generation without image
def test_5_pdf_without_image(report_service, sample_inspection_data, tmp_path):
    out = str(tmp_path / "test5.pdf")
    path = report_service.generate_report(sample_inspection_data, image_base64=None, output_path=out)
    assert Path(path).exists()
    text = read_pdf_text(path)
    assert "AUTOMATED COMPLIANCE INSPECTION REPORT" in text


# 6. PDF generation with PNG image
def test_6_pdf_with_png_image(report_service, sample_inspection_data, tmp_path):
    png_b64 = make_test_image_base64("PNG")
    out = str(tmp_path / "test6.pdf")
    path = report_service.generate_report(sample_inspection_data, image_base64=png_b64, output_path=out)
    assert Path(path).exists()
    text = read_pdf_text(path)
    assert "SOURCE PACKAGE IMAGE EVIDENCE" in text


# 7. PDF generation with JPEG image
def test_7_pdf_with_jpeg_image(report_service, sample_inspection_data, tmp_path):
    jpeg_b64 = make_test_image_base64("JPEG")
    out = str(tmp_path / "test7.pdf")
    path = report_service.generate_report(sample_inspection_data, image_base64=jpeg_b64, output_path=out)
    assert Path(path).exists()
    text = read_pdf_text(path)
    assert "SOURCE PACKAGE IMAGE EVIDENCE" in text


# 8. NOT VISIBLE rendering
def test_8_not_visible_rendering(report_service, sample_inspection_data, tmp_path):
    out = str(tmp_path / "test8.pdf")
    path = report_service.generate_report(sample_inspection_data, output_path=out)
    text = read_pdf_text(path)
    assert "NOT VISIBLE" in text
    assert "Consumer Care Details" in text
    assert "Other package panels may contain the declaration" in text


# 9. NOT ASSESSABLE rendering
def test_9_not_assessable_rendering(report_service, sample_inspection_data, tmp_path):
    out = str(tmp_path / "test9.pdf")
    path = report_service.generate_report(sample_inspection_data, output_path=out)
    text = read_pdf_text(path)
    assert "NOT ASSESSABLE" in text
    assert "Manufacturer / Packer / Importer" in text
    assert "Generic Name" in text


# 10. NON-COMPLIANT rendering
def test_10_non_compliant_rendering(report_service, sample_inspection_data, tmp_path):
    out = str(tmp_path / "test10.pdf")
    path = report_service.generate_report(sample_inspection_data, output_path=out)
    text = read_pdf_text(path)
    assert "NON-COMPLIANT" in text


# 11. PASS rendering
def test_11_pass_rendering(report_service, sample_inspection_data, tmp_path):
    out = str(tmp_path / "test11.pdf")
    path = report_service.generate_report(sample_inspection_data, output_path=out)
    text = read_pdf_text(path)
    assert "PASS" in text
    assert "Manufacturing / Packing Date" in text


# 12. Rule 7 table rendering
def test_12_rule7_table_rendering(report_service, sample_inspection_data, tmp_path):
    out = str(tmp_path / "test12.pdf")
    path = report_service.generate_report(sample_inspection_data, output_path=out)
    text = read_pdf_text(path)
    assert "3. RULE 7 NUMERAL-HEIGHT ASSESSMENT" in text
    assert "1.65 mm" in text
    assert "2.00 mm" in text
    assert "-0.35 mm" in text
    assert "sih_ps_26034" in text


# 13. Rule 8 table rendering
def test_13_rule8_table_rendering(report_service, sample_inspection_data, tmp_path):
    out = str(tmp_path / "test13.pdf")
    path = report_service.generate_report(sample_inspection_data, output_path=out)
    text = read_pdf_text(path)
    assert "4. RULE 8 SPATIAL CLEARANCE ASSESSMENT" in text
    assert "6.85 / 2.50 mm" in text
    assert "0.00 / 2.50 mm" in text
    assert "above threshold" in text or "below threshold" in text


# 14. Summary rendering
def test_14_summary_rendering(report_service, sample_inspection_data, tmp_path):
    out = str(tmp_path / "test14.pdf")
    path = report_service.generate_report(sample_inspection_data, output_path=out)
    text = read_pdf_text(path)
    assert "5. INSPECTION SUMMARY COUNTS" in text
    assert "Total Declarations" in text
    assert "Rules Evaluated" in text
    assert "PASS Rules" in text
    assert "NON-COMPLIANT Rules" in text


# 15. Disclaimer rendering
def test_15_disclaimer_rendering(report_service, sample_inspection_data, tmp_path):
    out = str(tmp_path / "test15.pdf")
    path = report_service.generate_report(sample_inspection_data, output_path=out)
    text = read_pdf_text(path)
    assert "LEGAL METROLOGY STATUTORY ADVISORY NOTICE" in text
    assert "Final enforcement and legal determination rests with the competent Legal Metrology Officer" in text


# 16. Multi-page report generation
def test_16_multipage_report(report_service, sample_inspection_data, tmp_path):
    # Add multiple repeated findings to trigger page break
    extended_data = copy.deepcopy(sample_inspection_data)
    for i in range(5):
        clone = copy.deepcopy(extended_data["findings"][0])
        clone["field"] = f"extra_item_{i}"
        clone["display_name"] = f"Extra Commodity Item {i}"
        extended_data["findings"].append(clone)

    out = str(tmp_path / "multipage.pdf")
    path = report_service.generate_report(extended_data, output_path=out)
    reader = pypdf.PdfReader(path)
    assert len(reader.pages) >= 2

    # Check footer on page 1 and page 2
    p1_text = reader.pages[0].extract_text()
    p2_text = reader.pages[1].extract_text()
    assert "Page 1 of" in p1_text
    assert "Page 2 of" in p2_text
    assert "INSP-TEST-12345" in p1_text
    assert "INSP-TEST-12345" in p2_text


# 17. Long notes wrap safely
def test_17_long_notes_wrap(report_service, sample_inspection_data, tmp_path):
    long_data = copy.deepcopy(sample_inspection_data)
    long_data["findings"][0]["notes"] = (
        "This is an exceptionally long legal observation note regarding the manufacturer and packer "
        "address declarations. Under Section 18 of the Legal Metrology Act, 2009 and Rule 6(1)(a) of the "
        "Packaged Commodities Rules, 2011, every package must clearly disclose the name and complete address "
        "including pin code, contact details, and registered office where the commodity was packed or manufactured."
    )
    out = str(tmp_path / "long_notes.pdf")
    path = report_service.generate_report(long_data, output_path=out)
    assert Path(path).exists()
    text = read_pdf_text(path)
    assert "Section 18 of the Legal Metrology Act" in text


# 18. Missing metadata uses N/A
def test_18_missing_metadata_uses_na(report_service, sample_inspection_data, tmp_path):
    sparse_data = copy.deepcopy(sample_inspection_data)
    sparse_data["metadata"]["brand_name"] = None
    sparse_data["metadata"]["generic_name"] = None
    sparse_data["metadata"]["package_dimensions_mm"] = None
    sparse_data["metadata"]["pdp_area_cm2"] = None

    out = str(tmp_path / "sparse.pdf")
    path = report_service.generate_report(sparse_data, output_path=out)
    text = read_pdf_text(path)
    assert "N/A" in text
    assert "None" not in text


# 19. Missing optional evidence does not crash
def test_19_missing_optional_evidence(report_service, sample_inspection_data, tmp_path):
    stripped_data = copy.deepcopy(sample_inspection_data)
    for f in stripped_data["findings"]:
        f["bbox"] = None
        f["confidence"] = None
        f["raw_text"] = None
        f["detected_value"] = None

    out = str(tmp_path / "stripped.pdf")
    path = report_service.generate_report(stripped_data, output_path=out)
    assert Path(path).exists()


# 20. Corrupted image does not crash the whole report
def test_20_corrupted_image_handling(report_service, sample_inspection_data, tmp_path):
    corrupted_b64 = "not-a-valid-base64-image-string!!!"
    out = str(tmp_path / "corrupted_img.pdf")
    path = report_service.generate_report(sample_inspection_data, image_base64=corrupted_b64, output_path=out)
    assert Path(path).exists()
    text = read_pdf_text(path)
    assert "Source image could not be rendered" in text


# 21. Input inspection object is not mutated
def test_21_input_not_mutated(report_service, sample_inspection_data, tmp_path):
    orig = copy.deepcopy(sample_inspection_data)
    out = str(tmp_path / "immutability.pdf")
    report_service.generate_report(sample_inspection_data, output_path=out)
    assert sample_inspection_data == orig


# 22. Rule results are rendered exactly as supplied
def test_22_rule_results_exact_rendering(report_service, sample_inspection_data, tmp_path):
    out = str(tmp_path / "exact.pdf")
    path = report_service.generate_report(sample_inspection_data, output_path=out)
    text = read_pdf_text(path)
    assert "1.65 mm" in text
    assert "2.00 mm" in text
    assert "-0.35 mm" in text
    assert "6.85 / 2.50 mm" in text


# 23. No status recalculation
def test_23_no_status_recalculation(report_service, sample_inspection_data, tmp_path):
    # If upstream says measured 999.0 mm but status is NON-COMPLIANT
    tampered = copy.deepcopy(sample_inspection_data)
    tampered["findings"][2]["rules"]["rule7"]["measured_height_mm"] = 999.0
    tampered["findings"][2]["rules"]["rule7"]["required_height_mm"] = 2.0
    tampered["findings"][2]["rules"]["rule7"]["status"] = "NON-COMPLIANT"

    out = str(tmp_path / "no_recalc.pdf")
    path = report_service.generate_report(tampered, output_path=out)
    text = read_pdf_text(path)
    assert "999.00 mm" in text
    assert "NON-COMPLIANT" in text


# 24. No summary recalculation
def test_24_no_summary_recalculation(report_service, sample_inspection_data, tmp_path):
    # Set artificial summary count 777
    tampered = copy.deepcopy(sample_inspection_data)
    tampered["summary"]["pass_findings_count"] = 777

    out = str(tmp_path / "summary_preserve.pdf")
    path = report_service.generate_report(tampered, output_path=out)
    text = read_pdf_text(path)
    assert "777" in text


# 25. No OpenAI/LLM calls in report service
def test_25_no_llm_invocation(report_service, sample_inspection_data, monkeypatch, tmp_path):
    # Monkeypatch openai or LLM callers if present to raise
    def fail_call(*args, **kwargs):
        raise RuntimeError("LLM must not be called by PDF report generator!")

    monkeypatch.setattr("openai.resources.chat.completions.Completions.create", fail_call, raising=False)
    out = str(tmp_path / "no_llm.pdf")
    path = report_service.generate_report(sample_inspection_data, output_path=out)
    assert Path(path).exists()


# 26. No OCR/CV calls in report service
def test_26_no_ocr_cv_invocation(report_service, sample_inspection_data, monkeypatch, tmp_path):
    # Monkeypatch cv2 functions to ensure no OpenCV passes
    def fail_cv(*args, **kwargs):
        raise RuntimeError("OpenCV must not be called by PDF report generator!")

    monkeypatch.setattr("cv2.findContours", fail_cv, raising=False)
    monkeypatch.setattr("cv2.threshold", fail_cv, raising=False)

    out = str(tmp_path / "no_cv.pdf")
    path = report_service.generate_report(sample_inspection_data, output_path=out)
    assert Path(path).exists()


# API Endpoint Integration Test
def test_api_report_pdf_endpoint(client, sample_inspection_data):
    """Integration test for POST /api/v1/report/pdf."""
    payload = {
        "inspection": sample_inspection_data,
        "image_base64": make_test_image_base64("PNG"),
    }
    response = client.post("/api/v1/report/pdf", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "inspection_INSP-TEST-12345.pdf" in response.headers["content-disposition"]

    # Verify returned PDF content with pypdf
    reader = pypdf.PdfReader(io.BytesIO(response.content))
    assert len(reader.pages) >= 1
    text = "".join(p.extract_text() or "" for p in reader.pages)
    assert "AUTOMATED COMPLIANCE INSPECTION REPORT" in text
    assert "Lay's" in text
    assert "NON-COMPLIANT" in text


# 27. Stored original image is used when available without image_base64
def test_27_stored_original_image_used(sample_inspection_data, tmp_path):
    storage = ImageStorageService(storage_root=tmp_path / "reports" / "inspections")
    service = ReportService(storage_service=storage)

    insp_data = copy.deepcopy(sample_inspection_data)
    insp_id = "INSP-STORED-27"
    insp_data["inspection_id"] = insp_id

    # Create and save Image A
    img_a_b64 = make_test_image_base64("PNG", color=(10, 180, 50))
    img_a_bytes = base64.b64decode(img_a_b64)
    storage.save_original(img_a_bytes, "img_a.png", insp_id)

    sha_a = storage.calculate_sha256(img_a_bytes)
    insp_data["metadata"]["source_image_sha256"] = sha_a
    insp_data["metadata"]["source_image_filename"] = "img_a.png"

    out = str(tmp_path / "test27.pdf")
    # Call without image_base64
    path = service.generate_report(insp_data, image_base64=None, output_path=out)
    assert Path(path).exists()

    text = read_pdf_text(path)
    assert "SOURCE PACKAGE IMAGE EVIDENCE" in text
    assert "ORIGINAL STORED IMAGE" in text
    assert "MATCHED" in text
    assert sha_a[:16] in text


# 28. Image A / Image B Protection: Stored image takes precedence over image_base64
def test_28_image_a_vs_image_b_protection(sample_inspection_data, tmp_path):
    storage = ImageStorageService(storage_root=tmp_path / "reports" / "inspections")
    service = ReportService(storage_service=storage)

    insp_data = copy.deepcopy(sample_inspection_data)
    insp_id = "INSP-PROT-28"
    insp_data["inspection_id"] = insp_id

    # Image A: 200x150 Green
    img_a_b64 = make_test_image_base64("PNG", color=(0, 255, 0))
    img_a_bytes = base64.b64decode(img_a_b64)
    storage.save_original(img_a_bytes, "stored_original.png", insp_id)

    sha_a = storage.calculate_sha256(img_a_bytes)
    insp_data["metadata"]["source_image_sha256"] = sha_a

    # Image B: 120x80 Blue passed later as image_base64
    img_b = PILImage.new("RGB", (120, 80), color=(0, 0, 255))
    buf_b = io.BytesIO()
    img_b.save(buf_b, format="PNG")
    img_b_b64 = base64.b64encode(buf_b.getvalue()).decode("utf-8")

    out = str(tmp_path / "test28_protected.pdf")
    # Pass Image B in call; ReportService MUST ignore Image B and use stored Image A
    path = service.generate_report(insp_data, image_base64=img_b_b64, output_path=out)
    assert Path(path).exists()

    text = read_pdf_text(path)
    assert "ORIGINAL STORED IMAGE" in text
    assert "MATCHED" in text

    # Extract embedded image from the PDF to prove Image A was used (not Image B)
    reader = pypdf.PdfReader(path)
    extracted_images = []
    for page in reader.pages:
        for img in page.images:
            extracted_images.append(img)

    assert len(extracted_images) >= 1
    # Check dimensions of embedded image (Image A was 200x150, Image B was 120x80)
    embedded_img = PILImage.open(io.BytesIO(extracted_images[0].data))
    assert embedded_img.size == (200, 150)
    assert embedded_img.size != (120, 80)


# 29. SHA-256 Mismatch Stops Report Generation with typed 422 error
def test_29_hash_mismatch_stops_report(sample_inspection_data, tmp_path):
    storage = ImageStorageService(storage_root=tmp_path / "reports" / "inspections")
    service = ReportService(storage_service=storage)

    insp_data = copy.deepcopy(sample_inspection_data)
    insp_id = "INSP-MISMATCH-29"
    insp_data["inspection_id"] = insp_id

    # Store Image A
    img_bytes = base64.b64decode(make_test_image_base64("PNG"))
    storage.save_original(img_bytes, "img.png", insp_id)

    # Tamper with expected SHA-256
    insp_data["metadata"]["source_image_sha256"] = "0" * 64

    out = str(tmp_path / "test29_fail.pdf")
    with pytest.raises(ReportServiceError) as exc_info:
        service.generate_report(insp_data, output_path=out)

    assert exc_info.value.code == "SOURCE_IMAGE_HASH_MISMATCH"
    assert "Stored source image does not match" in exc_info.value.message
    assert exc_info.value.status_code == 422
    assert not Path(out).exists()


# 30. Missing hash supports legacy UNVERIFIED mode
def test_30_missing_hash_supports_unverified_mode(sample_inspection_data, tmp_path):
    storage = ImageStorageService(storage_root=tmp_path / "reports" / "inspections")
    service = ReportService(storage_service=storage)

    insp_data = copy.deepcopy(sample_inspection_data)
    insp_id = "INSP-UNVERIFIED-30"
    insp_data["inspection_id"] = insp_id

    # Store Image without hash in metadata
    img_bytes = base64.b64decode(make_test_image_base64("PNG"))
    storage.save_original(img_bytes, "legacy.png", insp_id)
    insp_data["metadata"]["source_image_sha256"] = None

    out = str(tmp_path / "test30_unverified.pdf")
    path = service.generate_report(insp_data, output_path=out)
    assert Path(path).exists()

    text = read_pdf_text(path)
    assert "ORIGINAL STORED IMAGE" in text
    assert "UNVERIFIED" in text


# 31. Concise rule reference boxes and LMO review notes are rendered in PDF
def test_31_rule_references_and_lmo_notes(report_service, sample_inspection_data, tmp_path):
    out = str(tmp_path / "test31_refs.pdf")
    path = report_service.generate_report(sample_inspection_data, output_path=out)
    assert Path(path).exists()

    text = read_pdf_text(path)
    # Rule 6 Reference Box
    assert "APPLICABLE RULE REFERENCE" in text
    assert "Rule 6" in text
    assert "Mandatory Package Declarations" in text
    assert "Manufacturer/Packer/Importer" in text

    # Rule 7 Reference Box
    assert "Rule 7" in text
    assert "Letter and Numeral Size Requirements" in text
    assert "SIH PS 26034" in text
    assert "LMO Verification" in text

    # Rule 8 Reference Box
    assert "Rule 8(1) Proviso" in text
    assert "Clear Space Surrounding Net-Quantity" in text or "Clear Space" in text

    # LMO Review Notes Section
    assert "LMO REVIEW NOTES" in text
    assert "inspection-support evidence" in text
    assert "NOT VISIBLE" in text
