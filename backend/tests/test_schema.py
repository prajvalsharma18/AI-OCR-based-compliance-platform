"""Unit tests for Pydantic schemas and SIH PS 26034 extraction contracts."""

import pytest
from pydantic import ValidationError

from app.schemas.extraction import (
    ConsumerCareDeclaration,
    DateDeclaration,
    GenericNameDeclaration,
    ImageQuality,
    LabelExtractionResult,
    ManufacturerEntity,
    ManufacturerSection,
    MRPDeclaration,
    NetQuantityDeclaration,
    PackageContext,
)


def sample_valid_extraction_dict():
    """Returns a valid dictionary payload conforming to LabelExtractionResult."""
    return {
        "image_quality": {
            "overall": "good",
            "notes": "Sharp, clear lighting on front label."
        },
        "package": {
            "package_type": "retail",
            "brand_name": "ABC Brand",
            "generic_name": "Basmati Rice",
            "product_category": "rice",
            "food_status": "food",
            "retail_unit_count": None,
            "evidence": ["Single retail pouch", "Nutritional facts panel"]
        },
        "manufacturer": {
            "status": "present",
            "entities": [
                {
                    "role": "manufacturer",
                    "name": "ABC Foods Pvt. Ltd.",
                    "address": "14 MG Road, Pune 411001",
                    "raw_text": "Mfg by: ABC Foods Pvt. Ltd., 14 MG Road, Pune 411001",
                    "confidence": 0.98,
                    "bbox": [0.1, 0.7, 0.9, 0.8]
                }
            ]
        },
        "generic_name": {
            "status": "present",
            "raw_text": "Basmati Rice",
            "value": "Basmati Rice",
            "confidence": 0.95,
            "bbox": [0.2, 0.2, 0.8, 0.25]
        },
        "mrp": {
            "status": "present",
            "raw_text": "MRP ₹120.00 Inclusive of all taxes",
            "value": 120.00,
            "currency": "₹",
            "confidence": 0.96,
            "bbox": [0.1, 0.5, 0.5, 0.55],
            "numeral_region": [0.25, 0.5, 0.4, 0.55]
        },
        "dates": [
            {
                "status": "present",
                "date_type": "packing",
                "raw_text": "Pkd: 08/2026",
                "month": "08",
                "year": 2026,
                "confidence": 0.92,
                "bbox": [0.55, 0.5, 0.9, 0.55],
                "numeral_region": [0.65, 0.5, 0.85, 0.55]
            }
        ],
        "net_quantity": {
            "status": "present",
            "raw_text": "Net Qty: 500 g",
            "value": 500.0,
            "unit": "g",
            "quantity_type": "weight",
            "when_packed": False,
            "confidence": 0.99,
            "bbox": [0.1, 0.4, 0.5, 0.45],
            "numeral_region": [0.3, 0.4, 0.45, 0.45]
        },
        "consumer_care": {
            "status": "present",
            "name": "Customer Care Officer",
            "address": "Same as manufacturer address",
            "phone": "1800-123-4567",
            "email": "care@abcfoods.in",
            "raw_text": "Consumer Care: care@abcfoods.in, 1800-123-4567",
            "confidence": 0.94,
            "bbox": [0.1, 0.82, 0.9, 0.9]
        },
        "stickers": [],
        "additional_numeric_information": [
            {
                "semantic_role": "pin_code",
                "raw_text": "411001",
                "value": "411001",
                "bbox": None,
                "confidence": 0.99
            }
        ],
        "other_visible_text": [
            {
                "text_type": "promotional",
                "raw_text": "PREMIUM QUALITY AGED RICE",
                "bbox": None,
                "confidence": 0.9
            }
        ]
    }


def test_valid_extraction_schema():
    """Verify that a standard extraction response adheres strictly to the schema."""
    data = sample_valid_extraction_dict()
    model = LabelExtractionResult.model_validate(data)
    assert model.image_quality.overall == "good"
    assert model.package.generic_name == "Basmati Rice"
    assert model.mrp.value == 120.0
    assert model.net_quantity.unit == "g"
    assert model.dates[0].year == 2026


def test_missing_and_uncertain_statuses():
    """Verify tri-state status ('present', 'missing', 'uncertain')."""
    data = sample_valid_extraction_dict()
    data["mrp"]["status"] = "missing"
    data["mrp"]["value"] = None
    data["mrp"]["raw_text"] = None
    data["consumer_care"]["status"] = "uncertain"

    model = LabelExtractionResult.model_validate(data)
    assert model.mrp.status == "missing"
    assert model.consumer_care.status == "uncertain"


def test_invalid_status_literal():
    """Verify that unsupported status values raise ValidationError."""
    data = sample_valid_extraction_dict()
    data["net_quantity"]["status"] = "not_available"  # Invalid! Must be present/missing/uncertain

    with pytest.raises(ValidationError) as exc:
        LabelExtractionResult.model_validate(data)
    assert "net_quantity.status" in str(exc.value)


def test_confidence_bounds():
    """Verify confidence must be between 0.0 and 1.0."""
    data = sample_valid_extraction_dict()
    data["mrp"]["confidence"] = 1.5  # Invalid > 1.0

    with pytest.raises(ValidationError) as exc:
        LabelExtractionResult.model_validate(data)
    assert "mrp.confidence" in str(exc.value)

    data["mrp"]["confidence"] = -0.1  # Invalid < 0.0
    with pytest.raises(ValidationError) as exc:
        LabelExtractionResult.model_validate(data)
    assert "mrp.confidence" in str(exc.value)


def test_bbox_validation():
    """Verify bounding boxes accept 4 numbers, null, or reject invalid formats."""
    data = sample_valid_extraction_dict()

    # Valid: 4 floats (normalized 0..1)
    data["mrp"]["bbox"] = [0.10, 0.20, 0.50, 0.60]
    model = LabelExtractionResult.model_validate(data)
    assert model.mrp.bbox == [0.10, 0.20, 0.50, 0.60]

    # Valid: None
    data["mrp"]["bbox"] = None
    model = LabelExtractionResult.model_validate(data)
    assert model.mrp.bbox is None

    # Invalid: 3 numbers instead of 4
    data["mrp"]["bbox"] = [10.0, 20.0, 100.0]
    with pytest.raises(ValidationError):
        LabelExtractionResult.model_validate(data)

    # Invalid: non-numeric coordinate
    data["mrp"]["bbox"] = ["ten", 20.0, 100.0, 50.0]
    with pytest.raises(ValidationError):
        LabelExtractionResult.model_validate(data)


def test_separate_manufacturer_entities():
    """Verify separate entities for manufacturer and packer."""
    data = sample_valid_extraction_dict()
    data["manufacturer"]["entities"] = [
        {
            "role": "manufacturer",
            "name": "Factory Alpha Ltd.",
            "address": "GIDC Industrial Estate, Gujarat",
            "raw_text": "Manufactured by: Factory Alpha Ltd.",
            "confidence": 0.95,
            "bbox": None
        },
        {
            "role": "packer",
            "name": "Packer Beta Inc.",
            "address": "Sector 4, Bhiwandi, Maharashtra",
            "raw_text": "Packed by: Packer Beta Inc.",
            "confidence": 0.92,
            "bbox": None
        }
    ]
    model = LabelExtractionResult.model_validate(data)
    assert len(model.manufacturer.entities) == 2
    assert model.manufacturer.entities[0].role == "manufacturer"
    assert model.manufacturer.entities[1].role == "packer"


# ==============================================================================
# SIH TECHNICAL SPECIFICATION TEST CASES 1 - 7
# ==============================================================================

def test_sih_test_1_rice_package():
    """TEST 1: 500g rice package - all 6 declarations visibly present."""
    data = sample_valid_extraction_dict()
    data["package"]["product_category"] = "rice"
    data["net_quantity"]["value"] = 500.0
    data["net_quantity"]["unit"] = "g"
    data["net_quantity"]["quantity_type"] = "weight"

    model = LabelExtractionResult.model_validate(data)
    assert model.generic_name.status == "present"
    assert model.mrp.status == "present"
    assert len(model.dates) >= 1
    assert model.net_quantity.value == 500.0
    assert model.manufacturer.status == "present"
    assert model.consumer_care.status == "present"


def test_sih_test_2_namkeen_numeral_region():
    """TEST 2: 150g namkeen - extract category, quantity 150g, MRP numeral region (no 0.8mm font size)."""
    data = sample_valid_extraction_dict()
    data["package"]["product_category"] = "namkeen"
    data["net_quantity"]["value"] = 150.0
    data["net_quantity"]["unit"] = "g"
    data["net_quantity"]["raw_text"] = "Net Wt.: 150g"
    data["mrp"]["numeral_region"] = [0.4, 0.6, 0.48, 0.65]

    model = LabelExtractionResult.model_validate(data)
    assert model.package.product_category == "namkeen"
    assert model.net_quantity.value == 150.0
    assert model.mrp.numeral_region == [0.4, 0.6, 0.48, 0.65]
    # Verify we do NOT output millimeter measurements in Module 1
    assert not hasattr(model.mrp, "font_size_mm")


def test_sih_test_3_cream_when_packed():
    """TEST 3: 250g cosmetic cream with 'when packed'."""
    data = sample_valid_extraction_dict()
    data["package"]["product_category"] = "cream"
    data["package"]["food_status"] = "non_food"
    data["net_quantity"]["raw_text"] = "250 g when packed"
    data["net_quantity"]["value"] = 250.0
    data["net_quantity"]["unit"] = "g"
    data["net_quantity"]["when_packed"] = True

    model = LabelExtractionResult.model_validate(data)
    assert model.package.product_category == "cream"
    assert model.net_quantity.value == 250.0
    assert model.net_quantity.when_packed is True


def test_sih_test_4_detergent_when_packed():
    """TEST 4: 1kg detergent with 'when packed' (extract fact, do not judge legality)."""
    data = sample_valid_extraction_dict()
    data["package"]["product_category"] = "detergent"
    data["package"]["food_status"] = "non_food"
    data["net_quantity"]["raw_text"] = "1 kg when packed"
    data["net_quantity"]["value"] = 1.0
    data["net_quantity"]["unit"] = "kg"
    data["net_quantity"]["when_packed"] = True

    model = LabelExtractionResult.model_validate(data)
    assert model.package.product_category == "detergent"
    assert model.net_quantity.value == 1.0
    assert model.net_quantity.when_packed is True


def test_sih_test_5_cooking_oil_food_status():
    """TEST 5: 2L cooking oil without manufacturer address - food_status=food, address=None."""
    data = sample_valid_extraction_dict()
    data["package"]["product_category"] = "cooking oil"
    data["package"]["food_status"] = "food"
    data["net_quantity"]["value"] = 2.0
    data["net_quantity"]["unit"] = "L"
    data["net_quantity"]["quantity_type"] = "volume"
    # Manufacturer has name but no address visible
    data["manufacturer"]["entities"] = [
        {
            "role": "manufacturer",
            "name": "Pure Oils Co.",
            "address": None,
            "raw_text": "Pure Oils Co.",
            "confidence": 0.88,
            "bbox": None
        }
    ]

    model = LabelExtractionResult.model_validate(data)
    assert model.package.food_status == "food"
    assert model.net_quantity.value == 2.0
    assert model.manufacturer.entities[0].address is None


def test_sih_test_6_packaged_water_declared_quantity():
    """TEST 6: 500ml packaged water - extract declared quantity, no legal size comparison."""
    data = sample_valid_extraction_dict()
    data["package"]["product_category"] = "packaged water"
    data["net_quantity"]["value"] = 500.0
    data["net_quantity"]["unit"] = "ml"
    data["net_quantity"]["raw_text"] = "Net Volume: 500 ml"

    model = LabelExtractionResult.model_validate(data)
    assert model.package.product_category == "packaged water"
    assert model.net_quantity.value == 500.0
    assert model.net_quantity.unit == "ml"


def test_sih_test_7_wholesale_carton_biscuits():
    """TEST 7: Wholesale carton of 24 biscuit packets - package_type=wholesale, count=24."""
    data = sample_valid_extraction_dict()
    data["package"]["package_type"] = "wholesale"
    data["package"]["product_category"] = "biscuits"
    data["package"]["retail_unit_count"] = 24
    data["package"]["evidence"] = [
        "Brown corrugated shipper carton",
        "Printed text: Contains 24 biscuit packets"
    ]

    model = LabelExtractionResult.model_validate(data)
    assert model.package.package_type == "wholesale"
    assert model.package.product_category == "biscuits"
    assert model.package.retail_unit_count == 24
    assert len(model.package.evidence) == 2


# ==============================================================================
# SIH RULE 7 / 8 NUMERAL REGION TEST CASES (A THROUGH I)
# ==============================================================================

def test_numeral_region_case_a_net_quantity_integer():
    """Case A: 'NET WT: 500 g' -> numeral_region exists and tightly targets '500'."""
    data = sample_valid_extraction_dict()
    data["net_quantity"]["raw_text"] = "NET WT: 500 g"
    data["net_quantity"]["value"] = 500.0
    data["net_quantity"]["unit"] = "g"
    data["net_quantity"]["numeral_region"] = [0.30, 0.40, 0.45, 0.45]

    model = LabelExtractionResult.model_validate(data)
    assert model.net_quantity.numeral_region == [0.30, 0.40, 0.45, 0.45]
    assert model.net_quantity.value == 500.0


def test_numeral_region_case_b_net_quantity_decimal():
    """Case B: 'NET WT: 1.5 L' -> numeral_region exists and tightly targets '1.5'."""
    data = sample_valid_extraction_dict()
    data["net_quantity"]["raw_text"] = "NET WT: 1.5 L"
    data["net_quantity"]["value"] = 1.5
    data["net_quantity"]["unit"] = "L"
    data["net_quantity"]["quantity_type"] = "volume"
    data["net_quantity"]["numeral_region"] = [0.28, 0.40, 0.38, 0.45]

    model = LabelExtractionResult.model_validate(data)
    assert model.net_quantity.numeral_region == [0.28, 0.40, 0.38, 0.45]
    assert model.net_quantity.value == 1.5


def test_numeral_region_case_c_mrp():
    """Case C: 'MRP ₹120.00 Inclusive of all taxes' -> numeral_region exists and tightly targets '120.00'."""
    data = sample_valid_extraction_dict()
    data["mrp"]["raw_text"] = "MRP ₹120.00 Inclusive of all taxes"
    data["mrp"]["value"] = 120.00
    data["mrp"]["currency"] = "₹"
    data["mrp"]["numeral_region"] = [0.25, 0.50, 0.40, 0.55]

    model = LabelExtractionResult.model_validate(data)
    assert model.mrp.numeral_region == [0.25, 0.50, 0.40, 0.55]
    assert model.mrp.value == 120.00


def test_numeral_region_case_d_numeric_packing_date():
    """Case D: 'PKD: 08/2026' -> numeral_region exists and targets '08/2026'."""
    data = sample_valid_extraction_dict()
    data["dates"] = [
        {
            "status": "present",
            "date_type": "packing",
            "raw_text": "PKD: 08/2026",
            "month": "08",
            "year": 2026,
            "confidence": 0.95,
            "bbox": [0.55, 0.50, 0.90, 0.55],
            "numeral_region": [0.65, 0.50, 0.85, 0.55]
        }
    ]

    model = LabelExtractionResult.model_validate(data)
    assert len(model.dates) == 1
    assert model.dates[0].numeral_region == [0.65, 0.50, 0.85, 0.55]
    assert model.dates[0].month == "08"
    assert model.dates[0].year == 2026


def test_numeral_region_case_e_text_month_date():
    """Case E: 'PKD: AUGUST 2026' -> semantic date extracted, numeral_region is None (no fake numeric region)."""
    data = sample_valid_extraction_dict()
    data["dates"] = [
        {
            "status": "present",
            "date_type": "packing",
            "raw_text": "PKD: AUGUST 2026",
            "month": "August",
            "year": 2026,
            "confidence": 0.95,
            "bbox": [0.55, 0.50, 0.90, 0.55],
            "numeral_region": None  # Text month must NOT produce a Rule 7 numeric region
        }
    ]

    model = LabelExtractionResult.model_validate(data)
    assert model.dates[0].month == "August"
    assert model.dates[0].year == 2026
    assert model.dates[0].numeral_region is None


def test_numeral_region_case_f_phone_number_exclusion():
    """Case F: 'Customer Care: 022-67276727' -> in additional_numeric_information, no numeral_region."""
    data = sample_valid_extraction_dict()
    data["additional_numeric_information"] = [
        {
            "semantic_role": "phone",
            "raw_text": "Customer Care: 022-67276727",
            "value": "022-67276727",
            "bbox": [0.1, 0.85, 0.6, 0.9],
            "confidence": 0.95
        }
    ]

    model = LabelExtractionResult.model_validate(data)
    info = model.additional_numeric_information[0]
    assert info.semantic_role == "phone"
    assert info.value == "022-67276727"
    assert not hasattr(info, "numeral_region")


def test_numeral_region_case_g_pin_code_exclusion():
    """Case G: 'Mumbai 400002' -> in additional_numeric_information, no numeral_region."""
    data = sample_valid_extraction_dict()
    data["additional_numeric_information"] = [
        {
            "semantic_role": "pin_code",
            "raw_text": "Mumbai 400002",
            "value": "400002",
            "bbox": [0.1, 0.75, 0.4, 0.8],
            "confidence": 0.98
        }
    ]

    model = LabelExtractionResult.model_validate(data)
    info = model.additional_numeric_information[0]
    assert info.semantic_role == "pin_code"
    assert info.value == "400002"
    assert not hasattr(info, "numeral_region")


def test_numeral_region_case_h_license_number_exclusion():
    """Case H: 'Lic. No. 11517018000800' -> in additional_numeric_information, no numeral_region."""
    data = sample_valid_extraction_dict()
    data["additional_numeric_information"] = [
        {
            "semantic_role": "other",
            "raw_text": "Lic. No. 11517018000800",
            "value": "11517018000800",
            "bbox": [0.1, 0.9, 0.7, 0.95],
            "confidence": 0.96
        }
    ]

    model = LabelExtractionResult.model_validate(data)
    info = model.additional_numeric_information[0]
    assert info.value == "11517018000800"
    assert not hasattr(info, "numeral_region")


def test_numeral_region_case_i_barcode_exclusion():
    """Case I: '8 902901 226959' -> in additional_numeric_information, no numeral_region."""
    data = sample_valid_extraction_dict()
    data["additional_numeric_information"] = [
        {
            "semantic_role": "other",
            "raw_text": "8 902901 226959",
            "value": "8902901226959",
            "bbox": [0.7, 0.85, 0.95, 0.95],
            "confidence": 0.97
        }
    ]

    model = LabelExtractionResult.model_validate(data)
    info = model.additional_numeric_information[0]
    assert info.value == "8902901226959"
    assert not hasattr(info, "numeral_region")


def test_validate_bbox_list_unit_cases():
    """Verify bbox validator requirements (4 numeric coordinates, 0..1 range, ordering x_min<=x_max & y_min<=y_max, nulls, nested lists)."""
    from app.schemas.extraction import validate_bbox_list

    # Valid cases
    assert validate_bbox_list(None) is None
    assert validate_bbox_list([]) is None
    assert validate_bbox_list([0.205, 0.283, 0.43, 0.303]) == [0.205, 0.283, 0.43, 0.303]
    assert validate_bbox_list((0.0, 0.0, 1.0, 1.0)) == [0.0, 0.0, 1.0, 1.0]
    assert validate_bbox_list([[0.205, 0.283, 0.43, 0.303]]) == [0.205, 0.283, 0.43, 0.303]
    assert validate_bbox_list("[0.205, 0.283, 0.43, 0.303]") == [0.205, 0.283, 0.43, 0.303]

    # Invalid length
    with pytest.raises(ValueError, match="Bounding box must contain exactly 4 coordinates"):
        validate_bbox_list([0.1, 0.2, 0.3])

    with pytest.raises(ValueError, match="Bounding box must contain exactly 4 coordinates"):
        validate_bbox_list([0.1, 0.2, 0.3, 0.4, 0.5])

    # Invalid non-numeric
    with pytest.raises(ValueError, match="Bounding box coordinates must be numeric"):
        validate_bbox_list([0.1, "abc", 0.3, 0.4])

    # Invalid range (< 0 or > 1)
    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        validate_bbox_list([-0.1, 0.2, 0.3, 0.4])

    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        validate_bbox_list([0.1, 0.2, 1.5, 0.4])

    # Invalid ordering (x_min > x_max or y_min > y_max)
    with pytest.raises(ValueError, match="x_min .* must be <= x_max"):
        validate_bbox_list([0.5, 0.2, 0.3, 0.4])

    with pytest.raises(ValueError, match="y_min .* must be <= y_max"):
        validate_bbox_list([0.1, 0.6, 0.3, 0.4])


def test_lays_extraction_json_pattern_regression():
    """Regression test using the Lay's extraction JSON pattern with all 13 bbox/numeral_region fields."""
    lays_payload = {
        "image": {
            "width": 1200,
            "height": 1600,
            "coordinate_system": "normalized_0_1"
        },
        "raw_text_blocks": [
            {
                "text": "Lay's Classic Salted Potato Chips",
                "bbox": [0.10, 0.05, 0.90, 0.20],
                "confidence": 0.99
            }
        ],
        "semantic_extraction": {
            "image_quality": {
                "overall": "good",
                "notes": "Clear Lay's pack shot"
            },
            "package": {
                "package_type": "retail",
                "brand_name": "Lay's",
                "generic_name": "Potato Chips",
                "product_category": "namkeen",
                "food_status": "food",
                "retail_unit_count": None,
                "evidence": ["Pouch front label"]
            },
            "manufacturer": {
                "status": "present",
                "entities": [
                    {
                        "role": "manufacturer",
                        "name": "PepsiCo India Holdings Pvt. Ltd.",
                        "address": "Gurugram, Haryana",
                        "raw_text": "Mfg by PepsiCo India Holdings Pvt. Ltd.",
                        "confidence": 0.97,
                        "bbox": [0.10, 0.70, 0.90, 0.80]
                    }
                ]
            },
            "generic_name": {
                "status": "present",
                "raw_text": "Potato Chips",
                "value": "Potato Chips",
                "confidence": 0.98,
                "bbox": [0.20, 0.20, 0.80, 0.25]
            },
            "mrp": {
                "status": "present",
                "raw_text": "MRP ₹20.00 (Incl. of all taxes)",
                "value": 20.0,
                "currency": "₹",
                "confidence": 0.99,
                "bbox": [0.10, 0.45, 0.45, 0.50],
                "numeral_region": [0.205, 0.455, 0.300, 0.495]
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
                    "numeral_region": [0.600, 0.455, 0.800, 0.495]
                },
                {
                    "status": "present",
                    "date_type": "manufacture",
                    "raw_text": "Mfg: 10/08/2026",
                    "month": "08",
                    "year": 2026,
                    "confidence": 0.95,
                    "bbox": [0.50, 0.52, 0.85, 0.57],
                    "numeral_region": [0.600, 0.525, 0.800, 0.565]
                }
            ],
            "net_quantity": {
                "status": "present",
                "raw_text": "Net Qty: 52 g",
                "value": 52.0,
                "unit": "g",
                "quantity_type": "weight",
                "when_packed": True,
                "confidence": 0.99,
                "bbox": [0.10, 0.55, 0.45, 0.60],
                "numeral_region": [0.220, 0.555, 0.310, 0.595]
            },
            "consumer_care": {
                "status": "present",
                "name": "Feedback Manager",
                "address": "PO Box 27, New Delhi",
                "phone": "1800-222-444",
                "email": "feedback@pepsico.com",
                "raw_text": "Contact Feedback Manager 1800-222-444",
                "confidence": 0.95,
                "bbox": [0.10, 0.82, 0.90, 0.90]
            },
            "stickers": [
                {
                    "present": True,
                    "raw_text": "₹5 EXTRA VALUE",
                    "possible_field_affected": "mrp",
                    "bbox": [0.02, 0.02, 0.25, 0.15],
                    "confidence": 0.90
                }
            ],
            "additional_numeric_information": [
                {
                    "semantic_role": "pin_code",
                    "raw_text": "122002",
                    "value": "122002",
                    "bbox": [0.70, 0.78, 0.88, 0.81],
                    "confidence": 0.97
                }
            ],
            "other_visible_text": [
                {
                    "text_type": "promotional",
                    "raw_text": "50% MORE CHIPS",
                    "bbox": [0.20, 0.28, 0.80, 0.35],
                    "confidence": 0.92
                }
            ]
        }
    }

    result = LabelExtractionResult.model_validate(lays_payload)

    # Validate Lay's extraction model assertions
    assert result.package.brand_name == "Lay's"
    assert result.mrp.value == 20.0
    assert result.mrp.numeral_region == [0.205, 0.455, 0.300, 0.495]
    assert len(result.dates) == 2
    assert result.dates[0].numeral_region == [0.600, 0.455, 0.800, 0.495]
    assert result.dates[1].numeral_region == [0.600, 0.525, 0.800, 0.565]
    assert result.net_quantity.numeral_region == [0.220, 0.555, 0.310, 0.595]
    assert result.stickers[0].bbox == [0.02, 0.02, 0.25, 0.15]
    assert result.additional_numeric_information[0].bbox == [0.70, 0.78, 0.88, 0.81]
    assert result.other_visible_text[0].bbox == [0.20, 0.28, 0.80, 0.35]


def test_combined_manufacture_and_use_by_dates_in_single_entry():
    """Verify that a combined declaration string in a single date entry is split into 2 items."""
    data = sample_valid_extraction_dict()
    data["dates"] = [
        {
            "status": "present",
            "date_type": "manufacture",
            "raw_text": "MFD & USE BY: 26/07/26 & 08/12/26",
            "month": "07",
            "year": 2026,
            "bbox": [0.50, 0.45, 0.88, 0.50],
            "numeral_region": [0.50, 0.45, 0.88, 0.50]
        }
    ]

    model = LabelExtractionResult.model_validate(data)
    assert len(model.dates) == 2

    # Date 0: manufacture -> 26/07/26
    d0 = model.dates[0]
    assert d0.date_type == "manufacture"
    assert d0.raw_text == "26/07/26"
    assert d0.month == "07"
    assert d0.year == 2026
    assert d0.numeral_region is not None

    # Date 1: use_by -> 08/12/26
    d1 = model.dates[1]
    assert d1.date_type == "use_by"
    assert d1.raw_text == "08/12/26"
    assert d1.month == "12"
    assert d1.year == 2026
    assert d1.numeral_region is not None

    # Ensure separate tight regions
    assert d0.numeral_region[2] <= d1.numeral_region[0]


def test_combined_manufacture_and_use_by_recovery_from_raw_text_blocks():
    """Regression test for Lay's case where OCR has combined block but extraction returned only dates[0]."""
    payload = {
        "image": {"width": 1000, "height": 1000, "coordinate_system": "normalized_0_1"},
        "raw_text_blocks": [
            {
                "text": "MFD & USE BY: 26/07/26 & 08/12/26",
                "bbox": [0.50, 0.45, 0.88, 0.50],
                "confidence": 0.98
            }
        ],
        "semantic_extraction": {
            "image_quality": {"overall": "good", "notes": "Clear packaging"},
            "package": {
                "package_type": "retail",
                "brand_name": "Lay's",
                "generic_name": "Potato Chips",
                "product_category": "namkeen",
                "food_status": "food",
                "retail_unit_count": None,
                "evidence": ["Pouch front label"]
            },
            "manufacturer": {"status": "missing", "entities": []},
            "generic_name": {"status": "present", "raw_text": "Potato Chips", "value": "Potato Chips", "bbox": None},
            "mrp": {"status": "missing"},
            "dates": [
                {
                    "status": "present",
                    "date_type": "manufacture",
                    "raw_text": "26/07/26",
                    "month": "07",
                    "year": 2026,
                    "confidence": 0.95,
                    "bbox": [0.50, 0.45, 0.70, 0.50],
                    "numeral_region": [0.55, 0.455, 0.68, 0.495]
                }
            ],
            "net_quantity": {"status": "missing"},
            "consumer_care": {"status": "missing"},
            "stickers": [],
            "additional_numeric_information": [],
            "other_visible_text": []
        }
    }

    result = LabelExtractionResult.model_validate(payload)
    assert len(result.dates) == 2

    # dates[0] = manufacture -> 26/07/26
    assert result.dates[0].date_type == "manufacture"
    assert "26/07/26" in result.dates[0].raw_text
    assert result.dates[0].month == "07"
    assert result.dates[0].year == 2026
    assert result.dates[0].numeral_region is not None

    # dates[1] = use_by -> 08/12/26
    assert result.dates[1].date_type == "use_by"
    assert "08/12/26" in result.dates[1].raw_text
    assert result.dates[1].month == "12"
    assert result.dates[1].year == 2026
    assert result.dates[1].numeral_region is not None


def test_no_duplicate_dates_from_other_visible_text():
    """Verify that duplicate dates are not created if already present in dates[]."""
    payload = {
        "image": {"width": 1000, "height": 1000, "coordinate_system": "normalized_0_1"},
        "raw_text_blocks": [],
        "semantic_extraction": {
            "image_quality": {"overall": "good", "notes": "Clear packaging"},
            "package": {
                "package_type": "retail",
                "brand_name": "Lay's",
                "generic_name": "Potato Chips",
                "product_category": "namkeen",
                "food_status": "food",
                "retail_unit_count": None,
                "evidence": ["Pouch front label"]
            },
            "manufacturer": {"status": "missing", "entities": []},
            "generic_name": {"status": "present", "raw_text": "Potato Chips", "value": "Potato Chips", "bbox": None},
            "mrp": {"status": "missing"},
            "dates": [
                {
                    "status": "present",
                    "date_type": "manufacture",
                    "raw_text": "26/07/26",
                    "month": "07",
                    "year": 2026,
                    "confidence": 0.95,
                    "bbox": [0.50, 0.45, 0.70, 0.50],
                    "numeral_region": [0.55, 0.455, 0.68, 0.495]
                },
                {
                    "status": "present",
                    "date_type": "use_by",
                    "raw_text": "08/12/26",
                    "month": "12",
                    "year": 2026,
                    "confidence": 0.95,
                    "bbox": [0.72, 0.45, 0.88, 0.50],
                    "numeral_region": [0.75, 0.455, 0.88, 0.495]
                }
            ],
            "net_quantity": {"status": "missing"},
            "consumer_care": {"status": "missing"},
            "stickers": [],
            "additional_numeric_information": [],
            "other_visible_text": [
                {
                    "text_type": "best_before_information",
                    "raw_text": "MFD & USE BY: 26/07/26 & 08/12/26",
                    "bbox": [0.50, 0.45, 0.88, 0.50],
                    "confidence": 0.90
                }
            ]
        }
    }

    result = LabelExtractionResult.model_validate(payload)
    assert len(result.dates) == 2
    assert result.dates[0].date_type == "manufacture"
    assert result.dates[1].date_type == "use_by"


def test_single_date_preservation():
    """Verify single date declaration remains completely untouched."""
    data = sample_valid_extraction_dict()
    data["dates"] = [
        {
            "status": "present",
            "date_type": "packing",
            "raw_text": "Pkd: 15/08/2026",
            "month": "08",
            "year": 2026,
            "confidence": 0.96,
            "bbox": [0.50, 0.45, 0.85, 0.50],
            "numeral_region": [0.600, 0.455, 0.800, 0.495]
        }
    ]

    result = LabelExtractionResult.model_validate(data)
    assert len(result.dates) == 1
    assert result.dates[0].date_type == "packing"
    assert result.dates[0].raw_text == "Pkd: 15/08/2026"
    assert result.dates[0].month == "08"
    assert result.dates[0].year == 2026


def test_text_month_date_preservation():
    """Verify text-month date declaration remains completely untouched without numeric region."""
    data = sample_valid_extraction_dict()
    data["dates"] = [
        {
            "status": "present",
            "date_type": "manufacture",
            "raw_text": "Mfg: August 2026",
            "month": "August",
            "year": 2026,
            "confidence": 0.95,
            "bbox": [0.50, 0.45, 0.85, 0.50],
            "numeral_region": None
        }
    ]

    result = LabelExtractionResult.model_validate(data)
    assert len(result.dates) == 1
    assert result.dates[0].date_type == "manufacture"
    assert result.dates[0].raw_text == "Mfg: August 2026"
    assert result.dates[0].month == "August"
    assert result.dates[0].year == 2026
    assert result.dates[0].numeral_region is None


def test_date_type_normalization():
    """Verify date_type field validator normalizes common aliases."""
    d1 = DateDeclaration(raw_text="08/12/26", date_type="use-by")
    assert d1.date_type == "use_by"

    d2 = DateDeclaration(raw_text="08/12/26", date_type="expiry")
    assert d2.date_type == "use_by"

    d3 = DateDeclaration(raw_text="08/12/26", date_type="best_before")
    assert d3.date_type == "use_by"

    d4 = DateDeclaration(raw_text="26/07/26", date_type="mfg")
    assert d4.date_type == "manufacture"

    d5 = DateDeclaration(raw_text="26/07/26", date_type="pkd")
    assert d5.date_type == "packing"



