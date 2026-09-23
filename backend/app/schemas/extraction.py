"""Pydantic v2 schemas for Module 1: Vision-Based Semantic Label Extraction (V1.1).

Defines the exact JSON contract representing information visibly present on a
packaged commodity per SIH 2026 PS 26034.
Includes explicit image metadata, a raw visual text layer (raw_text_blocks),
and structured semantic declarations (semantic_extraction).
"""

import re
from typing import Annotated, Any, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Base Extraction Status Literals
PresenceStatus = Literal["present", "missing", "uncertain"]
DatePresenceStatus = Literal["present", "uncertain"]
PackageType = Literal["retail", "wholesale", "combination_pack", "unknown"]
FoodStatus = Literal["food", "non_food", "uncertain"]
ManufacturerRole = Literal["manufacturer", "packer", "importer", "marketer", "unknown"]
DateType = Literal["manufacture", "packing", "import", "use_by", "unknown"]
QuantityType = Literal["weight", "volume", "number", "length", "area", "unknown"]
StickerField = Literal["mrp", "quantity", "date", "unknown"]
NumericRole = Literal["phone", "pin_code", "survey_number", "other"]
VisibleTextType = Literal[
    "marketing",
    "promotional",
    "warning",
    "ingredient",
    "best_before_information",
    "other",
]
ImageQualityLevel = Literal["good", "moderate", "poor"]
CoordinateSystemType = Literal["normalized_0_1"]


def validate_bbox_list(v: Any) -> Optional[list[float]]:
    """Validates that a bounding box is None or a list of exactly 4 numeric values (0.0 to 1.0)."""
    if v is None:
        return None

    if isinstance(v, str):
        v_str = v.strip()
        if not v_str or v_str.lower() in ("null", "none"):
            return None
        import json
        try:
            parsed = json.loads(v_str)
            if isinstance(parsed, (list, tuple)):
                v = parsed
            else:
                parts = [p.strip() for p in v_str.split(",") if p.strip()]
                v = parts
        except Exception:
            parts = [p.strip() for p in v_str.strip("[]()").split(",") if p.strip()]
            v = parts

    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return None
        # Support single-nested list e.g. [[x_min, y_min, x_max, y_max]]
        if len(v) == 1 and isinstance(v[0], (list, tuple)):
            v = v[0]
        if len(v) != 4:
            raise ValueError("Bounding box must contain exactly 4 coordinates: [x_min, y_min, x_max, y_max]")
        try:
            coords = [float(x) for x in v]
        except (ValueError, TypeError) as exc:
            raise ValueError("Bounding box coordinates must be numeric") from exc

        x_min, y_min, x_max, y_max = coords

        for name, val in [("x_min", x_min), ("y_min", y_min), ("x_max", x_max), ("y_max", y_max)]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"Bounding box coordinate {name} must be between 0.0 and 1.0, got {val}")

        if x_min > x_max:
            raise ValueError(f"x_min ({x_min}) must be <= x_max ({x_max})")
        if y_min > y_max:
            raise ValueError(f"y_min ({y_min}) must be <= y_max ({y_max})")

        return coords

    raise ValueError("Bounding box must be a list of 4 numbers or null")


# ==============================================================================
# IMAGE METADATA & RAW TEXT BLOCKS
# ==============================================================================

class ImageMetadata(BaseModel):
    """Metadata regarding the processed packaging photo and bounding box coordinate system."""
    model_config = ConfigDict(extra="forbid")

    width: int = Field(
        default=0,
        ge=0,
        description="Image width in pixels.",
    )
    height: int = Field(
        default=0,
        ge=0,
        description="Image height in pixels.",
    )
    coordinate_system: CoordinateSystemType = Field(
        default="normalized_0_1",
        description="Explicit bounding box coordinate system (normalized from 0.0 to 1.0).",
    )


class RawTextBlock(BaseModel):
    """A raw visual text block detected on the packaging before semantic classification."""
    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        ...,
        description="Raw text string visibly detected on the package.",
    )
    bbox: Optional[list[float]] = Field(
        default=None,
        description="Normalized bounding box [x_min, y_min, x_max, y_max] (0.0 to 1.0) or null.",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Visual detection confidence score between 0.0 and 1.0.",
    )

    @field_validator("bbox", mode="before")
    @classmethod
    def check_bbox(cls, v: Any) -> Optional[list[float]]:
        return validate_bbox_list(v)


# ==============================================================================
# SEMANTIC DECLARATION SCHEMAS
# ==============================================================================

class ImageQuality(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall: ImageQualityLevel = Field(
        ...,
        description="Overall visual readability taking into account blur, glare, lighting, and occlusion."
    )
    notes: str = Field(
        default="",
        description="Brief descriptive assessment of the visual quality and any factors impairing label reading."
    )


class PackageContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    package_type: PackageType = Field(
        default="unknown",
        description="Identified package format: retail, wholesale, combination_pack, or unknown."
    )
    brand_name: Optional[str] = Field(
        default=None,
        description="Brand or trademark name (e.g. 'Tata Sampann', 'Parle-G'). Distinguishable from generic name."
    )
    generic_name: Optional[str] = Field(
        default=None,
        description="Common or descriptive commodity name visible on package context (e.g. 'Basmati Rice')."
    )
    product_category: Optional[str] = Field(
        default=None,
        description="Commodity category (e.g. 'rice', 'namkeen', 'cooking oil', 'cream', 'detergent', 'packaged water')."
    )
    food_status: FoodStatus = Field(
        default="uncertain",
        description="Classification whether product is food, non_food, or uncertain based only on package evidence."
    )
    retail_unit_count: Optional[int] = Field(
        default=None,
        description="Count of individual retail packs inside if package is wholesale or multi-pack (e.g. 24)."
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Visual clues supporting package type, category, or unit count classifications."
    )


class ManufacturerEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: ManufacturerRole = Field(
        default="unknown",
        description="Role declared on label: manufacturer, packer, importer, marketer, or unknown."
    )
    name: str = Field(
        ...,
        description="Declared entity name."
    )
    address: Optional[str] = Field(
        default=None,
        description="Declared registered office or factory address."
    )
    raw_text: Optional[str] = Field(
        default=None,
        description="Exact visible text declaring this entity."
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this entity extraction (0.0 to 1.0)."
    )
    bbox: Optional[list[float]] = Field(
        default=None,
        description="Bounding box [x_min, y_min, x_max, y_max] or null."
    )

    @field_validator("bbox", mode="before")
    @classmethod
    def check_bbox(cls, v: Any) -> Optional[list[float]]:
        return validate_bbox_list(v)


class ManufacturerSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: PresenceStatus = Field(
        ...,
        description="Status of manufacturer/packer/importer details: present, missing, or uncertain."
    )
    entities: list[ManufacturerEntity] = Field(
        default_factory=list,
        description="List of all visible manufacturing, packing, importing, or marketing entities."
    )


class GenericNameDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: PresenceStatus = Field(
        ...,
        description="Status of generic/common commodity name declaration: present, missing, or uncertain."
    )
    raw_text: Optional[str] = Field(
        default=None,
        description="Smallest contiguous visible text span declaring generic/common name."
    )
    value: Optional[str] = Field(
        default=None,
        description="Extracted descriptive name of the commodity (e.g. 'Basmati Rice', 'Toilet Soap')."
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0."
    )
    bbox: Optional[list[float]] = Field(
        default=None,
        description="Bounding box [x_min, y_min, x_max, y_max] or null."
    )

    @field_validator("bbox", mode="before")
    @classmethod
    def check_bbox(cls, v: Any) -> Optional[list[float]]:
        return validate_bbox_list(v)


class MRPDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: PresenceStatus = Field(
        ...,
        description="Status of Retail Sale Price / MRP: present, missing, or uncertain."
    )
    raw_text: Optional[str] = Field(
        default=None,
        description="Smallest contiguous visible text declaration (e.g. 'MRP ₹120.00 Inclusive of all taxes')."
    )
    value: Optional[float] = Field(
        default=None,
        description="Normalized numeric monetary amount (e.g. 120.00)."
    )
    currency: Optional[str] = Field(
        default=None,
        description="Visible currency designation (e.g. '₹', 'Rs.', 'INR')."
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0."
    )
    bbox: Optional[list[float]] = Field(
        default=None,
        description="Bounding box of complete MRP declaration [x_min, y_min, x_max, y_max] or null."
    )
    numeral_region: Optional[list[float]] = Field(
        default=None,
        description="Bounding box of the isolated numeric digits for future CV measurement, or null."
    )

    @field_validator("bbox", "numeral_region", mode="before")
    @classmethod
    def check_bbox_coords(cls, v: Any) -> Optional[list[float]]:
        return validate_bbox_list(v)


DATE_REGEX = re.compile(
    r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})\b|\b(\d{1,2})[/.-](\d{2,4})\b"
)


def parse_numeric_date_from_match(m: re.Match) -> tuple[Optional[str], Optional[int]]:
    """Extracts (month, year) from a regex date match."""
    g = m.groups()
    if g[0] is not None:
        p1, p2, p3 = int(g[0]), int(g[1]), int(g[2])
        year = 2000 + p3 if p3 < 100 else p3
        if p1 > 12:
            month = f"{p2:02d}"
        elif p2 > 12:
            month = f"{p1:02d}"
        else:
            month = f"{p2:02d}"
        return month, year
    elif g[3] is not None:
        p1, p2 = int(g[3]), int(g[4])
        month = f"{p1:02d}"
        year = 2000 + p2 if p2 < 100 else p2
        return month, year
    return None, None


def determine_date_types_for_matches(text: str, matches: list[re.Match]) -> list[str]:
    """Determines semantic DateType ('manufacture', 'packing', 'use_by', 'unknown') for matched dates."""
    text_lower = text.lower()
    first_match_start = matches[0].start() if matches else 0
    header = text_lower[:first_match_start]

    header_has_mfd = any(w in header for w in ["mfd", "mfg", "mfr", "manufactur"])
    header_has_pkd = any(w in header for w in ["pkd", "packed", "packing"])
    header_has_exp = any(w in header for w in ["use by", "use_by", "use-by", "exp", "expiry", "best before", "bb"])

    if (header_has_mfd or header_has_pkd) and header_has_exp and len(matches) == 2:
        first_type = "packing" if header_has_pkd and not header_has_mfd else "manufacture"
        return [first_type, "use_by"]

    types = []
    for i, m in enumerate(matches):
        prev_end = matches[i - 1].end() if i > 0 else 0
        segment = text_lower[prev_end:m.start()]

        if any(w in segment for w in ["use by", "use_by", "use-by", "exp", "expiry", "best before", "bb"]):
            types.append("use_by")
        elif any(w in segment for w in ["pkd", "packed", "packing"]):
            types.append("packing")
        elif any(w in segment for w in ["mfd", "mfg", "mfr", "manufactur"]):
            types.append("manufacture")
        else:
            if i == 0 and (header_has_mfd or header_has_pkd):
                types.append("packing" if header_has_pkd else "manufacture")
            elif i == 1 and header_has_exp:
                types.append("use_by")
            elif any(w in text_lower for w in ["mfd", "mfg", "mfr", "manufactur"]):
                types.append("manufacture" if i == 0 else "use_by")
            else:
                types.append("unknown")
    return types


def is_date_already_extracted(
    num_str: str,
    month: Optional[str],
    year: Optional[int],
    date_type: str,
    existing_dates: list[Any],
) -> bool:
    """Checks if a date candidate is already present in existing dates list to avoid duplicates."""
    for ed in existing_dates:
        ed_raw = ed.get("raw_text", "") if isinstance(ed, dict) else getattr(ed, "raw_text", "")
        ed_m = ed.get("month") if isinstance(ed, dict) else getattr(ed, "month", None)
        ed_y = ed.get("year") if isinstance(ed, dict) else getattr(ed, "year", None)
        ed_t = ed.get("date_type") if isinstance(ed, dict) else getattr(ed, "date_type", None)

        if num_str and ed_raw and (num_str in str(ed_raw) or str(ed_raw) in num_str):
            return True
        if month and year and ed_m and ed_y:
            try:
                if int(month) == int(ed_m) and int(year) == int(ed_y):
                    if ed_t == date_type or ed_t == "unknown" or date_type == "unknown":
                        return True
            except (ValueError, TypeError):
                if str(month).lower() == str(ed_m).lower() and int(year) == int(ed_y):
                    return True
    return False


def split_combined_date_item(item: Any) -> list[dict[str, Any]]:
    """Splits a DateDeclaration or dict containing multiple numeric dates into separate date dictionaries."""
    if not isinstance(item, dict) and not isinstance(item, DateDeclaration):
        return [item]

    raw_text = item.get("raw_text", "") if isinstance(item, dict) else getattr(item, "raw_text", "")
    matches = list(DATE_REGEX.finditer(raw_text))
    if len(matches) <= 1:
        return [item if isinstance(item, dict) else item.model_dump()]

    types = determine_date_types_for_matches(raw_text, matches)
    bbox = item.get("bbox") if isinstance(item, dict) else getattr(item, "bbox", None)
    conf = item.get("confidence", 0.95) if isinstance(item, dict) else getattr(item, "confidence", 0.95)
    status = item.get("status", "present") if isinstance(item, dict) else getattr(item, "status", "present")
    L = len(raw_text)

    results = []
    for i, m in enumerate(matches):
        m_str = m.group(0)
        month, year = parse_numeric_date_from_match(m)
        dtype = types[i]

        num_region = None
        item_bbox = None
        if bbox and len(bbox) == 4 and L > 0:
            x0, y0, x1, y1 = bbox
            span_x = x1 - x0
            mx0 = round(x0 + (m.start() / L) * span_x, 4)
            mx1 = round(x0 + (m.end() / L) * span_x, 4)
            num_region = [mx0, y0, mx1, y1]
            item_bbox = [mx0, y0, mx1, y1]

        results.append({
            "status": status,
            "date_type": dtype,
            "raw_text": m_str,
            "month": month,
            "year": year,
            "confidence": conf,
            "bbox": item_bbox,
            "numeral_region": num_region,
        })
    return results


class DateDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: DatePresenceStatus = Field(
        default="present",
        description="Status of date declaration: present or uncertain."
    )
    date_type: DateType = Field(
        default="unknown",
        description="Type of date: manufacture, packing, import, use_by, or unknown."
    )
    raw_text: str = Field(
        ...,
        description="Smallest contiguous visible date string (e.g. 'Pkd: 08/2026', 'Mfg: August 2026', 'Use By: 08/12/2026')."
    )
    month: Optional[str] = Field(
        default=None,
        description="Extracted month representation (e.g. '08' or 'August')."
    )
    year: Optional[int] = Field(
        default=None,
        description="Extracted 4-digit calendar year (e.g. 2026)."
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0."
    )
    bbox: Optional[list[float]] = Field(
        default=None,
        description="Bounding box of complete date declaration [x_min, y_min, x_max, y_max] or null."
    )
    numeral_region: Optional[list[float]] = Field(
        default=None,
        description="Bounding box of numeric month/year digits for future CV measurement, or null."
    )

    @field_validator("date_type", mode="before")
    @classmethod
    def normalize_date_type(cls, v: Any) -> str:
        if not v:
            return "unknown"
        val = str(v).lower().strip().replace("-", "_").replace(" ", "_")
        if val in ("manufacture", "mfd", "mfg", "mfr", "date_of_mfg", "date_of_manufacture"):
            return "manufacture"
        if val in ("packing", "pkd", "pkg", "packed", "date_of_pkd", "date_of_packing"):
            return "packing"
        if val in ("import", "imp", "imported"):
            return "import"
        if val in ("use_by", "useby", "exp", "expiry", "best_before", "bb", "date_of_expiry"):
            return "use_by"
        if val in ("unknown",):
            return "unknown"
        return val

    @field_validator("month", mode="before")
    @classmethod
    def normalize_month(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        return str(v).strip()

    @field_validator("bbox", "numeral_region", mode="before")
    @classmethod
    def check_bbox_coords(cls, v: Any) -> Optional[list[float]]:
        return validate_bbox_list(v)


class NetQuantityDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: PresenceStatus = Field(
        ...,
        description="Status of net quantity declaration: present, missing, or uncertain."
    )
    raw_text: Optional[str] = Field(
        default=None,
        description="Smallest contiguous visible wording (e.g. 'Net Qty: 500 g', '250 g when packed')."
    )
    value: Optional[float] = Field(
        default=None,
        description="Extracted numeric quantity amount (e.g. 500.0, 1.0, 12)."
    )
    unit: Optional[str] = Field(
        default=None,
        description="Visible unit representation (e.g. 'g', 'kg', 'ml', 'L', 'N', 'U')."
    )
    quantity_type: QuantityType = Field(
        default="unknown",
        description="Type of measurement: weight, volume, number, length, area, or unknown."
    )
    when_packed: bool = Field(
        default=False,
        description="True if qualified by 'when packed' or equivalent visible wording."
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0."
    )
    bbox: Optional[list[float]] = Field(
        default=None,
        description="Bounding box of entire net quantity declaration [x_min, y_min, x_max, y_max] or null."
    )
    numeral_region: Optional[list[float]] = Field(
        default=None,
        description="Bounding box of quantity numerals for future CV measurement, or null."
    )

    @field_validator("bbox", "numeral_region", mode="before")
    @classmethod
    def check_bbox_coords(cls, v: Any) -> Optional[list[float]]:
        return validate_bbox_list(v)


class ConsumerCareField(BaseModel):
    """Sub-field detail within the Consumer Care declaration."""
    model_config = ConfigDict(extra="forbid")

    status: PresenceStatus = Field(
        default="missing",
        description="Detection status for this sub-field: present, missing, or uncertain."
    )
    value: Optional[str] = Field(
        default=None,
        description="Extracted value for this consumer care sub-field or null."
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0."
    )


class ConsumerCareDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: PresenceStatus = Field(
        ...,
        description="Overall status of consumer care details: present, missing, or uncertain."
    )
    name: ConsumerCareField = Field(
        default_factory=lambda: ConsumerCareField(status="missing", value=None, confidence=0.0),
        description="Contact person or officer/company name details."
    )
    address: ConsumerCareField = Field(
        default_factory=lambda: ConsumerCareField(status="missing", value=None, confidence=0.0),
        description="Contact grievance address details."
    )
    phone: ConsumerCareField = Field(
        default_factory=lambda: ConsumerCareField(status="missing", value=None, confidence=0.0),
        description="Contact telephone or helpline details."
    )
    email: ConsumerCareField = Field(
        default_factory=lambda: ConsumerCareField(status="missing", value=None, confidence=0.0),
        description="Contact email address details."
    )
    raw_text: Optional[str] = Field(
        default=None,
        description="Smallest contiguous visible text span declaring consumer care or null."
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence score between 0.0 and 1.0."
    )
    bbox: Optional[list[float]] = Field(
        default=None,
        description="Bounding box of consumer care block [x_min, y_min, x_max, y_max] or null."
    )

    @field_validator("name", "address", "phone", "email", mode="before")
    @classmethod
    def normalize_consumer_care_field(cls, v: Any) -> Any:
        if v is None:
            return ConsumerCareField(status="missing", value=None, confidence=0.0)
        if isinstance(v, str):
            v_clean = v.strip()
            if not v_clean:
                return ConsumerCareField(status="missing", value=None, confidence=0.0)
            return ConsumerCareField(status="present", value=v_clean, confidence=1.0)
        return v

    @field_validator("bbox", mode="before")
    @classmethod
    def check_bbox(cls, v: Any) -> Optional[list[float]]:
        return validate_bbox_list(v)


class StickerOverlay(BaseModel):
    model_config = ConfigDict(extra="forbid")

    present: bool = Field(
        default=True,
        description="Indicates an overlay sticker was visually detected."
    )
    raw_text: Optional[str] = Field(
        default=None,
        description="Text appearing on the overlay sticker."
    )
    possible_field_affected: StickerField = Field(
        default="unknown",
        description="Declaration category potentially covered or altered by the sticker."
    )
    bbox: Optional[list[float]] = Field(
        default=None,
        description="Bounding box of sticker [x_min, y_min, x_max, y_max] or null."
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0."
    )

    @field_validator("bbox", mode="before")
    @classmethod
    def check_bbox(cls, v: Any) -> Optional[list[float]]:
        return validate_bbox_list(v)


class AdditionalNumericInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semantic_role: NumericRole = Field(
        ...,
        description="Category of number: phone, pin_code, survey_number, or other."
    )
    raw_text: str = Field(
        ...,
        description="Visible text string containing the number."
    )
    value: Optional[str] = Field(
        default=None,
        description="Extracted digits or formatted number string."
    )
    bbox: Optional[list[float]] = Field(
        default=None,
        description="Bounding box [x_min, y_min, x_max, y_max] or null."
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0."
    )

    @field_validator("bbox", mode="before")
    @classmethod
    def check_bbox(cls, v: Any) -> Optional[list[float]]:
        return validate_bbox_list(v)


class OtherVisibleText(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text_type: VisibleTextType = Field(
        ...,
        description="Nature of text: marketing, promotional, warning, ingredient, best_before_information, or other."
    )
    raw_text: str = Field(
        ...,
        description="Visible non-declaration text (e.g. '20% EXTRA FREE', 'BEST BEFORE 12 MONTHS FROM PACKAGING')."
    )
    bbox: Optional[list[float]] = Field(
        default=None,
        description="Bounding box [x_min, y_min, x_max, y_max] or null."
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0."
    )

    @field_validator("bbox", mode="before")
    @classmethod
    def check_bbox(cls, v: Any) -> Optional[list[float]]:
        return validate_bbox_list(v)


# ==============================================================================
# SEMANTIC EXTRACTION & ROOT RESULT
# ==============================================================================

class SemanticExtraction(BaseModel):
    """Encapsulates all semantic declarations extracted from package visual evidence."""
    model_config = ConfigDict(extra="forbid")

    image_quality: ImageQuality
    package: PackageContext
    manufacturer: ManufacturerSection
    generic_name: GenericNameDeclaration
    mrp: MRPDeclaration
    dates: list[DateDeclaration] = Field(default_factory=list)
    net_quantity: NetQuantityDeclaration
    consumer_care: ConsumerCareDeclaration
    stickers: list[StickerOverlay] = Field(default_factory=list)
    additional_numeric_information: list[AdditionalNumericInfo] = Field(default_factory=list)
    other_visible_text: list[OtherVisibleText] = Field(default_factory=list)

    @field_validator("dates", mode="before")
    @classmethod
    def expand_combined_dates(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return v
        expanded = []
        for item in v:
            split_items = split_combined_date_item(item)
            expanded.extend(split_items)
        return expanded


class LabelExtractionResult(BaseModel):
    """V1.1 Root model containing image metadata, raw text blocks, and semantic extraction."""
    model_config = ConfigDict(extra="forbid")

    image: ImageMetadata
    raw_text_blocks: list[RawTextBlock] = Field(default_factory=list)
    semantic_extraction: SemanticExtraction

    @model_validator(mode="before")
    @classmethod
    def handle_v1_and_v1_1_payload(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # If payload is in V1 flat format (without "semantic_extraction" and has semantic fields)
            if "semantic_extraction" not in data and ("package" in data or "net_quantity" in data):
                semantic_field_names = {
                    "image_quality",
                    "package",
                    "manufacturer",
                    "generic_name",
                    "mrp",
                    "dates",
                    "net_quantity",
                    "consumer_care",
                    "stickers",
                    "additional_numeric_information",
                    "other_visible_text",
                }
                semantic_dict = {k: v for k, v in data.items() if k in semantic_field_names}
                image_dict = data.get("image", {"width": 0, "height": 0, "coordinate_system": "normalized_0_1"})
                raw_blocks = data.get("raw_text_blocks", [])
                data = {
                    "image": image_dict,
                    "raw_text_blocks": raw_blocks,
                    "semantic_extraction": semantic_dict,
                }

            # Inspect raw_text_blocks and other_visible_text to recover any missing dates from combined declarations
            sem = data.get("semantic_extraction")
            if isinstance(sem, dict):
                dates_list = sem.get("dates", [])
                if not isinstance(dates_list, list):
                    dates_list = []

                blocks_to_check = []
                for b in data.get("raw_text_blocks", []):
                    if isinstance(b, dict) and b.get("text"):
                        blocks_to_check.append({
                            "text": b.get("text"),
                            "bbox": b.get("bbox"),
                            "confidence": b.get("confidence", 0.95),
                        })
                    elif hasattr(b, "text") and getattr(b, "text", None):
                        blocks_to_check.append({
                            "text": getattr(b, "text"),
                            "bbox": getattr(b, "bbox", None),
                            "confidence": getattr(b, "confidence", 0.95),
                        })

                for ovt in sem.get("other_visible_text", []):
                    if isinstance(ovt, dict) and ovt.get("raw_text"):
                        blocks_to_check.append({
                            "text": ovt.get("raw_text"),
                            "bbox": ovt.get("bbox"),
                            "confidence": ovt.get("confidence", 0.95),
                        })
                    elif hasattr(ovt, "raw_text") and getattr(ovt, "raw_text", None):
                        blocks_to_check.append({
                            "text": getattr(ovt, "raw_text"),
                            "bbox": getattr(ovt, "bbox", None),
                            "confidence": getattr(ovt, "confidence", 0.95),
                        })

                new_dates_list = list(dates_list)
                for b in blocks_to_check:
                    text = b.get("text") or ""
                    matches = list(DATE_REGEX.finditer(text))
                    if len(matches) <= 1:
                        continue
                    types = determine_date_types_for_matches(text, matches)
                    bbox = b.get("bbox")
                    conf = b.get("confidence", 0.95)
                    L = len(text)

                    # If date 0 has a wide numeral_region covering both dates, tighten date 0
                    if len(new_dates_list) > 0 and bbox and len(bbox) == 4 and L > 0:
                        x0, y0, x1, y1 = bbox
                        span_x = x1 - x0
                        m0 = matches[0]
                        m1 = matches[1]
                        m1_start_x = round(x0 + (m1.start() / L) * span_x, 4)
                        for ed in new_dates_list:
                            if isinstance(ed, dict) and m0.group(0) in str(ed.get("raw_text", "")):
                                ed_nr = ed.get("numeral_region")
                                if ed_nr and len(ed_nr) == 4 and ed_nr[2] > m1_start_x:
                                    ed["numeral_region"] = [
                                        round(x0 + (m0.start() / L) * span_x, 4),
                                        y0,
                                        round(x0 + (m0.end() / L) * span_x, 4),
                                        y1,
                                    ]

                    for i, m in enumerate(matches):
                        m_str = m.group(0)
                        month, year = parse_numeric_date_from_match(m)
                        dtype = types[i]

                        # Check if already extracted
                        if is_date_already_extracted(m_str, month, year, dtype, new_dates_list):
                            # Update existing date_type if it was unknown
                            for ed in new_dates_list:
                                if isinstance(ed, dict):
                                    if m_str in str(ed.get("raw_text", "")) and ed.get("date_type") in ("unknown", None):
                                        ed["date_type"] = dtype
                            continue

                        # Calculate tight numeral_region and bbox
                        num_region = None
                        item_bbox = None
                        if bbox and len(bbox) == 4 and L > 0:
                            x0, y0, x1, y1 = bbox
                            span_x = x1 - x0
                            mx0 = round(x0 + (m.start() / L) * span_x, 4)
                            mx1 = round(x0 + (m.end() / L) * span_x, 4)
                            num_region = [mx0, y0, mx1, y1]
                            item_bbox = [mx0, y0, mx1, y1]

                        new_dates_list.append({
                            "status": "present",
                            "date_type": dtype,
                            "raw_text": m_str,
                            "month": month,
                            "year": year,
                            "confidence": conf,
                            "bbox": item_bbox,
                            "numeral_region": num_region,
                        })

                sem["dates"] = new_dates_list

        return data

    # Backward-compatibility property accessors for flat V1 code access
    @property
    def image_quality(self) -> ImageQuality:
        return self.semantic_extraction.image_quality

    @property
    def package(self) -> PackageContext:
        return self.semantic_extraction.package

    @property
    def manufacturer(self) -> ManufacturerSection:
        return self.semantic_extraction.manufacturer

    @property
    def generic_name(self) -> GenericNameDeclaration:
        return self.semantic_extraction.generic_name

    @property
    def mrp(self) -> MRPDeclaration:
        return self.semantic_extraction.mrp

    @property
    def dates(self) -> list[DateDeclaration]:
        return self.semantic_extraction.dates

    @property
    def net_quantity(self) -> NetQuantityDeclaration:
        return self.semantic_extraction.net_quantity

    @property
    def consumer_care(self) -> ConsumerCareDeclaration:
        return self.semantic_extraction.consumer_care

    @property
    def stickers(self) -> list[StickerOverlay]:
        return self.semantic_extraction.stickers

    @property
    def additional_numeric_information(self) -> list[AdditionalNumericInfo]:
        return self.semantic_extraction.additional_numeric_information

    @property
    def other_visible_text(self) -> list[OtherVisibleText]:
        return self.semantic_extraction.other_visible_text


class ExtractionResponse(BaseModel):
    """Standardized successful API response envelope."""
    success: Literal[True] = True
    data: LabelExtractionResult


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    """Standardized error API response envelope."""
    success: Literal[False] = False
    error: ErrorDetail
