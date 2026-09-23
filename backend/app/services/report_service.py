"""Deterministic PDF Inspection Report Generator using ReportLab.

Consumes the Unified Inspection JSON as its SINGLE logical input and renders a
comprehensive, multi-page statutory inspection document for the Legal Metrology Officer (LMO).

Strict Constraints:
- Pure presentation layer: NO recalculation of compliance rules or thresholds.
- Faithfully renders values, statuses, and evidence from Unified Inspection JSON.
- Two-pass canvas for dynamic 'Inspection ID: <id> | Page X of Y' running footer.
- Exact section ordering: Header, Package Info, Visibility Table, Rule 7 Table,
  Rule 8 Table, Summary Metrics, Detailed Evidence, Source Image, and Statutory Disclaimer.
"""

import base64
import io
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    Image as ReportLabImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.schemas.inspection import (
    InspectionMetadata,
    InspectionResponse,
    InspectionResponseData,
    InspectionSummary,
    UnifiedDeclarationFinding,
)
from app.services.image_storage_service import ImageStorageService

logger = logging.getLogger(__name__)

# Color Palette (Professional Regulatory/LMO Styling)
COLOR_NAVY = colors.HexColor("#1B365D")
COLOR_NAVY_LIGHT = colors.HexColor("#2C4D75")
COLOR_SLATE_DARK = colors.HexColor("#334155")
COLOR_SLATE_LIGHT = colors.HexColor("#64748B")
COLOR_BORDER = colors.HexColor("#CBD5E1")
COLOR_BG_LIGHT = colors.HexColor("#F8FAFC")
COLOR_BG_ALT = colors.HexColor("#F1F5F9")
COLOR_LINE = colors.HexColor("#E2E8F0")

# Status Accent Colors
STATUS_STYLES = {
    "PASS": {
        "text": colors.HexColor("#15803D"),
        "bg": colors.HexColor("#DCFCE7"),
        "tag": "[PASS]",
    },
    "NON-COMPLIANT": {
        "text": colors.HexColor("#B91C1C"),
        "bg": colors.HexColor("#FEE2E2"),
        "tag": "[NON-COMPLIANT]",
    },
    "NOT VISIBLE": {
        "text": colors.HexColor("#B45309"),
        "bg": colors.HexColor("#FEF3C7"),
        "tag": "[NOT VISIBLE]",
    },
    "NOT ASSESSABLE": {
        "text": colors.HexColor("#4338CA"),
        "bg": colors.HexColor("#E0E7FF"),
        "tag": "[NOT ASSESSABLE]",
    },
    "NOT APPLICABLE": {
        "text": colors.HexColor("#475569"),
        "bg": colors.HexColor("#F1F5F9"),
        "tag": "[NOT APPLICABLE]",
    },
}


class NumberedCanvas(canvas.Canvas):
    """Two-pass ReportLab canvas to compute dynamic running footers with total page counts."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: List[Dict[str, Any]] = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, total_pages: int) -> None:
        self.saveState()

        # Running Footer
        self.setFont("Helvetica", 8)
        self.setFillColor(COLOR_SLATE_LIGHT)

        insp_id = getattr(self, "inspection_id", "N/A")
        footer_left = f"Inspection ID: {insp_id}"
        footer_right = f"Page {self._pageNumber} of {total_pages}"

        # Left & Right margins are 15mm (~42.52 pt)
        page_w, _ = A4
        left_margin = 15 * mm
        right_margin = page_w - (15 * mm)

        # Thin divider line above footer
        self.setStrokeColor(COLOR_LINE)
        self.setLineWidth(0.5)
        self.line(left_margin, 28, right_margin, 28)

        # Draw footer strings
        self.drawString(left_margin, 18, footer_left)
        self.drawRightString(right_margin, 18, footer_right)

        self.restoreState()


def make_numbered_canvas(inspection_id: str) -> Type[NumberedCanvas]:
    """Factory creating NumberedCanvas subclass with bound inspection_id."""

    class BoundNumberedCanvas(NumberedCanvas):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.inspection_id = inspection_id

    return BoundNumberedCanvas


class ReportServiceError(Exception):
    """Exception raised for PDF report generation errors."""

    def __init__(self, code: str, message: str, status_code: int = 422):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _clean_text(val: Any) -> str:
    """Sanitizes text to avoid unrenderable glyphs (such as rupee symbol, typographic quotes/dashes) in standard Helvetica."""
    if val is None:
        return "N/A"
    text = str(val)
    replacements = {
        "₹": "Rs. ",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "—": "-",
        "–": "-",
        "…": "...",
        "\u00a0": " ",
        "\u200b": "",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


class ReportService:
    """Generates structured PDF inspection reports from Unified Inspection JSON."""

    def __init__(self, storage_service: Optional[ImageStorageService] = None) -> None:
        self.styles = self._init_styles()
        self.storage = storage_service or ImageStorageService()

    def _init_styles(self) -> Dict[str, ParagraphStyle]:
        """Initializes standard paragraph typography styles."""
        base = getSampleStyleSheet()

        return {
            "Title": ParagraphStyle(
                "ReportTitle",
                parent=base["Normal"],
                fontName="Helvetica-Bold",
                fontSize=18,
                leading=22,
                textColor=COLOR_NAVY,
                spaceAfter=3,
            ),
            "Subtitle": ParagraphStyle(
                "ReportSubtitle",
                parent=base["Normal"],
                fontName="Helvetica-Bold",
                fontSize=11,
                leading=14,
                textColor=COLOR_NAVY_LIGHT,
                spaceAfter=10,
            ),
            "MetaHeading": ParagraphStyle(
                "MetaHeading",
                parent=base["Normal"],
                fontName="Helvetica",
                fontSize=8.5,
                leading=11,
                textColor=COLOR_SLATE_LIGHT,
            ),
            "MetaValue": ParagraphStyle(
                "MetaValue",
                parent=base["Normal"],
                fontName="Helvetica-Bold",
                fontSize=9,
                leading=11,
                textColor=COLOR_SLATE_DARK,
            ),
            "SectionHeader": ParagraphStyle(
                "SectionHeader",
                parent=base["Normal"],
                fontName="Helvetica-Bold",
                fontSize=11,
                leading=14,
                textColor=COLOR_NAVY,
                spaceBefore=8,
                spaceAfter=5,
                keepWithNext=True,
            ),
            "TableHeader": ParagraphStyle(
                "TableHeader",
                parent=base["Normal"],
                fontName="Helvetica-Bold",
                fontSize=8,
                leading=10,
                textColor=colors.white,
                alignment=0,
            ),
            "TableCell": ParagraphStyle(
                "TableCell",
                parent=base["Normal"],
                fontName="Helvetica",
                fontSize=8,
                leading=10,
                textColor=COLOR_SLATE_DARK,
            ),
            "TableCellBold": ParagraphStyle(
                "TableCellBold",
                parent=base["Normal"],
                fontName="Helvetica-Bold",
                fontSize=8,
                leading=10,
                textColor=COLOR_SLATE_DARK,
            ),
            "MetricLabel": ParagraphStyle(
                "MetricLabel",
                parent=base["Normal"],
                fontName="Helvetica",
                fontSize=8,
                leading=10,
                textColor=COLOR_SLATE_LIGHT,
                alignment=1,
            ),
            "MetricValue": ParagraphStyle(
                "MetricValue",
                parent=base["Normal"],
                fontName="Helvetica-Bold",
                fontSize=16,
                leading=18,
                textColor=COLOR_NAVY,
                alignment=1,
            ),
            "EvidenceKey": ParagraphStyle(
                "EvidenceKey",
                parent=base["Normal"],
                fontName="Helvetica-Bold",
                fontSize=8,
                leading=10,
                textColor=COLOR_SLATE_LIGHT,
            ),
            "EvidenceVal": ParagraphStyle(
                "EvidenceVal",
                parent=base["Normal"],
                fontName="Helvetica",
                fontSize=8,
                leading=10,
                textColor=COLOR_SLATE_DARK,
            ),
            "Disclaimer": ParagraphStyle(
                "Disclaimer",
                parent=base["Normal"],
                fontName="Helvetica-Oblique",
                fontSize=7.5,
                leading=10,
                textColor=COLOR_SLATE_LIGHT,
            ),
            "RuleRefTitle": ParagraphStyle(
                "RuleRefTitle",
                parent=base["Normal"],
                fontName="Helvetica-Bold",
                fontSize=7.5,
                leading=9,
                textColor=COLOR_NAVY,
            ),
            "RuleRefBody": ParagraphStyle(
                "RuleRefBody",
                parent=base["Normal"],
                fontName="Helvetica",
                fontSize=7,
                leading=8.5,
                textColor=COLOR_SLATE_DARK,
            ),
        }

    def _normalize_inspection(self, raw_input: Any) -> InspectionResponseData:
        """Parses and validates inspection input into an InspectionResponseData instance."""
        if raw_input is None:
            raise ReportServiceError(
                code="MISSING_INSPECTION_INPUT",
                message="Inspection input payload is required.",
                status_code=422,
            )

        if isinstance(raw_input, InspectionResponseData):
            return raw_input

        if isinstance(raw_input, InspectionResponse):
            return raw_input.data

        if isinstance(raw_input, str):
            trimmed = raw_input.strip()
            if not trimmed or trimmed == "{}":
                raise ReportServiceError(
                    code="EMPTY_INSPECTION_JSON",
                    message="Inspection JSON string cannot be empty.",
                    status_code=422,
                )
            try:
                raw_input = json.loads(trimmed)
            except Exception as exc:
                raise ReportServiceError(
                    code="INVALID_INSPECTION_JSON",
                    message=f"Failed to parse inspection JSON string: {exc}",
                    status_code=422,
                ) from exc

        if isinstance(raw_input, dict):
            # Unwrap optional 'data' envelope
            if "data" in raw_input and isinstance(raw_input["data"], dict) and "findings" in raw_input["data"]:
                raw_input = raw_input["data"]

            # MongoDB adds persistence metadata around the unchanged unified payload.
            raw_input = {
                key: value
                for key, value in raw_input.items()
                if key not in {"_id", "created_at", "updated_at", "schema_version", "review", "artifacts"}
            }

            try:
                return InspectionResponseData.model_validate(raw_input)
            except Exception as exc:
                raise ReportServiceError(
                    code="INVALID_INSPECTION_SCHEMA",
                    message=f"Inspection payload does not conform to Unified Inspection schema: {exc}",
                    status_code=422,
                ) from exc

        raise ReportServiceError(
            code="UNSUPPORTED_INSPECTION_TYPE",
            message=f"Unsupported inspection input type: {type(raw_input).__name__}",
            status_code=422,
        )

    def _format_status_badge(self, status: str) -> Paragraph:
        """Renders an accessible text status tag with contrasting style."""
        cfg = STATUS_STYLES.get(status, STATUS_STYLES["NOT APPLICABLE"])
        tag_text = cfg["tag"]
        color_hex = cfg["text"].hexval()
        if len(color_hex) == 8:
            color_hex = f"#{color_hex[2:]}"  # strip alpha channel if 0xRRGGBB00
        elif len(color_hex) == 9:
            color_hex = f"#{color_hex[3:]}"

        style = ParagraphStyle(
            f"Status_{status}",
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=9,
            textColor=cfg["text"],
        )
        return Paragraph(f"<b>{tag_text}</b>", style)

    def _create_rule_reference_box(self, title: str, text: str, width: float) -> Table:
        """Creates a compact regulatory reference callout box for LMO context."""
        content = [
            Paragraph(f"<b>APPLICABLE RULE REFERENCE:</b> {title}", self.styles["RuleRefTitle"]),
            Spacer(1, 2),
            Paragraph(text, self.styles["RuleRefBody"]),
        ]
        box = Table([[content]], colWidths=[width])
        box.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), COLOR_BG_ALT),
                    ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        return box

    def generate_report(
        self,
        inspection: Union[InspectionResponseData, InspectionResponse, dict, str],
        image_base64: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """Renders Unified Inspection JSON into a multi-page PDF inspection report.

        Args:
            inspection: Unified inspection result (Pydantic model, envelope, dict, or JSON string).
            image_base64: Optional legacy base64 encoded packaging photo.
            output_path: Optional destination filepath.

        Returns:
            Absolute file path to the generated PDF document.
        """
        inspection_data = self._normalize_inspection(inspection)
        inspection_id = inspection_data.inspection_id

        # Determine target file destination
        if not output_path:
            output_path = str(self.storage.get_report_path(inspection_id))
        else:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Persist Unified Inspection JSON snapshot in inspection folder
        try:
            self.storage.save_inspection_json(inspection_id, inspection_data)
        except Exception as exc:
            logger.warning("Could not persist inspection JSON snapshot for %s: %s", inspection_id, exc)

        # A4 page size = 595.27 x 841.89 pt. 15mm margins = 42.52 pt. Printable width = 510.23 pt
        page_width, page_height = A4
        margin_pt = 15 * mm
        printable_width = page_width - (2 * margin_pt)

        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            leftMargin=margin_pt,
            rightMargin=margin_pt,
            topMargin=margin_pt,
            bottomMargin=margin_pt + 8,  # allow space for footer line
        )

        story: List[Any] = []

        # SECTION 0 — HEADER
        self._add_header_section(story, inspection_data, printable_width)

        # SECTION 1 — PACKAGE & INSPECTION INFORMATION
        self._add_package_info_section(story, inspection_data, printable_width)

        # SECTION 2 — DECLARATION VISIBILITY
        self._add_visibility_section(story, inspection_data, printable_width)

        # SECTION 3 — RULE 7 NUMERAL-HEIGHT ASSESSMENT
        self._add_rule7_section(story, inspection_data, printable_width)

        # SECTION 4 — RULE 8 SPATIAL CLEARANCE
        self._add_rule8_section(story, inspection_data, printable_width)

        # SECTION 5 — FINDINGS SUMMARY
        self._add_summary_section(story, inspection_data, printable_width)

        # SECTION 6 — DETAILED FINDINGS & EVIDENCE
        self._add_detailed_evidence_section(story, inspection_data, printable_width)

        # SECTION 7 — SOURCE IMAGE EVIDENCE
        # Strict Image Source Priority:
        # 1. Stored original image for inspection_id (always takes precedence)
        # 2. Legacy image_base64 fallback
        # 3. No image
        target_image_bytes: Optional[bytes] = None
        is_stored_original: bool = False
        verification_status: str = "UNVERIFIED"
        image_sha256: Optional[str] = None

        expected_hash = (
            inspection_data.metadata.source_image_sha256
            if inspection_data.metadata
            else None
        )

        if self.storage.exists(inspection_id):
            is_valid, status, computed_hash = self.storage.verify_hash(
                inspection_id, expected_hash
            )
            if status == "MISMATCH":
                raise ReportServiceError(
                    code="SOURCE_IMAGE_HASH_MISMATCH",
                    message="Stored source image does not match the inspection source image hash.",
                    status_code=422,
                )
            target_image_bytes = self.storage.get_original_bytes(inspection_id)
            is_stored_original = True
            verification_status = status  # "MATCHED" or "UNVERIFIED"
            image_sha256 = computed_hash

            if image_base64:
                logger.info(
                    "Stored original image takes precedence for inspection %s; ignoring legacy image_base64 parameter.",
                    inspection_id,
                )

        elif image_base64:
            # Legacy fallback path
            try:
                raw_b64 = image_base64.strip()
                if "," in raw_b64:
                    raw_b64 = raw_b64.split(",", 1)[1]
                decoded_bytes = base64.b64decode(raw_b64)
                computed_hash = self.storage.calculate_sha256(decoded_bytes)
                image_sha256 = computed_hash
                if expected_hash:
                    if computed_hash.lower() == expected_hash.strip().lower():
                        verification_status = "MATCHED"
                    else:
                        raise ReportServiceError(
                            code="SOURCE_IMAGE_HASH_MISMATCH",
                            message="Stored source image does not match the inspection source image hash.",
                            status_code=422,
                        )
                else:
                    verification_status = "UNVERIFIED"
                target_image_bytes = decoded_bytes
            except ReportServiceError:
                raise
            except Exception as exc:
                logger.warning("Failed to decode legacy base64 image: %s", exc)
                target_image_bytes = b"corrupted"
                verification_status = "UNVERIFIED"
                image_sha256 = None
            is_stored_original = False

        if target_image_bytes is not None:
            self._add_source_image_section(
                story=story,
                image_bytes=target_image_bytes,
                width=printable_width,
                is_stored_original=is_stored_original,
                verification_status=verification_status,
                sha256_hash=image_sha256,
            )

        # SECTION 10 — LEGAL DISCLAIMER & LMO NOTES
        self._add_disclaimer_section(story, inspection_data, printable_width)

        # Build document using two-pass NumberedCanvas
        canvas_cls = make_numbered_canvas(inspection_data.inspection_id)
        doc.build(story, canvasmaker=canvas_cls)

        return str(Path(output_path).resolve())

    def _add_header_section(
        self,
        story: List[Any],
        data: InspectionResponseData,
        width: float,
    ) -> None:
        """Renders Document Title, Subtitle, and metadata banner."""
        story.append(Paragraph("AUTOMATED COMPLIANCE INSPECTION REPORT", self.styles["Title"]))
        story.append(
            Paragraph("Packaged Commodities - SIH 2026 PS 26034", self.styles["Subtitle"])
        )

        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        header_data = [
            [
                Paragraph("<b>Inspection ID:</b>", self.styles["MetaHeading"]),
                Paragraph(data.inspection_id, self.styles["MetaValue"]),
                Paragraph("<b>Generated:</b>", self.styles["MetaHeading"]),
                Paragraph(now_utc, self.styles["MetaValue"]),
            ]
        ]
        header_table = Table(header_data, colWidths=[width * 0.18, width * 0.32, width * 0.16, width * 0.34])
        header_table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                    ("TOPPADDING", (0, 0), (-1, -1), 2),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )
        story.append(header_table)
        story.append(Spacer(1, 4))
        story.append(HRFlowable(width="100%", thickness=1.5, color=COLOR_NAVY, spaceAfter=8))

    def _add_package_info_section(
        self,
        story: List[Any],
        data: InspectionResponseData,
        width: float,
    ) -> None:
        """SECTION 1: Package & Inspection Information table."""
        story.append(Paragraph("1. PACKAGE & INSPECTION INFORMATION", self.styles["SectionHeader"]))

        meta = data.metadata

        # Format dimensions
        width_str = "N/A"
        height_str = "N/A"
        if meta.package_dimensions_mm:
            w_val = meta.package_dimensions_mm.get("width_mm")
            h_val = meta.package_dimensions_mm.get("height_mm")
            if w_val is not None:
                width_str = f"{w_val:.1f} mm"
            if h_val is not None:
                height_str = f"{h_val:.1f} mm"

        pdp_str = f"{meta.pdp_area_cm2:.2f} cm2" if meta.pdp_area_cm2 is not None else "N/A"

        info_rows = [
            [
                Paragraph("Brand Name", self.styles["MetaHeading"]),
                Paragraph(_clean_text(meta.brand_name or "N/A"), self.styles["TableCellBold"]),
                Paragraph("Generic Name", self.styles["MetaHeading"]),
                Paragraph(_clean_text(meta.generic_name or "N/A"), self.styles["TableCellBold"]),
            ],
            [
                Paragraph("Package Type", self.styles["MetaHeading"]),
                Paragraph(_clean_text(meta.package_type or "N/A"), self.styles["TableCell"]),
                Paragraph("Rule Source Mode", self.styles["MetaHeading"]),
                Paragraph(_clean_text(meta.rule_source_mode or "N/A"), self.styles["TableCellBold"]),
            ],
            [
                Paragraph("Package Width", self.styles["MetaHeading"]),
                Paragraph(width_str, self.styles["TableCell"]),
                Paragraph("Package Height", self.styles["MetaHeading"]),
                Paragraph(height_str, self.styles["TableCell"]),
            ],
            [
                Paragraph("PDP Face Area", self.styles["MetaHeading"]),
                Paragraph(pdp_str, self.styles["TableCell"]),
                Paragraph("Image Quality", self.styles["MetaHeading"]),
                Paragraph(_clean_text(meta.image_quality or "N/A"), self.styles["TableCell"]),
            ],
        ]

        info_table = Table(info_rows, colWidths=[width * 0.22, width * 0.28, width * 0.22, width * 0.28])
        info_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), COLOR_BG_LIGHT),
                    ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_LINE),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(info_table)
        story.append(Spacer(1, 8))

    def _add_visibility_section(
        self,
        story: List[Any],
        data: InspectionResponseData,
        width: float,
    ) -> None:
        """SECTION 2: Declaration Visibility Table."""
        story.append(Paragraph("2. DECLARATION VISIBILITY (RULE 6)", self.styles["SectionHeader"]))

        header_row = [
            Paragraph("Declaration", self.styles["TableHeader"]),
            Paragraph("Visibility", self.styles["TableHeader"]),
            Paragraph("Detected Value / Text", self.styles["TableHeader"]),
            Paragraph("Status", self.styles["TableHeader"]),
        ]

        table_rows = [header_row]

        for finding in data.findings:
            det_val = _clean_text(finding.detected_value or finding.raw_text or "-")
            # Truncate very long raw_text in compact visibility table
            if len(det_val) > 75:
                det_val = det_val[:72] + "..."

            vis_text = f"<b>{_clean_text(finding.visibility)}</b>"
            badge = self._format_status_badge(finding.status)

            table_rows.append(
                [
                    Paragraph(_clean_text(finding.display_name), self.styles["TableCellBold"]),
                    Paragraph(vis_text, self.styles["TableCell"]),
                    Paragraph(det_val, self.styles["TableCell"]),
                    badge,
                ]
            )

        col_w = [width * 0.35, width * 0.17, width * 0.32, width * 0.16]
        vis_table = Table(table_rows, colWidths=col_w, repeatRows=1)
        vis_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), COLOR_NAVY),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_LINE),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_BG_ALT]),
                ]
            )
        )
        story.append(vis_table)
        story.append(Spacer(1, 4))
        r6_box = self._create_rule_reference_box(
            title="Rule 6 — Mandatory Package Declarations",
            text=(
                "Mandatory declaration categories: (1) Manufacturer/Packer/Importer, (2) Generic/Common Name, "
                "(3) Net Quantity, (4) Retail Sale Price (MRP), (5) Manufacture/Packing/Import Date, (6) Consumer Care Details. "
                "Notice: 'NOT VISIBLE' indicates declaration was undetected in the supplied image/panel; it does not establish legal absence."
            ),
            width=width,
        )
        story.append(r6_box)
        story.append(Spacer(1, 8))

    def _add_rule7_section(
        self,
        story: List[Any],
        data: InspectionResponseData,
        width: float,
    ) -> None:
        """SECTION 3: Rule 7 Numeral-Height Assessment Table."""
        rule7_findings = [f for f in data.findings if f.rules.rule7 is not None]
        if not rule7_findings:
            return

        story.append(Paragraph("3. RULE 7 NUMERAL-HEIGHT ASSESSMENT", self.styles["SectionHeader"]))

        header_row = [
            Paragraph("Declaration", self.styles["TableHeader"]),
            Paragraph("Status", self.styles["TableHeader"]),
            Paragraph("Measured", self.styles["TableHeader"]),
            Paragraph("Required", self.styles["TableHeader"]),
            Paragraph("Margin", self.styles["TableHeader"]),
            Paragraph("Quality", self.styles["TableHeader"]),
            Paragraph("Conf", self.styles["TableHeader"]),
            Paragraph("Rule Source", self.styles["TableHeader"]),
        ]

        table_rows = [header_row]

        for finding in rule7_findings:
            r7 = finding.rules.rule7
            assert r7 is not None

            measured_str = f"{r7.measured_height_mm:.2f} mm" if r7.measured_height_mm is not None else "-"
            required_str = f"{r7.required_height_mm:.2f} mm" if r7.required_height_mm is not None else "-"

            margin_str = "-"
            if r7.height_margin_mm is not None:
                sign = "+" if r7.height_margin_mm > 0 else ""
                margin_str = f"{sign}{r7.height_margin_mm:.2f} mm"

            quality_str = _clean_text(r7.measurement_quality or "-")
            conf_str = f"{r7.confidence:.2f}" if r7.confidence is not None else "-"
            mode_str = _clean_text(r7.rule_source_mode or "-")

            badge = self._format_status_badge(r7.status)

            table_rows.append(
                [
                    Paragraph(_clean_text(finding.display_name), self.styles["TableCellBold"]),
                    badge,
                    Paragraph(measured_str, self.styles["TableCell"]),
                    Paragraph(required_str, self.styles["TableCell"]),
                    Paragraph(margin_str, self.styles["TableCellBold"]),
                    Paragraph(quality_str, self.styles["TableCell"]),
                    Paragraph(conf_str, self.styles["TableCell"]),
                    Paragraph(mode_str, self.styles["TableCell"]),
                ]
            )

        col_w = [
            width * 0.23,
            width * 0.16,
            width * 0.11,
            width * 0.11,
            width * 0.11,
            width * 0.08,
            width * 0.06,
            width * 0.14,
        ]
        r7_table = Table(table_rows, colWidths=col_w, repeatRows=1)
        r7_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), COLOR_NAVY),
                    ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_LINE),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_BG_ALT]),
                ]
            )
        )
        story.append(r7_table)
        story.append(Spacer(1, 4))
        mode_val = data.metadata.rule_source_mode if data.metadata and data.metadata.rule_source_mode else "sih_ps_26034"
        r7_box = self._create_rule_reference_box(
            title="Rule 7 — Letter and Numeral Size Requirements",
            text=(
                f"Automated Source Mode: {mode_val.upper()} | "
                "Automated result is based on SIH PS 26034 mode. Statutory verification should be performed by the LMO against the applicable rule version. &nbsp;&nbsp; "
                "<b>LMO Verification:</b> [ &nbsp; ] Verify &nbsp;&nbsp; [ &nbsp; ] Issue &nbsp;&nbsp; [ &nbsp; ] N/A"
            ),
            width=width,
        )
        story.append(r7_box)
        story.append(Spacer(1, 8))

    def _add_rule8_section(
        self,
        story: List[Any],
        data: InspectionResponseData,
        width: float,
    ) -> None:
        """SECTION 4: Rule 8 Spatial Clearance Table."""
        rule8_findings = [f for f in data.findings if f.rules.rule8 is not None]
        if not rule8_findings:
            return

        story.append(Paragraph("4. RULE 8 SPATIAL CLEARANCE ASSESSMENT", self.styles["SectionHeader"]))

        header_row = [
            Paragraph("Declaration", self.styles["TableHeader"]),
            Paragraph("Status", self.styles["TableHeader"]),
            Paragraph("Top (m/r)", self.styles["TableHeader"]),
            Paragraph("Bottom (m/r)", self.styles["TableHeader"]),
            Paragraph("Left (m/r)", self.styles["TableHeader"]),
            Paragraph("Right (m/r)", self.styles["TableHeader"]),
            Paragraph("Numeral H", self.styles["TableHeader"]),
            Paragraph("Conf", self.styles["TableHeader"]),
        ]

        table_rows = [header_row]

        def fmt_dir(clr: Any) -> str:
            if not clr or clr.measured_mm is None or clr.required_mm is None:
                return "-"
            status_clean = clr.status.replace("_", " ")
            return f"{clr.measured_mm:.2f} / {clr.required_mm:.2f} mm<br/>({status_clean})"

        for finding in rule8_findings:
            r8 = finding.rules.rule8
            assert r8 is not None

            h_str = f"{r8.target_numeral_height_mm:.2f} mm" if r8.target_numeral_height_mm is not None else "-"
            conf_str = f"{r8.confidence:.2f}" if r8.confidence is not None else "-"
            badge = self._format_status_badge(r8.status)

            table_rows.append(
                [
                    Paragraph(_clean_text(finding.display_name), self.styles["TableCellBold"]),
                    badge,
                    Paragraph(fmt_dir(r8.top), self.styles["TableCell"]),
                    Paragraph(fmt_dir(r8.bottom), self.styles["TableCell"]),
                    Paragraph(fmt_dir(r8.left), self.styles["TableCell"]),
                    Paragraph(fmt_dir(r8.right), self.styles["TableCell"]),
                    Paragraph(h_str, self.styles["TableCell"]),
                    Paragraph(conf_str, self.styles["TableCell"]),
                ]
            )

        col_w = [
            width * 0.18,
            width * 0.14,
            width * 0.14,
            width * 0.14,
            width * 0.13,
            width * 0.13,
            width * 0.08,
            width * 0.06,
        ]
        r8_table = Table(table_rows, colWidths=col_w, repeatRows=1)
        r8_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), COLOR_NAVY),
                    ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_LINE),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_BG_ALT]),
                ]
            )
        )
        story.append(r8_table)
        story.append(Spacer(1, 4))
        r8_box = self._create_rule_reference_box(
            title="Rule 8(1) Proviso — Clear Space Surrounding Net-Quantity Declaration",
            text=(
                "Reference geometry shown in this report: Top / Bottom -> H &nbsp;|&nbsp; Left / Right -> 2H (where H = numeral height). "
                "Clear space must be free of all printed words and inscriptions."
            ),
            width=width,
        )
        story.append(r8_box)
        story.append(Spacer(1, 8))

    def _add_summary_section(
        self,
        story: List[Any],
        data: InspectionResponseData,
        width: float,
    ) -> None:
        """SECTION 5: Findings Summary (Counts Only — Zero Scores or Verdicts)."""
        s = data.summary

        # Row 1: Declaration visibility counts
        row1 = [
            [
                Paragraph(str(s.total_declarations_evaluated), self.styles["MetricValue"]),
                Paragraph(str(s.detected_declarations_count), self.styles["MetricValue"]),
                Paragraph(str(s.not_visible_declarations_count), self.styles["MetricValue"]),
            ],
            [
                Paragraph("Total Declarations", self.styles["MetricLabel"]),
                Paragraph("Detected in Image", self.styles["MetricLabel"]),
                Paragraph("Not Visible in Image", self.styles["MetricLabel"]),
            ],
        ]

        t1 = Table(row1, colWidths=[width / 3] * 3)
        t1.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), COLOR_BG_LIGHT),
                    ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_LINE),
                    ("TOPPADDING", (0, 0), (-1, 0), 4),
                    ("BOTTOMPADDING", (0, -1), (-1, -1), 4),
                ]
            )
        )

        # Row 2: Specific Rule findings counts (Rule 7 and Rule 8)
        row2 = [
            [
                Paragraph(str(s.rules_evaluated_count), self.styles["MetricValue"]),
                Paragraph(str(s.pass_findings_count), self.styles["MetricValue"]),
                Paragraph(str(s.non_compliant_findings_count), self.styles["MetricValue"]),
                Paragraph(str(s.not_assessable_findings_count), self.styles["MetricValue"]),
            ],
            [
                Paragraph("Rules Evaluated", self.styles["MetricLabel"]),
                Paragraph("PASS Rules", self.styles["MetricLabel"]),
                Paragraph("NON-COMPLIANT Rules", self.styles["MetricLabel"]),
                Paragraph("NOT ASSESSABLE Rules", self.styles["MetricLabel"]),
            ],
        ]

        t2 = Table(row2, colWidths=[width / 4] * 4)
        t2.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), COLOR_BG_LIGHT),
                    ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                    ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_LINE),
                    ("TOPPADDING", (0, 0), (-1, 0), 4),
                    ("BOTTOMPADDING", (0, -1), (-1, -1), 4),
                ]
            )
        )
        story.append(
            KeepTogether(
                [
                    Paragraph("5. INSPECTION SUMMARY COUNTS", self.styles["SectionHeader"]),
                    t1,
                    Spacer(1, 4),
                    t2,
                    Spacer(1, 8),
                ]
            )
        )

    def _add_detailed_evidence_section(
        self,
        story: List[Any],
        data: InspectionResponseData,
        width: float,
    ) -> None:
        """SECTION 6: Detailed Findings & Evidence (Itemized structured cards)."""
        story.append(Paragraph("6. DETAILED DECLARATION FINDINGS & EVIDENCE", self.styles["SectionHeader"]))

        for idx, finding in enumerate(data.findings, 1):
            badge = self._format_status_badge(finding.status)

            # Format normalized bounding box [x_min, y_min, x_max, y_max]
            bbox_str = "N/A"
            if finding.bbox:
                coords = [f"{c:.3f}" for c in finding.bbox]
                bbox_str = f"[{', '.join(coords)}]"

            conf_str = f"{finding.confidence:.2f}" if finding.confidence is not None else "N/A"

            card_rows = [
                # Card Header
                [
                    Paragraph(f"<b>{idx}. {_clean_text(finding.display_name)}</b>", self.styles["TableCellBold"]),
                    Paragraph(f"Visibility: <b>{_clean_text(finding.visibility)}</b>", self.styles["TableCell"]),
                    Paragraph("Status:", self.styles["TableCellBold"]),
                    badge,
                ],
                # Raw text
                [
                    Paragraph("Raw Text", self.styles["EvidenceKey"]),
                    Paragraph(_clean_text(finding.raw_text or "N/A"), self.styles["EvidenceVal"]),
                    Paragraph("Detected Val", self.styles["EvidenceKey"]),
                    Paragraph(_clean_text(finding.detected_value or "N/A"), self.styles["EvidenceVal"]),
                ],
                # Numeral, Unit, Conf, BBox
                [
                    Paragraph("Numeral / Unit", self.styles["EvidenceKey"]),
                    Paragraph(
                        _clean_text(f"{finding.declared_numeral or 'N/A'} {finding.declared_unit or ''}".strip()),
                        self.styles["EvidenceVal"],
                    ),
                    Paragraph("Conf / BBox", self.styles["EvidenceKey"]),
                    Paragraph(f"{conf_str} | {bbox_str}", self.styles["EvidenceVal"]),
                ],
                # Rule Reference & Notes
                [
                    Paragraph("Rule Ref", self.styles["EvidenceKey"]),
                    Paragraph(_clean_text(finding.rule_reference or "N/A"), self.styles["EvidenceVal"]),
                    Paragraph("Notes", self.styles["EvidenceKey"]),
                    Paragraph(_clean_text(finding.notes or "-"), self.styles["EvidenceVal"]),
                ],
            ]

            if finding.rules and finding.rules.rule7:
                r7 = finding.rules.rule7
                meas_req = f"{r7.measured_height_mm:.2f} mm (req: {r7.required_height_mm:.2f} mm)" if r7.measured_height_mm and r7.required_height_mm else "-"
                card_rows.append([
                    Paragraph("Rule 7 Status", self.styles["EvidenceKey"]),
                    self._format_status_badge(r7.status),
                    Paragraph("Rule 7 Evidence", self.styles["EvidenceKey"]),
                    Paragraph(_clean_text(f"{meas_req} | {r7.notes or '-'}"), self.styles["EvidenceVal"]),
                ])

            if finding.rules and finding.rules.rule8:
                r8 = finding.rules.rule8
                card_rows.append([
                    Paragraph("Rule 8 Status", self.styles["EvidenceKey"]),
                    self._format_status_badge(r8.status),
                    Paragraph("Rule 8 Evidence", self.styles["EvidenceKey"]),
                    Paragraph(_clean_text(r8.notes or f"Numeral H: {r8.target_numeral_height_mm:.2f} mm"), self.styles["EvidenceVal"]),
                ])

            card_table = Table(
                card_rows,
                colWidths=[width * 0.18, width * 0.40, width * 0.16, width * 0.26],
            )
            card_table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), COLOR_BG_ALT),
                        ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                        ("INNERGRID", (0, 0), (-1, -1), 0.5, COLOR_LINE),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 3),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )

            story.append(KeepTogether([card_table, Spacer(1, 5)]))

        story.append(Spacer(1, 4))

    def _add_source_image_section(
        self,
        story: List[Any],
        image_bytes: bytes,
        width: float,
        is_stored_original: bool = True,
        verification_status: str = "MATCHED",
        sha256_hash: Optional[str] = None,
    ) -> None:
        """SECTION 7: Source Image Evidence (Safely scaled, preserving aspect ratio)."""
        header_para = Paragraph("7. SOURCE PACKAGE IMAGE EVIDENCE", self.styles["SectionHeader"])

        source_label = "ORIGINAL STORED IMAGE" if is_stored_original else "LEGACY ATTACHMENT"
        meta_parts = [
            f"<b>Source:</b> {source_label}",
            f"<b>Verification:</b> {verification_status}",
        ]
        if sha256_hash:
            meta_parts.append(f"<b>SHA-256:</b> <font name='Courier' size='6.5'>{sha256_hash}</font>")
        meta_line = " &nbsp;&nbsp;|&nbsp;&nbsp; ".join(meta_parts)
        meta_para = Paragraph(meta_line, self.styles["TableCell"])

        try:
            # Inspect original dimensions and format using Pillow
            pil_img = PILImage.open(io.BytesIO(image_bytes))
            if pil_img.format not in ("PNG", "JPEG", "JPG", "MPO", "WEBP", "BMP", "TIFF", "GIF"):
                raise ValueError(f"Unsupported image format: {pil_img.format}")

            orig_w, orig_h = pil_img.size

            if orig_w <= 0 or orig_h <= 0:
                raise ValueError("Invalid image dimensions")

            # Max constraints: width=450 pt, height=300 pt
            max_w = min(450.0, width)
            max_h = 300.0

            scale = min(max_w / orig_w, max_h / orig_h, 1.0)
            target_w = orig_w * scale
            target_h = orig_h * scale

            report_img = ReportLabImage(io.BytesIO(image_bytes), width=target_w, height=target_h)

            img_table = Table([[report_img]], colWidths=[width])
            img_table.setStyle(
                TableStyle(
                    [
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                        ("BACKGROUND", (0, 0), (-1, -1), COLOR_BG_LIGHT),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            story.append(KeepTogether([header_para, meta_para, Spacer(1, 4), img_table, Spacer(1, 8)]))

        except Exception as exc:
            logger.warning("Failed to render source package image in report: %s", exc)
            fallback = Table(
                [[Paragraph("<i>Source image could not be rendered.</i>", self.styles["TableCell"])]],
                colWidths=[width],
            )
            fallback.setStyle(
                TableStyle(
                    [
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                        ("BACKGROUND", (0, 0), (-1, -1), COLOR_BG_LIGHT),
                        ("TOPPADDING", (0, 0), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ]
                )
            )
            story.append(KeepTogether([header_para, meta_para, Spacer(1, 4), fallback, Spacer(1, 8)]))

    def _add_disclaimer_section(
        self,
        story: List[Any],
        data: InspectionResponseData,
        width: float,
    ) -> None:
        """SECTION 10: Legal Metrology Advisory Notice / Disclaimer & LMO Review Notes."""
        story.append(Spacer(1, 4))
        story.append(HRFlowable(width="100%", thickness=0.5, color=COLOR_LINE, spaceAfter=4))

        # LMO Review Notes Box
        lmo_notes = [
            "<b>LMO REVIEW NOTES & STATUTORY INSTRUCTIONS</b>",
            "1. Automated findings are inspection-support evidence for the Legal Metrology Officer (LMO).",
            "2. 'NOT VISIBLE' means not detected in the supplied image/PDP; it does not establish legal absence from unsubmitted faces.",
            "3. SIH automated rule mode must not be assumed to equal current statutory thresholds where amendments differ.",
            "4. Final legal determination and statutory enforcement rest solely with the competent Legal Metrology Officer.",
            "5. Current applicable amendments/rules should be verified before any statutory enforcement action is initiated.",
        ]
        lmo_para = Paragraph("<br/>".join(lmo_notes), self.styles["RuleRefBody"])
        lmo_table = Table([[lmo_para]], colWidths=[width])
        lmo_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), COLOR_BG_LIGHT),
                    ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(lmo_table)
        story.append(Spacer(1, 4))

        disclaimer_text = (
            f"<b>LEGAL METROLOGY STATUTORY ADVISORY NOTICE:</b> {data.disclaimer} "
            "This document is an automated inspection-support report generated under SIH 2026 PS 26034. "
            "It does not constitute an independent statutory enforcement order under the Legal Metrology Act, 2009."
        )

        story.append(Paragraph(disclaimer_text, self.styles["Disclaimer"]))
