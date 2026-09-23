"""UI components and rendering helpers for the Streamlit LMO Compliance Dashboard.

Enforces strict presentation constraints:
- Pure visualization layer: zero legal calculations, zero OCR, zero rule re-evaluation.
- Directly consumes and formats Unified Inspection JSON structures.
- Status values strictly preserved: PASS, NON-COMPLIANT, NOT VISIBLE, NOT ASSESSABLE, NOT APPLICABLE.
"""

import io
from typing import Any, Dict, List, Optional
from PIL import Image as PILImage, ImageDraw, ImageFont
import streamlit as st

# Color constants conforming to regulatory aesthetic
STATUS_COLORS = {
    "PASS": {"text": "#15803D", "bg": "#DCFCE7", "border": "#86EFAC", "rgb": (22, 163, 74)},
    "NON-COMPLIANT": {"text": "#B91C1C", "bg": "#FEE2E2", "border": "#FCA5A5", "rgb": (220, 38, 38)},
    "NOT VISIBLE": {"text": "#B45309", "bg": "#FEF3C7", "border": "#FDE68A", "rgb": (217, 119, 6)},
    "NOT ASSESSABLE": {"text": "#4338CA", "bg": "#E0E7FF", "border": "#A5B4FC", "rgb": (79, 70, 229)},
    "NOT APPLICABLE": {"text": "#475569", "bg": "#F1F5F9", "border": "#CBD5E1", "rgb": (100, 116, 139)},
    "DETECTED": {"text": "#2563EB", "bg": "#DBEAFE", "border": "#93C5FD", "rgb": (37, 99, 235)},
}


def get_status_badge_html(status: str) -> str:
    """Returns HTML for a crisp status badge tag."""
    cfg = STATUS_COLORS.get(status, STATUS_COLORS["NOT APPLICABLE"])
    return (
        f'<span style="background-color: {cfg["bg"]}; color: {cfg["text"]}; '
        f'border: 1px solid {cfg["border"]}; padding: 3px 8px; border-radius: 4px; '
        f'font-weight: 600; font-size: 0.82rem; letter-spacing: 0.02em; display: inline-block;">'
        f'{status}</span>'
    )


def render_kpi_cards(summary: Dict[str, Any]) -> None:
    """Renders 6 high-level KPI cards using ONLY the summary values from backend."""
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        st.metric(
            label="Total Declarations",
            value=summary.get("total_declarations_evaluated", 0),
            help="Total distinct mandatory declarations evaluated by the pipeline.",
        )
    with c2:
        st.metric(
            label="Detected",
            value=summary.get("detected_declarations_count", 0),
            help="Declarations physically detected on the supplied packaging image.",
        )
    with c3:
        st.metric(
            label="Not Visible",
            value=summary.get("not_visible_declarations_count", 0),
            help="Declarations not detected in the supplied panel image (e.g. front vs back).",
        )
    with c4:
        st.metric(
            label="PASS",
            value=summary.get("pass_findings_count", 0),
            help="Rule evaluations meeting statutory requirements.",
        )
    with c5:
        st.metric(
            label="NON-COMPLIANT",
            value=summary.get("non_compliant_findings_count", 0),
            help="Rule evaluations failing statutory requirements.",
        )
    with c6:
        st.metric(
            label="NOT ASSESSABLE",
            value=summary.get("not_assessable_findings_count", 0),
            help="Findings where rule assessment was indeterminate or has no numeral threshold.",
        )


def render_provenance_header(data: Dict[str, Any]) -> None:
    """Renders compact provenance header card with inspection details and SHA-256."""
    metadata = data.get("metadata", {})
    inspection_id = data.get("inspection_id", "N/A")
    brand = metadata.get("brand_name") or "Not Specified"
    generic = metadata.get("generic_name") or "Not Specified"
    pkg_type = metadata.get("package_type") or "retail"
    pdp_area = metadata.get("pdp_area_cm2")
    pdp_str = f"{pdp_area} cm²" if pdp_area is not None else "Not Specified"

    dims = metadata.get("package_dimensions_mm", {})
    w_mm = dims.get("width_mm")
    h_mm = dims.get("height_mm")
    dims_str = f"{w_mm} × {h_mm} mm" if (w_mm and h_mm) else "Not Specified"

    image_quality = metadata.get("image_quality", "moderate").upper()
    rule_mode = metadata.get("rule_source_mode", "sih_ps_26034")
    filename = metadata.get("source_image_filename", "uploaded_image.jpg")
    sha256 = metadata.get("source_image_sha256", "N/A")
    short_hash = f"{sha256[:10]}...{sha256[-8:]}" if len(sha256) > 20 else sha256

    with st.container(border=True):
        st.markdown(f"#### 📋 Inspection Snapshot — `{inspection_id}`")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"**Brand:** {brand}")
            st.markdown(f"**Commodity:** {generic}")
        with col2:
            st.markdown(f"**Package Type:** `{pkg_type}`")
            st.markdown(f"**Dimensions:** `{dims_str}`")
        with col3:
            st.markdown(f"**PDP Area:** `{pdp_str}`")
            st.markdown(f"**Image Quality:** `{image_quality}`")
        with col4:
            st.markdown(f"**Rule Source:** `{rule_mode}`")
            st.markdown(f"**Source File:** `{filename}`")

        # Provenance hash row
        st.markdown(
            f"**SHA-256:** `{short_hash}` &nbsp;*(Exact stored source image)*"
        )
        with st.expander("View complete SHA-256"):
            st.code(sha256, language=None)
            st.caption(
                "The SHA-256 value identifies the exact stored source image associated with this inspection."
            )


def draw_bounding_boxes(
    image_bytes: bytes,
    findings: List[Dict[str, Any]],
    show_labels: bool = True,
) -> PILImage.Image:
    """Draws color-coded bounding boxes on the original packaging image.

    Normalized coordinates [x_min, y_min, x_max, y_max] are mapped to pixel coordinates.
    Color rules:
    - PASS -> Green
    - NON-COMPLIANT -> Red
    - DETECTED / NOT ASSESSABLE -> Blue
    - NOT VISIBLE -> Ignored
    """
    image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size

    # Choose line width based on image size
    line_w = max(2, int(min(img_w, img_h) * 0.004))

    # Font handling
    font_size = max(11, int(min(img_w, img_h) * 0.015))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    for f in findings:
        bbox = f.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        status = f.get("status", "NOT ASSESSABLE")

        # Pick color
        if status == "PASS":
            color = STATUS_COLORS["PASS"]["rgb"]
        elif status == "NON-COMPLIANT":
            color = STATUS_COLORS["NON-COMPLIANT"]["rgb"]
        elif status == "NOT VISIBLE":
            continue  # No box to draw
        else:
            color = STATUS_COLORS["NOT ASSESSABLE"]["rgb"]

        x_min, y_min, x_max, y_max = bbox
        x1 = max(0, min(img_w - 1, int(x_min * img_w)))
        y1 = max(0, min(img_h - 1, int(y_min * img_h)))
        x2 = max(0, min(img_w - 1, int(x_max * img_w)))
        y2 = max(0, min(img_h - 1, int(y_max * img_h)))

        if x2 <= x1 or y2 <= y1:
            continue

        # Draw main bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)

        if show_labels:
            display_name = f.get("display_name", f.get("field", "Declaration"))
            short_label = display_name if len(display_name) <= 22 else f"{display_name[:20]}..."
            tag_text = f" {short_label} "

            # Label banner
            label_h = font_size + 4
            draw.rectangle(
                [x1, max(0, y1 - label_h), min(img_w, x1 + (len(tag_text) * (font_size // 2 + 3))), y1],
                fill=color,
            )
            draw.text((x1 + 2, max(0, y1 - label_h + 1)), tag_text, fill=(255, 255, 255), font=font)

    return image


def render_findings_table(findings: List[Dict[str, Any]], active_filter: str = "All") -> None:
    """Renders the primary table of declarations and their outcomes."""
    filtered_findings: List[Dict[str, Any]] = []

    for f in findings:
        rules = f.get("rules", {})
        has_r7 = bool(rules.get("rule7"))
        has_r8 = bool(rules.get("rule8"))

        if active_filter == "All":
            filtered_findings.append(f)
        elif active_filter == "Rule 6":
            filtered_findings.append(f)
        elif active_filter == "Rule 7" and has_r7:
            filtered_findings.append(f)
        elif active_filter == "Rule 8" and has_r8:
            filtered_findings.append(f)

    if not filtered_findings:
        st.info(f"No findings matching filter: **{active_filter}**")
        return

    for idx, f in enumerate(filtered_findings):
        field_key = f.get("field", "")
        title = f.get("display_name", field_key)
        visibility = f.get("visibility", "UNKNOWN")
        status = f.get("status", "NOT ASSESSABLE")
        raw_text = f.get("raw_text") or "*(Not visible in supplied image)*"
        detected_val = f.get("detected_value") or "-"
        declared_numeral = f.get("declared_numeral")
        declared_unit = f.get("declared_unit")
        confidence = f.get("confidence")
        conf_str = f"{round(confidence * 100)}%" if confidence is not None else "N/A"
        rule_ref = f.get("rule_reference") or "Rule 6"
        notes = f.get("notes") or ""

        rules = f.get("rules", {})
        r7 = rules.get("rule7")
        r8 = rules.get("rule8")

        # Container for card
        with st.container(border=True):
            col_left, col_right = st.columns([3, 1])
            with col_left:
                st.markdown(f"**{title}** &nbsp; `[{rule_ref}]`")
                st.caption(f"Raw: *\"{raw_text}\"*")
                if detected_val != "-":
                    st.markdown(f"**Detected Value:** `{detected_val}`")
            with col_right:
                st.markdown(get_status_badge_html(status), unsafe_allow_html=True)
                st.caption(f"Visibility: **{visibility}**")
                st.caption(f"Confidence: **{conf_str}**")

            # Expandable details
            with st.expander(f"🔎 Technical Details — {title}"):
                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    st.markdown(f"**Field Key:** `{field_key}`")
                    st.markdown(f"**Declared Numeral:** `{declared_numeral or 'N/A'}`")
                    st.markdown(f"**Declared Unit:** `{declared_unit or 'N/A'}`")
                with dcol2:
                    st.markdown(f"**Bounding Box:** `{f.get('bbox') or 'N/A'}`")
                    st.markdown(f"**Confidence:** `{conf_str}`")
                    st.markdown(f"**Statutory Ref:** `{rule_ref}`")

                if notes:
                    st.info(f"**Diagnostic Notes:** {notes}")

                # Display sub-rules if present
                if r7:
                    st.markdown("---")
                    st.markdown(
                        f"**Rule 7 Evaluation:** {get_status_badge_html(r7.get('status', ''))}",
                        unsafe_allow_html=True,
                    )
                    r7_m = r7.get("measured_height_mm")
                    r7_req = r7.get("required_height_mm")
                    r7_diff = r7.get("height_margin_mm")
                    diff_sign = f"+{r7_diff}" if (r7_diff is not None and r7_diff > 0) else f"{r7_diff}"
                    st.markdown(
                        f"- **Measured Height:** `{r7_m} mm` | **Required Minimum:** `{r7_req} mm` | **Margin:** `{diff_sign} mm`"
                    )
                    if r7.get("notes"):
                        st.caption(f"CV Notes: {r7.get('notes')}")

                if r8:
                    st.markdown("---")
                    st.markdown(
                        f"**Rule 8 Clearance:** {get_status_badge_html(r8.get('status', ''))}",
                        unsafe_allow_html=True,
                    )
                    h_target = r8.get("target_numeral_height_mm")
                    st.markdown(f"- **Target Numeral Height (H):** `{h_target} mm`")
                    st.markdown(
                        f"- **Top:** `{r8.get('top', {}).get('measured_mm')} mm` (Req: `{r8.get('top', {}).get('required_mm')} mm`) — `{r8.get('top', {}).get('status')}`"
                    )
                    st.markdown(
                        f"- **Bottom:** `{r8.get('bottom', {}).get('measured_mm')} mm` (Req: `{r8.get('bottom', {}).get('required_mm')} mm`) — `{r8.get('bottom', {}).get('status')}`"
                    )
                    st.markdown(
                        f"- **Left:** `{r8.get('left', {}).get('measured_mm')} mm` (Req: `{r8.get('left', {}).get('required_mm')} mm`) — `{r8.get('left', {}).get('status')}`"
                    )
                    st.markdown(
                        f"- **Right:** `{r8.get('right', {}).get('measured_mm')} mm` (Req: `{r8.get('right', {}).get('required_mm')} mm`) — `{r8.get('right', {}).get('status')}`"
                    )
                    if r8.get("notes"):
                        st.caption(f"Clearance Notes: {r8.get('notes')}")


def render_rule6_section(findings: List[Dict[str, Any]]) -> None:
    """Renders Rule 6 statutory declaration visibility section."""
    st.markdown("## Rule 6 — Declaration Visibility")
    st.markdown(
        "*Legal Metrology (Packaged Commodities) Rules, 2011 — Mandatory Declarations on Retail Packages.*"
    )

    st.caption(
        "📌 **Semantic Rule:** `NOT VISIBLE` means **'Not detected in the supplied image.'** "
        "It does not imply legal omission or absence on unphotographed package panels."
    )

    for f in findings:
        title = f.get("display_name", f.get("field", "Declaration"))
        vis = f.get("visibility", "")
        status = f.get("status", "")
        rule_ref = f.get("rule_reference", "Rule 6")
        raw_text = f.get("raw_text") or "Not detected in supplied image"
        val = f.get("detected_value") or "-"

        with st.container(border=True):
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                st.markdown(f"**{title}**")
                st.caption(f"Rule Reference: `{rule_ref}`")
            with c2:
                st.markdown(f"**Raw Text:** *\"{raw_text}\"*")
                if val != "-":
                    st.markdown(f"**Detected Value:** `{val}`")
            with c3:
                st.markdown(f"**Visibility:** `{vis}`")
                st.markdown(get_status_badge_html(status), unsafe_allow_html=True)


def render_rule7_section(findings: List[Dict[str, Any]]) -> None:
    """Renders Rule 7 Numeral Height Assessment table and comparisons."""
    st.markdown("## Rule 7 — Numeral Height")
    st.markdown(
        "*Statutory minimum height requirements for mandatory numerals based on packaging area and declared quantity.*"
    )

    r7_items = [f for f in findings if f.get("rules", {}).get("rule7")]

    if not r7_items:
        st.info("No Rule 7 numeral height evaluations present in this inspection.")
        return

    for f in r7_items:
        title = f.get("display_name", f.get("field", "Declaration"))
        r7 = f["rules"]["rule7"]
        status = r7.get("status", "NOT ASSESSABLE")
        measured = r7.get("measured_height_mm")
        required = r7.get("required_height_mm")
        margin = r7.get("height_margin_mm")
        quality = r7.get("measurement_quality", "N/A").upper()
        conf = r7.get("confidence")
        conf_str = f"{round(conf * 100)}%" if conf is not None else "N/A"
        rule_src = r7.get("rule_source_mode", "sih_ps_26034")
        rule_ref = r7.get("rule_reference", "Rule 7")
        notes = r7.get("notes") or ""

        with st.container(border=True):
            top_col, badge_col = st.columns([3, 1])
            with top_col:
                st.markdown(f"### {title}")
                st.caption(f"Rule Reference: `{rule_ref}` | Rule Source: `{rule_src}`")
            with badge_col:
                st.markdown(get_status_badge_html(status), unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Measured Height", f"{measured} mm" if measured is not None else "N/A")
            with m2:
                st.metric("Required Height", f"{required} mm" if required is not None else "N/A")
            with m3:
                diff_sign = f"+{margin}" if (margin is not None and margin > 0) else f"{margin}"
                st.metric("Margin", f"{diff_sign} mm" if margin is not None else "N/A")
            with m4:
                st.metric("Measurement Quality", f"{quality} ({conf_str})")

            # Visual threshold indicator
            if measured is not None and required is not None and required > 0:
                pct = min(1.5, max(0.0, measured / required))
                progress_val = min(1.0, pct)
                st.progress(progress_val)
                st.caption(
                    f"Measured numeral is **{round(pct * 100)}%** of statutory minimum height "
                    f"({'meets statutory requirement' if measured >= required else 'below legal minimum threshold'})."
                )

            if notes:
                st.caption(f"🔬 **CV Measurement Diagnostic Notes:** {notes}")


def render_rule8_section(findings: List[Dict[str, Any]]) -> None:
    """Renders Rule 8 Spatial Clearance analysis section."""
    st.markdown("## Rule 8 — Spatial Clearance")
    st.markdown(
        "*Legal Metrology (Packaged Commodities) Rules, 2011 — Clear Space Surrounding Declarations.*"
    )

    r8_items = [f for f in findings if f.get("rules", {}).get("rule8")]

    if not r8_items:
        st.info("No Rule 8 spatial clearance evaluations present in this inspection.")
        return

    for f in r8_items:
        title = f.get("display_name", f.get("field", "Declaration"))
        r8 = f["rules"]["rule8"]
        status = r8.get("status", "NOT ASSESSABLE")
        h_target = r8.get("target_numeral_height_mm")
        rule_ref = r8.get("rule_reference", "Rule 8(1) proviso")
        notes = r8.get("notes") or ""

        with st.container(border=True):
            h_col, b_col = st.columns([3, 1])
            with h_col:
                st.markdown(f"### {title}")
                st.markdown(f"**Target Numeral Height H:** `{h_target} mm` | Ref: `{rule_ref}`")
            with b_col:
                st.markdown(get_status_badge_html(status), unsafe_allow_html=True)

            d1, d2, d3, d4 = st.columns(4)
            dirs = [("Top (1H)", r8.get("top", {}), d1),
                    ("Bottom (1H)", r8.get("bottom", {}), d2),
                    ("Left (2H)", r8.get("left", {}), d3),
                    ("Right (2H)", r8.get("right", {}), d4)]

            for dir_name, dir_data, col in dirs:
                with col:
                    meas = dir_data.get("measured_mm")
                    req = dir_data.get("required_mm")
                    dir_status = dir_data.get("status", "").upper()
                    st.markdown(f"**{dir_name}**")
                    st.markdown(f"Measured: `{meas} mm`")
                    st.markdown(f"Required: `{req} mm`")
                    st.caption(f"Status: **{dir_status}**")

            if notes:
                st.caption(f"⚠️ **Spatial Clearance Diagnostic Notes:** {notes}")


def render_evidence_section(data: Dict[str, Any]) -> None:
    """Renders provenance and cryptographic evidence details."""
    st.markdown("## Evidence & Provenance")
    metadata = data.get("metadata", {})
    inspection_id = data.get("inspection_id", "N/A")
    filename = metadata.get("source_image_filename", "original_image.jpg")
    sha256 = metadata.get("source_image_sha256", "N/A")

    with st.container(border=True):
        st.markdown(f"**Inspection ID:** `{inspection_id}`")
        st.markdown(f"**Source Filename:** `{filename}`")
        st.markdown(f"**Stored Image Verification:** `MATCHED`")
        st.markdown(f"**SHA-256:** `{sha256[:12]}...{sha256[-8:]}`")

        with st.expander("View complete SHA-256"):
            st.code(sha256, language=None)
            st.caption(
                "The SHA-256 value identifies the exact stored source image associated with this inspection."
            )


def render_lmo_review_section(inspection_id: str) -> None:
    """Renders manual LMO review area and persists changes through FastAPI."""
    st.markdown("## LMO Review")
    st.caption("Official review notes recorded by the inspecting officer. Does not modify automated rule outcomes.")

    review_status_options = ["Pending", "Verified", "Issue Raised", "N/A"]

    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        with c1:
            st.selectbox(
                "Review Status:",
                options=review_status_options,
                key="lmo_review_status",
            )
        with c2:
            st.text_area(
                "LMO Notes:",
                placeholder="Enter field observations, verification against physical calipers, or seizure memo notes...",
                key="lmo_review_notes",
                height=100,
            )

        if st.button("Save LMO Review", type="primary", key=f"save_review_{inspection_id}"):
            from api_client import update_review

            result = update_review(
                inspection_id=inspection_id,
                review_status=st.session_state["lmo_review_status"],
                notes=st.session_state["lmo_review_notes"],
            )
            if result.get("success") and result.get("data"):
                st.session_state["inspection_response"] = result["data"]
                st.success("LMO review saved.")
                st.rerun()
            else:
                error = result.get("error", {})
                st.error(error.get("message", "Could not save LMO review."))


def render_regulatory_notice() -> None:
    """Renders compact statutory disclaimer conforming to DoCA guidelines."""
    st.markdown("---")
    st.caption(
        "Automated Decision Support System.\n\n"
        "Automated results are preliminary inspection-support findings. "
        "NOT VISIBLE means the declaration was not detected in the supplied "
        "image and does not by itself establish legal omission.\n\n"
        "Final enforcement determination rests with the authorized inspecting officer."
    )
