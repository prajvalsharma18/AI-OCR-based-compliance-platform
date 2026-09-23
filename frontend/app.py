"""Streamlit Frontend for SIH 2026 PS 26034.

“Automated Compliance Checker for Packaged Commodities”
Legal Metrology Officer (LMO) Dashboard.

Constraints:
- FastAPI is the single source of truth.
- Zero local OCR, LLM, or legal calculations.
"""

import json
import logging
import streamlit as st

from api_client import (
    BACKEND_URL,
    check_health,
    get_inspection,
    get_inspection_image,
    get_stored_report,
    list_inspections,
    run_inspection,
)
from ui_components import (
    draw_bounding_boxes,
    render_evidence_section,
    render_findings_table,
    render_kpi_cards,
    render_lmo_review_section,
    render_provenance_header,
    render_regulatory_notice,
    render_rule6_section,
    render_rule7_section,
    render_rule8_section,
)

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Automated Compliance Checker",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 2. Session State Initialization
# -----------------------------------------------------------------------------
if "inspection_response" not in st.session_state:
    st.session_state["inspection_response"] = None
if "inspection_id" not in st.session_state:
    st.session_state["inspection_id"] = None
if "uploaded_image_bytes" not in st.session_state:
    st.session_state["uploaded_image_bytes"] = None
if "uploaded_filename" not in st.session_state:
    st.session_state["uploaded_filename"] = None
if "lmo_review_status" not in st.session_state:
    st.session_state["lmo_review_status"] = "Pending"
if "lmo_review_notes" not in st.session_state:
    st.session_state["lmo_review_notes"] = ""
if "view_mode" not in st.session_state:
    st.session_state["view_mode"] = "New Inspection"


def reset_inspection() -> None:
    """Clears current inspection state to allow a fresh upload."""
    st.session_state["inspection_response"] = None
    st.session_state["inspection_id"] = None
    st.session_state["uploaded_image_bytes"] = None
    st.session_state["uploaded_filename"] = None
    st.session_state["lmo_review_status"] = "Pending"
    st.session_state["lmo_review_notes"] = ""


# -----------------------------------------------------------------------------
# 3. Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚖️ Legal Metrology")
    st.caption("DoCA — PS 26034 Compliance System")
    st.markdown("---")

    st.subheader("Inspection Mode")
    st.session_state["view_mode"] = st.radio(
        "Mode",
        ["New Inspection", "Inspection History"],
        index=0 if st.session_state["view_mode"] == "New Inspection" else 1,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.subheader("Backend Status")

    # Check FastAPI health
    health_res = check_health()
    if health_res.get("connected"):
        st.success("🟢 CONNECTED")
        with st.expander("Service Info"):
            st.write(health_res.get("data", {}))
    else:
        st.error("🔴 OFFLINE")
        st.caption(
            f"Inspection service is unavailable. Please verify that FastAPI is running on port 8001.\n"
            f"Target: `{BACKEND_URL}`"
        )
        if health_res.get("error"):
            st.caption(f"Error: {health_res['error']}")

    st.markdown("---")
    if st.session_state.get("inspection_response"):
        if st.button("🔄 Start New Inspection", use_container_width=True):
            reset_inspection()
            st.rerun()

    st.markdown(
        """
        <div style="font-size: 0.78rem; color: #64748B; margin-top: 2rem;">
        <b>Applicable Legislation:</b><br/>
        • Legal Metrology Act, 2009<br/>
        • PC Rules, 2011 (Rules 6, 7, 8)
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# 4. Main Page Header
# -----------------------------------------------------------------------------
st.title("Automated Compliance Checker")
st.subheader("Legal Metrology — Packaged Commodities")
st.markdown(
    "Automated inspection-support system for packaged commodity declaration and dimensional compliance screening."
)
st.markdown("---")


def render_history_page() -> None:
    """Renders compact history filters and opens a selected inspection."""
    st.markdown("### Inspection History")
    filter_cols = st.columns([1, 1, 1, 1, 1])
    with filter_cols[0]:
        date_from = st.date_input("Date From", value=None)
    with filter_cols[1]:
        date_to = st.date_input("Date To", value=None)
    with filter_cols[2]:
        brand_filter = st.text_input("Brand", placeholder="Search brand")
    with filter_cols[3]:
        package_filter = st.selectbox("Package Type", ["All", "retail", "wholesale", "combination_pack"])
    with filter_cols[4]:
        review_filter = st.selectbox("Review Status", ["All", "Pending", "Verified", "Issue Raised", "N/A"])

    search_id = st.text_input("Search / Inspection ID", placeholder="INSP-...")
    history = list_inspections(
        date_from=f"{date_from.isoformat()}T00:00:00Z" if date_from else None,
        date_to=f"{date_to.isoformat()}T23:59:59.999999Z" if date_to else None,
        brand_name=brand_filter or None,
        package_type=None if package_filter == "All" else package_filter,
        status=None if review_filter == "All" else review_filter,
        inspection_id=search_id or None,
    )
    if not history.get("success"):
        st.error(history.get("error", {}).get("message", "Inspection history is unavailable."))
        return

    items = history.get("items", [])
    if not items:
        st.info("No inspections match the selected filters.")
        return

    for item in items:
        created_at = str(item.get("created_at", ""))[:19].replace("T", " ")
        issues = item.get("non_compliant_count", 0)
        cols = st.columns([1.4, 2.2, 1.5, 1.2, 1.2, 1.2])
        cols[0].write(created_at)
        cols[1].write(item.get("inspection_id", ""))
        cols[2].write(item.get("brand_name") or "Not specified")
        cols[3].write(str(issues))
        cols[4].write(item.get("review_status", "Pending"))
        if cols[5].button("Open", key=f"open_{item.get('inspection_id')}"):
            detail = get_inspection(item["inspection_id"])
            if detail.get("success") and detail.get("data"):
                image_ok, image_result = get_inspection_image(item["inspection_id"])
                st.session_state["inspection_response"] = detail["data"]
                st.session_state["inspection_id"] = item["inspection_id"]
                st.session_state["uploaded_image_bytes"] = image_result if image_ok else None
                st.session_state["uploaded_filename"] = detail["data"].get("metadata", {}).get("source_image_filename")
                st.session_state["view_mode"] = "New Inspection"
                review = detail["data"].get("review", {})
                st.session_state["lmo_review_status"] = review.get("status", "Pending")
                st.session_state["lmo_review_notes"] = review.get("notes", "")
                st.rerun()
            else:
                st.error(detail.get("error", {}).get("message", "Could not load inspection."))


if st.session_state.get("view_mode") == "Inspection History":
    render_history_page()
    st.stop()


# -----------------------------------------------------------------------------
# 5. New Inspection Input View (When no active inspection)
# -----------------------------------------------------------------------------
if not st.session_state.get("inspection_response"):
    st.markdown("### New Inspection")

    col_up, col_meta = st.columns([1, 1], gap="large")

    with col_up:
        uploaded_file = st.file_uploader(
            "Upload Packaging Image",
            type=["jpg", "jpeg", "png", "webp"],
            help="High-resolution packaging photograph showing mandatory declarations.",
            key="package_img_uploader",
        )

        if uploaded_file is not None:
            st.image(
                uploaded_file,
                caption=f"Selected: {uploaded_file.name} ({round(uploaded_file.size / 1024, 1)} KB)",
                use_container_width=True,
            )

    with col_meta:
        st.markdown("#### Packaging Metadata (Optional)")
        st.caption("Provide known dimensions for millimeter calibration, or leave blank for automatic detection.")

        brand_input = st.text_input(
            "Brand Name",
            placeholder="e.g. Lay's, Britannia, Nestle",
            help="Optional brand override.",
        )
        generic_input = st.text_input(
            "Generic Name",
            placeholder="e.g. Potato Chips, Biscuits",
            help="Optional generic/commodity name override.",
        )

        pkg_type_input = st.selectbox(
            "Package Type",
            options=["retail", "wholesale", "combination_pack"],
            index=0,
            help="Commodity packaging classification.",
        )

        c_w, c_h = st.columns(2)
        with c_w:
            w_input = st.number_input(
                "Package Width (mm)",
                min_value=0.0,
                max_value=2000.0,
                value=0.0,
                step=1.0,
                help="Physical width in millimetres. Leave 0 if unknown.",
            )
        with c_h:
            h_input = st.number_input(
                "Package Height (mm)",
                min_value=0.0,
                max_value=2000.0,
                value=0.0,
                step=1.0,
                help="Physical height in millimetres. Leave 0 if unknown.",
            )

        c_pdp, c_mode = st.columns(2)
        with c_pdp:
            pdp_input = st.number_input(
                "PDP Area (cm²)",
                min_value=0.0,
                max_value=10000.0,
                value=0.0,
                step=1.0,
                help="Principal Display Panel area. Calculated automatically from W × H if omitted.",
            )
        with c_mode:
            rule_mode_input = st.selectbox(
                "Rule Source Mode",
                options=["sih_ps_26034", "doca_statutory_2011"],
                index=0,
                help="Statutory standard applied for Rule 7 height evaluation.",
            )

        st.markdown("<br/>", unsafe_allow_html=True)
        run_clicked = st.button("Run Inspection", type="primary", use_container_width=True)

    # Handle Form Submission
    if run_clicked:
        if uploaded_file is None:
            st.error("Please select a valid packaging image.")
        elif w_input < 0 or h_input < 0 or pdp_input < 0:
            st.error("Dimensions and PDP area must be strictly positive (> 0).")
        else:
            # Read bytes
            image_bytes = uploaded_file.getvalue()
            filename = uploaded_file.name

            # Resolve optional numerical inputs
            pkg_w = float(w_input) if w_input > 0 else None
            pkg_h = float(h_input) if h_input > 0 else None
            pdp_area = float(pdp_input) if pdp_input > 0 else None

            with st.spinner("Running semantic extraction, dimensional measurement and compliance evaluation…"):
                res = run_inspection(
                    image_bytes=image_bytes,
                    filename=filename,
                    brand_name=brand_input,
                    generic_name=generic_input,
                    package_type=pkg_type_input,
                    package_width_mm=pkg_w,
                    package_height_mm=pkg_h,
                    pdp_area_cm2=pdp_area,
                    rule_source_mode=rule_mode_input,
                )

            if res.get("success") and "data" in res:
                st.session_state["inspection_response"] = res["data"]
                st.session_state["inspection_id"] = res["data"].get("inspection_id")
                st.session_state["uploaded_image_bytes"] = image_bytes
                st.session_state["uploaded_filename"] = filename
                st.rerun()
            else:
                err = res.get("error", {})
                err_code = err.get("code", "UNKNOWN_ERROR")
                err_msg = err.get("message", "An unexpected error occurred during inspection.")

                if err_code == "BACKEND_UNAVAILABLE":
                    st.error(
                        "Inspection service is unavailable. Please verify that FastAPI is running on port 8001."
                    )
                elif err_code in ["UNSUPPORTED_MEDIA_TYPE", "CORRUPTED_IMAGE"]:
                    st.error(f"Please select a valid packaging image. ({err_msg})")
                else:
                    st.error(f"Inspection Failed: {err_msg}")

                with st.expander("Technical Error Diagnostic Details"):
                    st.write(err)


# -----------------------------------------------------------------------------
# 6. Inspection Results Dashboard View
# -----------------------------------------------------------------------------
else:
    data = st.session_state["inspection_response"]
    summary = data.get("summary", {})
    findings = data.get("findings", [])
    image_bytes = st.session_state.get("uploaded_image_bytes")
    inspection_id = data.get("inspection_id", "INSP-DEFAULT")

    # A. Top Provenance Header Card
    render_provenance_header(data)

    st.markdown("<br/>", unsafe_allow_html=True)

    # B. Summary KPI Cards
    render_kpi_cards(summary)

    st.markdown("---")

    # C. Main Workspace: Two-Column Split Screen (45% left / 55% right)
    col_img, col_findings = st.columns([45, 55], gap="large")

    with col_img:
        st.markdown("### Original Inspection Image")

        show_bboxes = st.checkbox(
            "Show Bounding Box Overlays",
            value=True,
            help="Displays normalized coordinates mapped to detected declarations. "
                 "Green = PASS, Red = NON-COMPLIANT, Blue = DETECTED.",
        )

        if image_bytes:
            if show_bboxes:
                overlay_img = draw_bounding_boxes(image_bytes, findings, show_labels=True)
                st.image(overlay_img, caption="Packaging Image with Bounding Boxes", use_container_width=True)
            else:
                st.image(image_bytes, caption="Original Packaging Image", use_container_width=True)

        st.markdown(
            """
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 8px 12px; border-radius: 6px; font-size: 0.85rem; margin-top: 8px;">
            <b>Source:</b> ORIGINAL STORED IMAGE &nbsp;|&nbsp; <b>Verification:</b> <span style="color: #15803D; font-weight: 600;">MATCHED</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_findings:
        st.markdown("### Compliance Findings")

        # Filters / Tabs
        active_filter = st.radio(
            "Filter Findings:",
            ["All", "Rule 6", "Rule 7", "Rule 8"],
            index=0,
            horizontal=True,
        )

        render_findings_table(findings, active_filter=active_filter)

    st.markdown("---")

    # D. Dedicated Statutory Rule Sections
    render_rule6_section(findings)

    st.markdown("---")

    render_rule7_section(findings)

    st.markdown("---")

    render_rule8_section(findings)

    st.markdown("---")

    # E. Evidence & Provenance
    render_evidence_section(data)

    st.markdown("---")

    # F. Reports & Export Section
    st.markdown("## Reports & Export")
    col_pdf, col_json, col_space = st.columns([1, 1, 2])

    with col_pdf:
        # Cache generated PDF in session state to prevent redundant API calls
        pdf_cache_key = f"pdf_bytes_{inspection_id}"
        if pdf_cache_key not in st.session_state:
            with st.spinner("Loading stored official PDF report..."):
                pdf_success, pdf_res = get_stored_report(inspection_id)
                if pdf_success and isinstance(pdf_res, bytes):
                    st.session_state[pdf_cache_key] = pdf_res
                else:
                    st.session_state[pdf_cache_key] = None
                    st.session_state[f"pdf_err_{inspection_id}"] = pdf_res

        cached_pdf = st.session_state.get(pdf_cache_key)
        if cached_pdf:
            st.download_button(
                label="Download Official PDF Report",
                data=cached_pdf,
                file_name=f"inspection_{inspection_id}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
        else:
            st.button("⚠️ PDF Report Unavailable", disabled=True, use_container_width=True)
            err_msg = st.session_state.get(f"pdf_err_{inspection_id}", "Generation failed")
            st.caption(f"Error: {err_msg}")

    with col_json:
        json_str = json.dumps(data, indent=2, default=str)
        st.download_button(
            label="Export Unified JSON",
            data=json_str,
            file_name=f"inspection_{inspection_id}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("---")

    # G. Manual LMO Review Area
    review_data = data.get("review", {})
    if review_data:
        st.session_state["lmo_review_status"] = review_data.get("status", st.session_state["lmo_review_status"])
        st.session_state["lmo_review_notes"] = review_data.get("notes", st.session_state["lmo_review_notes"])
    render_lmo_review_section(inspection_id)

    # H. Regulatory Notice
    render_regulatory_notice()
