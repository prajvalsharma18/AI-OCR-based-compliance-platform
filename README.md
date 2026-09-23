# Automated Compliance Checker for Packaged Commodities

**SIH 2026 — Problem Statement 26034**  
**Legal Metrology (Packaged Commodities) Rules, 2011**

An end-to-end inspection-support system that accepts a packaged-product image, extracts regulatory declarations, measures declaration numerals, evaluates Rule 6 / Rule 7 / Rule 8 conditions, preserves original-image provenance, and generates a structured compliance inspection report.

> **Project status:** Backend inspection pipeline complete and integrated. The single-entry inspection API is implemented and verified. Streamlit frontend is the next development stage.

---

## 1. Problem Statement

The system is designed around SIH 2026 PS 26034: **Automated Compliance Checker for Packaged Commodities**.

The source specification describes a workflow in which a package image is captured, declarations are extracted, layout and font-size information are analyzed, legal rules are evaluated, evidence is attached, and an inspection report is produced.

The implemented system currently covers the core automated inspection path:

```text
Package Image
     │
     ▼
Image Storage + SHA-256 Provenance
     │
     ▼
Semantic Extraction
     │
     ▼
Package / Numeral Measurement
     │
     ▼
Rule 6 Visibility
     │
     ▼
Rule 7 Numeral-Height Evaluation
     │
     ▼
Rule 8 Clearance Evaluation
     │
     ▼
Unified Inspection JSON
     │
     ▼
Inspection Persistence
     │
     ▼
PDF Inspection Report
```

The project deliberately separates deterministic compliance logic from LLM-based semantic extraction.

---

## 2. Current Technology Stack

### Backend

- **Python 3.14**
- **FastAPI**
- **Uvicorn**
- **Pydantic v2**
- **OpenAI API / Vision** for semantic extraction
- **OpenCV** for image processing and measurement support
- **NumPy**
- **Pillow**
- **python-multipart** for file uploads
- **ReportLab** for PDF report generation
- **pypdf / pypdfium2** for report validation and rendering checks

### Testing

- **pytest**
- **pytest-asyncio**
- **httpx**

### Planned Frontend

- **Streamlit**
- Frontend calls the FastAPI backend; no compliance logic is duplicated in the UI.

---

## 3. Project Architecture

```text
ai_ocr_pipeline/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── compliance.py
│   │   │   ├── inspection.py
│   │   │   └── report.py
│   │   │
│   │   ├── schemas/
│   │   │   ├── compliance.py
│   │   │   └── inspection.py
│   │   │
│   │   ├── services/
│   │   │   ├── extraction_service.py
│   │   │   ├── measurement_service.py
│   │   │   ├── package_detector.py
│   │   │   ├── rule6_service.py
│   │   │   ├── rule7_service.py
│   │   │   ├── rule8_service.py
│   │   │   ├── inspection_service.py
│   │   │   ├── image_storage_service.py
│   │   │   └── report_service.py
│   │   │
│   │   ├── main.py
│   │   └── ...
│   │
│   ├── tests/
│   │   ├── test_extraction_*.py
│   │   ├── test_measurement_*.py
│   │   ├── test_rule6.py
│   │   ├── test_rule7.py
│   │   ├── test_rule8.py
│   │   ├── test_inspection_endpoint.py
│   │   ├── test_report.py
│   │   └── ...
│   │
│   ├── reports/
│   │   └── inspections/
│   │       └── <inspection_id>/
│   │           ├── original_image.*
│   │           ├── inspection.json
│   │           └── report.pdf
│   │
│   └── venv/
│
└── frontend/              # Streamlit frontend — next stage
```

---

# 4. Implemented Modules

## 4.1 Semantic Extraction

The extraction layer uses OpenAI Vision to identify package declarations and return normalized structured data.

Key capabilities:

- Raw text extraction
- Bounding boxes normalized to `0–1`
- Confidence values
- Numeral-region extraction
- Individual consumer-care subfields
- Declaration classification
- Date semantic classification
- Filtering of unrelated numeric information

The extraction layer distinguishes regulatory numerals from unrelated numbers such as:

- Phone numbers
- PIN codes
- License / registration numbers
- FSSAI identifiers
- Lot / batch references
- Barcodes
- Survey/reference numbers
- Address numbers

Those values can be retained as additional numeric information without being incorrectly treated as Rule 7 declaration numerals.

### Date handling

Combined declarations such as:

```text
MFD & USE BY: 26/07/26 & 08/12/26
```

are normalized into separate semantic entries:

```text
manufacture
use_by
```

The implementation also supports:

- `mfd`
- `mfg`
- `expiry`
- `best_before`
- `use-by`
- `exp`

Date mapping is semantic rather than positional, preventing manufacture and use-by dates from being swapped when the OCR order changes.

---

## 4.2 Package and Numeral Measurement

The measurement layer converts pixel measurements into physical dimensions using calibration information.

Supported calibration approaches include:

- User-provided physical package dimensions
- Existing package-boundary detection
- Other project-supported calibration references

When a package bounding box is not supplied, the existing measurement service can automatically detect the package boundary using the package detector.

### Example calibration

For the current Lay's sample setup:

```text
Package Width   = 150 mm
Package Height  = 200 mm
PDP Area        = 300 cm²
```

The service retains:

- pixel bounding boxes
- normalized bounding boxes
- pixel height
- physical height in mm
- measurement quality
- confidence

### Important design rule

The measurement service does **not** determine legal compliance. It only produces measurement evidence used by Rule 7 and Rule 8.

---

# 5. Rule 6 — Declaration Visibility

Rule 6 is implemented as deterministic declaration mapping and visibility analysis.

The current declaration model covers:

| Rule | Declaration |
|---|---|
| Rule 6(1)(a) | Manufacturer / Packer / Importer |
| Rule 6(1)(b) | Generic Name |
| Rule 6(1)(c) | Retail Sale Price (MRP) |
| Rule 6(1)(d) | Date of Manufacture / Packing / Import |
| Rule 6(1)(e) | Net Quantity |
| Rule 6(1)(f) | Consumer Care Details |

### Visibility semantics

```text
DETECTED
NOT VISIBLE
NOT APPLICABLE
```

`NOT VISIBLE` means:

> The declaration was not detected in the supplied image.

It does **not** mean that the declaration is legally absent from the entire package.

This distinction is important because the current automated pipeline can inspect only the supplied package face/image.

---

# 6. Rule 7 — Numeral / Text Height Evaluation

Rule 7 is implemented as a deterministic rule engine.

The engine performs:

1. Unit normalization
2. Requirement selection
3. Applicable-field filtering
4. Measured-vs-required comparison
5. Margin calculation
6. Measurement-confidence propagation
7. Low-confidence / missing-measurement handling

No LLM is used for the final Rule 7 comparison.

## Rule source modes

The API supports two explicit modes:

```text
sih_ps_26034
```

and

```text
doca_statutory_2011
```

This allows the implementation to keep the SIH problem specification and the statutory-mode configuration distinct instead of silently mixing thresholds.

The SIH technical specification contains its own stated Rule 7 thresholds and describes font-size analysis as a core functional requirement.

The project therefore keeps the selected rule source visible in the API response and generated report.

### Example result

```text
Measured Height : 1.65 mm
Required Height : 2.00 mm
Margin          : -0.35 mm
Status          : NON-COMPLIANT
```

The frontend/report consumes this backend result; it does not recompute the legal threshold.

---

# 7. Rule 8 — Spatial Clearance Evaluation

Rule 8 is implemented as a deterministic spatial-clearance engine.

For a numeral declaration with height `H`, the engine evaluates the required blank space around the complete declaration.

The implementation evaluates:

```text
Top    = H
Bottom = H
Left   = 2H
Right  = 2H
```

The legal reference used in the report is the **Rule 8(1) proviso**.

### Measurement design

The engine uses the complete declaration bounding box rather than only the numeral glyph box.

The pipeline combines:

- semantic extraction boxes
- declaration layout boxes
- local computer-vision evidence
- package boundary information where verified
- directional clearance searches

Low-quality or missing measurements are propagated as indeterminate states rather than being silently treated as compliant.

### Example result

```text
Top    : 0.24 / 1.67 mm  → below threshold
Bottom : 0.00 / 1.67 mm  → below threshold
Left   : 6.84 / 3.34 mm  → above threshold
Right  : 7.02 / 3.34 mm  → above threshold
```

The overall declaration status becomes `NON-COMPLIANT` when an applicable Rule 8 requirement fails.

---

# 8. Unified Inspection Model

The project consolidates Rule 6, Rule 7, and Rule 8 outputs into one inspection response.

Primary response structure:

```json
{
  "success": true,
  "data": {
    "inspection_id": "...",
    "metadata": {},
    "summary": {},
    "findings": [],
    "disclaimer": "..."
  }
}
```

## Metadata

The unified model can contain:

- inspection ID
- brand name
- generic name
- package type
- image quality
- package dimensions
- PDP area
- rule source mode
- original filename
- original image SHA-256
- source storage path

## Finding fields

Each declaration finding can contain:

- field
- display name
- visibility
- raw text
- detected value
- declared numeral
- declared unit
- confidence
- bounding box
- Rule 6 reference
- Rule 7 result
- Rule 8 result
- notes

## Status model

Supported unified statuses:

```text
PASS
NON-COMPLIANT
NOT VISIBLE
NOT ASSESSABLE
NOT APPLICABLE
```

Important semantics:

- `DETECTED` is **not** automatically `PASS`.
- A detected declaration with no applicable measurement/rule assessment becomes `NOT ASSESSABLE`.
- For applicable rule results, the primary status precedence is:

```text
NON-COMPLIANT
      ↓
NOT ASSESSABLE
      ↓
PASS
```

- `NOT VISIBLE` remains a separate visibility state.

---

# 9. End-to-End Inspection API

The project now exposes a single entry point for the complete inspection workflow:

```http
POST /api/v1/inspection
Content-Type: multipart/form-data
```

## Required field

```text
image
```

## Optional fields

```text
brand_name
 generic_name
 package_type
 package_width_mm
 package_height_mm
 pdp_area_cm2
 rule_source_mode
```

### Allowed package types

```text
retail
wholesale
combination_pack
```

### Allowed rule modes

```text
sih_ps_26034
doca_statutory_2011
```

## Processing sequence

```text
Upload image
   ↓
Generate inspection_id
   ↓
Store exact original image
   ↓
Compute SHA-256
   ↓
Semantic extraction
   ↓
Measurement
   ↓
Rule 6
   ↓
Rule 7
   ↓
Rule 8
   ↓
Unified Inspection JSON
   ↓
Persist inspection.json
```

The endpoint is intentionally the orchestration layer. Individual rule modules remain reusable and independently testable.

---

# 10. Image Storage and Provenance

Original-image provenance is built into the inspection workflow.

For every inspection, the project stores the exact uploaded image bytes without PIL re-encoding or arbitrary recompression.

Storage structure:

```text
reports/
└── inspections/
    └── <inspection_id>/
        ├── original_image.<ext>
        ├── inspection.json
        └── report.pdf
```

The inspection metadata records:

- `source_image_sha256`
- `source_image_filename`
- `source_image_path`
- `source_image_storage_path`

The report layer prefers the stored original image over arbitrary client-supplied image data.

If a source-image hash mismatch is detected, report generation fails instead of silently using a different image.

### Verified sample

Current Lay's end-to-end sample:

```text
Inspection ID:
INSP-1790182901528

SHA-256:
ca8a36b6ef4c74493f861937dbfecef8905f3bc4eba0e0230a0cc900314f2e7e
```

The stored original image was verified byte-for-byte against the uploaded source.

---

# 11. PDF Inspection Report

The project includes a structured PDF inspection report generated with ReportLab.

The report contains:

### Page 1

- Inspection header
- Package / inspection metadata
- Rule 6 declaration visibility
- Rule 7 numeral-height results
- Rule 8 clearance results

### Page 2

- Inspection summary counts
- Detailed declaration findings
- Evidence-oriented rule results
- Confidence / bounding-box information

### Page 3

- Original source package image
- Source-image provenance
- LMO review notes
- Legal/automation disclaimer

The report includes a running footer with:

```text
Inspection ID
Page X of Y
```

The report layer does not recalculate legal statuses. It renders the already evaluated inspection data.

---

# 12. LMO Review and Evidence Model

The automated result is intentionally separated from human review.

The system distinguishes:

### Automated findings

Generated by the inspection pipeline:

```text
PASS
NON-COMPLIANT
NOT VISIBLE
NOT ASSESSABLE
```

### LMO review

A later UI layer can provide manual review states such as:

```text
Pending
Verified
Issue Raised
N/A
```

Manual review must not overwrite the automated finding status.

This keeps machine-generated evidence and authorized officer decisions separate.

---

# 13. Sample Inspection Result

A representative Lay's package inspection currently contains seven declaration findings:

```text
Manufacturer / Packer / Importer → DETECTED
Generic Name                    → DETECTED
MRP                             → DETECTED
Manufacture Date                → DETECTED
Use By / Best Before            → DETECTED
Net Quantity                    → DETECTED
Consumer Care                   → NOT VISIBLE
```

A sample report demonstrates combined Rule 7 and Rule 8 evaluation for Net Quantity and other measurable declarations. The project also maintains separate compliant and non-compliant Rule 8 fixtures for validation.

---

# 14. Test Coverage

The backend has a broad automated test suite covering:

- semantic extraction
- date normalization
- numeric-field filtering
- package detection
- physical measurement
- Rule 6 visibility mapping
- Rule 7 threshold selection and comparison
- Rule 8 spatial clearance
- unified inspection aggregation
- source-image storage
- SHA-256 provenance
- PDF generation
- PDF evidence rendering
- end-to-end inspection API
- error handling
- backward compatibility

## Latest full-suite result

```text
Collected : 221
Passed    : 220
Failed    : 0
Skipped   : 1
```

The single skipped test is a visual inspection confirmation associated with Rule 8.

The implementation is currently passing the complete automated backend suite with zero failures.

---

# 15. Running the Backend

From the backend directory:

```powershell
cd "C:\Users\Prajval Sharms\OneDrive\Desktop\Projects\web_Projects\ai_ocr_pipeline\backend"
```

Activate the virtual environment:

```powershell
.\venv\Scripts\Activate.ps1
```

Start FastAPI:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

Swagger documentation:

```text
http://127.0.0.1:8001/docs
```

Run tests:

```powershell
.\venv\Scripts\python -m pytest -v
```

---

# 16. Example End-to-End Request

Using Swagger:

```text
POST /api/v1/inspection
```

Example values:

```text
image              = <package image>
brand_name         = Lay's
generic_name       = Potato Chips
package_type       = retail
package_width_mm   = 150.0
package_height_mm  = 200.0
pdp_area_cm2       = 300.0
rule_source_mode   = sih_ps_26034
```

The backend returns the inspection ID and full unified finding set.

---

# 17. Design Principles

## 17.1 Deterministic legal evaluation

Rule 6, Rule 7 and Rule 8 are implemented as deterministic backend logic rather than relying on an LLM to make the final compliance decision.

## 17.2 LLM only where it adds value

OpenAI Vision is used for semantic interpretation of the package image. The legal comparison itself remains deterministic.

## 17.3 Evidence before verdict

Measurements, confidence, bounding boxes, source-image hashes, and rule references are retained so that a result can be reviewed rather than treated as an unexplained model output.

## 17.4 Do not overclaim missing information

A declaration that is not detected in the supplied image is represented as `NOT VISIBLE`, not automatically as legally missing.

## 17.5 Backend is the source of truth

The future frontend consumes the Unified Inspection JSON and does not independently calculate compliance rules.

## 17.6 Human review remains separate

The system is an automated inspection-support tool. Final enforcement decisions remain with authorized Legal Metrology personnel.

---

# 18. Known Scope and Limitations

### Single supplied image / package face

The current inspection operates on the image supplied by the user. A declaration printed on another package panel can therefore be `NOT VISIBLE` without implying that it is absent from the complete package.

### PDP identification

The current pipeline has package-boundary and measurement support, but full production-grade automatic principal-display-panel determination across arbitrary package geometries remains an area for further improvement.

### Image quality and calibration

Accurate physical font-size measurement depends on sufficient image quality and valid scale calibration.

### Food / category-specific legal nuances

The SIH specification includes special cases and exemptions. The automated pipeline should not silently treat the simplified SIH description as the complete current statutory text.

### Statutory verification

The SIH technical specification itself advises verification against the current official Department of Consumer Affairs rules because legal provisions can be amended. The repository therefore exposes an explicit statutory rule-source mode rather than pretending the SIH summary is permanently identical to the current law.

The source specification also identifies additional production concerns such as curved/reflective packaging, multilingual labels, PDP detection, combination packs, and low-connectivity field use.

---

# 19. SIH Functional Requirement Mapping

The SIH specification identifies functional requirements including image upload/product scanning, Rule 6 extraction, font-size analysis, non-standard declaration detection, report generation, evidence attachment, inspection history, role-based access, dashboards, search/retrieval, and technical documentation.

Current implementation status:

| SIH Requirement | Current Status |
|---|---|
| Image upload / product scanning | ✅ Implemented |
| Rule 6 declaration extraction | ✅ Implemented |
| Rule 7 font/numeral analysis | ✅ Implemented |
| Rule 8 spacing analysis | ✅ Implemented |
| Detection of non-compliant declarations | ✅ Core automated checks implemented |
| PDF inspection report | ✅ Implemented |
| Original-image evidence | ✅ Implemented |
| Inspection JSON persistence | ✅ Implemented |
| Inspection history UI | ⏳ Planned |
| Search / retrieval UI | ⏳ Planned |
| Role-based authentication | ⏳ Planned |
| Enforcement dashboard | ⏳ Planned |
| Editable report format | ⏳ Planned |
| Production database | ⏳ Planned |
| Streamlit frontend | 🚧 Next stage |

This intentionally distinguishes what is already implemented from broader SIH requirements that have not yet been built.

---

# 20. Next Development Stage — Streamlit Frontend

The recommended next stage is a Streamlit application using the single inspection endpoint.

Target flow:

```text
Streamlit
   │
   │ POST /api/v1/inspection
   ▼
Unified Inspection JSON
   │
   ├── Inspection Metadata
   ├── Summary Cards
   ├── Declaration Findings
   ├── Rule 7 Details
   ├── Rule 8 Details
   ├── Source Image
   ├── Provenance / SHA-256
   └── PDF Download
```

The frontend should **not** contain legal rule logic, OCR, OpenAI calls, package detection, or measurement calculations.

---

# 21. Security / Operational Notes

- Keep API keys outside source control.
- Use environment variables or a secrets manager for OpenAI credentials.
- Do not commit `.env` files.
- Do not expose stored inspection directories directly in production without appropriate access controls.
- Add authentication/authorization before deployment for real officer workflows.
- The current local filesystem storage is suitable for the current development stage; production deployment should use controlled object storage and persistent database infrastructure.

---

# 22. Source and Regulatory Reference

Primary project source:

**SIH 2026 — PS 26034 — Automated Compliance Checker for Packaged Commodities — Full Technical Specification**

The source specification covers the problem definition, Rule 6 declarations, Rule 7 / Rule 8 requirements, functional requirements, suggested architecture, data model, technical challenges, exemptions, and test cases.

This repository implements an **automated inspection-support system based on the SIH specification**. It should not be interpreted as an independent legal enforcement authority or as a substitute for current official statutory text and authorized Legal Metrology Officer review.

---

# 23. Current Project Milestone

```text
✅ Semantic extraction
✅ Date normalization
✅ Package measurement
✅ Automatic package detection path
✅ Rule 6 visibility engine
✅ Rule 7 deterministic engine
✅ Rule 8 deterministic engine
✅ Unified inspection schema/service
✅ Original image storage
✅ SHA-256 provenance
✅ Inspection JSON persistence
✅ PDF inspection report
✅ End-to-end inspection API
✅ Integration tests
✅ Full backend test suite: 220 passed / 1 skipped / 0 failed
🚧 Streamlit frontend — next
⏳ Inspection history / database
⏳ Authentication / roles
⏳ Dashboard / analytics
⏳ Production deployment hardening
```

---

## License / Usage

This repository is an SIH 2026 project implementation. Add the project-specific license, team information, and deployment instructions here before public release.
