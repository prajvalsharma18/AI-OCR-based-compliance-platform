# Automated Compliance Checker for Packaged Commodities

A **multimodal AI-assisted inspection platform** for analyzing packaged-product labels against Legal Metrology requirements. The system combines **vision-language extraction, calibrated computer vision, deterministic rule engines, FastAPI, MongoDB Atlas, PDF reporting, and Streamlit** to produce traceable inspection results.

> **Core principle:** The LLM extracts meaning; computer vision measures physical dimensions; deterministic code evaluates compliance rules.

LIVE DEMO - https://ai-ocr-based-compliance-platform.onrender.com/


Activate-backend - https://ai-ocr-based-compliance-platform-backend.onrender.com/

This project solves automated compliance screening for packaged commodities. The input is a packaging image plus optional physical package dimensions.

The first stage uses OpenAI Vision for semantic extraction — it identifies declarations such as manufacturer information, generic name, MRP, dates, net quantity and consumer-care details, along with confidence and bounding boxes.

The second stage is classical computer vision using OpenCV. Instead of asking an LLM to estimate font size, I use calibrated pixel measurements to estimate the physical numeral height in millimetres.

Then I run deterministic compliance engines for Rule 6, Rule 7 and Rule 8. Rule 6 handles declaration visibility, Rule 7 compares measured numeral height against the selected ruleset, and Rule 8 evaluates surrounding clear space.

The results are aggregated into a Unified Inspection JSON, while the exact source image is preserved with a SHA-256 hash for provenance. The same inspection can generate a structured PDF report.

On top of that, I built a FastAPI orchestration layer, Streamlit LMO dashboard, MongoDB Atlas persistence, inspection history, historical report retrieval, and persistent LMO review.

The important architectural decision was to keep LLM reasoning, computer vision, and deterministic legal rules separate, rather than making the LLM responsible for all compliance decisions

---

## Overview

The system accepts a packaged-product image and optionally provided package dimensions, then:

1. Extracts regulatory declarations using **OpenAI Vision**.
2. Detects and measures relevant numerals using **OpenCV and image calibration**.
3. Evaluates **Rule 6, Rule 7, and Rule 8** using deterministic Python logic.
4. Produces a unified structured inspection result.
5. Preserves the original image and **SHA-256 provenance**.
6. Generates a PDF inspection report.
7. Persists inspection metadata, findings, and review state in **MongoDB Atlas**.
8. Provides current and historical inspections through **Streamlit**.

The architecture intentionally keeps LLM-based interpretation separate from numerical measurement and compliance decisions.

---

## Architecture

```text
                    Streamlit UI
                Inspection + History
                         │
                         ▼
                   FastAPI Backend
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
  OpenAI Vision      OpenCV/CV       Image Storage
  Semantic OCR       Measurement      + SHA-256
        │                │
        └────────┬───────┘
                 ▼
        Deterministic Rules
             Rule 6/7/8
                 │
                 ▼
        Unified Inspection JSON
                 │
        ┌────────┴─────────┐
        ▼                  ▼
   PDF Report          MongoDB Atlas
                            │
                            ▼
                   Inspection History

```

### Processing Separation

```text
OpenAI Vision
     ↓
Semantic Extraction

OpenCV + Calibration
     ↓
Physical Measurement

Deterministic Python
     ↓
Compliance Evaluation

FastAPI
     ↓
Orchestration

MongoDB Atlas
     ↓
Persistence + History

```

This separation prevents the LLM from acting as the final authority for numerical or spatial compliance decisions.

---

## Core Modules

### 1. Vision-Based Declaration Extraction

Extracts packaging information such as:

- Manufacturer / Packer / Importer
- Generic Name
- MRP
- Manufacturing / Packing / Import Date
- Use By / Best Before
- Net Quantity
- Consumer Care Details

Extraction results can include normalized values, confidence, bounding boxes, and numeral regions.

### 2. Calibrated Computer Vision

Uses OpenCV to perform:

- Package boundary detection
- Image calibration
- Numeral-region isolation
- Connected-component processing
- Pixel-to-mm conversion
- Numeral-height measurement
- Measurement quality estimation

Measurements can use user-provided package dimensions or automatic package-boundary detection.

### 3. Deterministic Compliance Engine

Evaluates:

- **Rule 6** — declaration visibility
- **Rule 7** — applicable numeral-height requirements
- **Rule 8** — spatial clearance requirements

The system supports separate rule-source modes:

```text
sih_ps_26034
doca_statutory_2011

```

The selected rule source is persisted with each inspection.

---

## Inspection Status

The system distinguishes detection from compliance:

| StatusMeaning    |                                                                   |
| ---------------- | ----------------------------------------------------------------- |
| `PASS`           | Applicable requirement satisfied                                  |
| `NON-COMPLIANT`  | Reliable evaluation indicates requirement is not satisfied        |
| `NOT VISIBLE`    | Declaration not detected in the supplied image                    |
| `NOT ASSESSABLE` | Evidence or measurement is insufficient for a reliable conclusion |
| `NOT APPLICABLE` | Rule does not apply                                               |

Importantly:

```text
DETECTED ≠ PASS
NOT VISIBLE ≠ Legally absent

```

Low-confidence measurements can result in `NOT ASSESSABLE` instead of producing an unreliable compliance conclusion.

---

## Evidence & Provenance

Each inspection preserves the original uploaded image and records:

```text
inspection_id
source_image_filename
source_image_sha256
source_image_path
source_image_storage_path

```

SHA-256 is used as an integrity fingerprint for the stored source image.

---

## Reports & Persistence

Each inspection produces:

```text
inspection.json
report.pdf

```

The JSON contains metadata, provenance, findings, rule results, summaries, disclaimers, and artifact references.

PDF reports are generated using **ReportLab** and contain declaration findings, Rule 6/7/8 results, evidence, provenance, and review information.

Historical reports are retrieved as the **stored PDF** rather than regenerated.

### MongoDB Atlas

MongoDB stores:

- Inspection metadata
- Findings
- Summary counters
- Rule-source information
- Timestamps
- Review state
- Artifact references

Image and PDF binaries remain in artifact storage rather than being embedded in MongoDB.

---

## Inspection History

The Streamlit interface supports:

- New inspections
- Historical inspection search
- Date/brand/package filtering
- Inspection reopening
- Source-image retrieval
- Stored PDF retrieval
- JSON export
- Persistent LMO review

The frontend communicates with MongoDB **through FastAPI**, rather than connecting directly to the database.

---

## API

| MethodEndpointPurpose |                                   |                             |
| --------------------- | --------------------------------- | --------------------------- |
| `GET`                 | `/health`                         | Application/database health |
| `POST`                | `/api/v1/inspection`              | End-to-end inspection       |
| `POST`                | `/api/v1/extract`                 | Vision extraction           |
| `POST`                | `/api/v1/measure`                 | Calibrated measurement      |
| `POST`                | `/api/v1/compliance/rule6`        | Rule 6                      |
| `POST`                | `/api/v1/compliance/rule7`        | Rule 7                      |
| `POST`                | `/api/v1/compliance/rule8`        | Rule 8                      |
| `POST`                | `/api/v1/report/pdf`              | Generate PDF                |
| `GET`                 | `/api/v1/inspections`             | Inspection history          |
| `GET`                 | `/api/v1/inspections/{id}`        | Inspection details          |
| `GET`                 | `/api/v1/inspections/{id}/image`  | Source image                |
| `GET`                 | `/api/v1/inspections/{id}/report` | Stored PDF                  |
| `PATCH`               | `/api/v1/inspections/{id}/review` | LMO review                  |

---

## Technology Stack

### AI / GenAI

- OpenAI Vision
- Structured multimodal extraction

### Computer Vision

- OpenCV
- NumPy
- Pillow

### Backend

- Python
- FastAPI
- Pydantic
- Uvicorn

### Database

- MongoDB Atlas
- PyMongo
- mongomock

### Reporting

- ReportLab
- pypdf
- pypdfium2

### Frontend

- Streamlit
- Requests

### Testing

- pytest
- pytest-asyncio
- httpx
- Mocked integrations

---

## Project Structure

```text
ai_ocr_pipeline/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── db/
│   │   ├── repositories/
│   │   ├── schemas/
│   │   └── services/
│   ├── tests/
│   ├── reports/
│   └── requirements.txt
│
├── frontend/
│   ├── app.py
│   ├── api_client.py
│   ├── ui_components.py
│   └── requirements.txt
│
├── README.md
└── .gitignore

```

---

## Setup

### Backend

```powershell
cd backend
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

```

Configure MongoDB:

```powershell
$env:MONGODB_URI="<MongoDB Atlas connection string>"
$env:MONGODB_DATABASE="AI-OCR-based-platform"
$env:MONGODB_ENABLED="true"

```

> Do not commit `.env`, database credentials, or API keys.

### Run Backend

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload

```

API documentation:

```text
http://127.0.0.1:8001/docs

```

### Run Frontend

```powershell
streamlit run frontend/app.py

```

Open:

```text
http://localhost:8501

```

---

## Testing

Latest reported regression suite:

```text
231 passed
2 skipped
1 warning
0 failed

```

Coverage includes extraction, measurement, Rule 6/7/8 evaluation, image provenance, report generation, artifact retrieval, inspection history, review behavior, and MongoDB persistence.

Optional live Atlas verification:

```powershell
$env:RUN_LIVE_MONGODB_TESTS="1"
.\venv\Scripts\python -m pytest -q tests/test_live_mongodb_optional.py

```

Latest reported live verification:

```text
MongoDB: connected
Opt-in live persistence test: 1 passed

```

---

## Current Limitations

- Single-image inspection; multi-panel package inspection is not implemented.
- Image/PDF artifacts currently use local storage.
- Authentication and RBAC are not implemented.
- Advanced enforcement analytics are outside the current core scope.
- Bounding boxes are rendered as overlays rather than an advanced interactive canvas.

### Planned Extensions

- Multi-panel inspection
- S3-compatible object storage
- Authentication/RBAC
- Inspection analytics
- Interactive evidence visualization
- Offline/mobile inspection workflow
- Stateful orchestration where justified

LangChain/LangGraph is **not required by the current architecture**; it can be introduced later if stateful routing, retries, human-in-the-loop workflows, or tool orchestration provide genuine architectural value.

---

## SIH Functional Coverage

### Implemented

```text
Image upload                    ✅
Vision extraction               ✅
Calibrated measurement          ✅
Rule 6                          ✅
Rule 7                          ✅
Rule 8                          ✅
Unified inspection JSON         ✅
PDF reporting                   ✅
Image provenance + SHA-256      ✅
FastAPI backend                 ✅
Streamlit interface             ✅
MongoDB persistence             ✅
Inspection history              ✅
Historical artifact retrieval   ✅
LMO review                      ✅
Automated regression testing    ✅

```

### Extended Scope

```text
Role-based authentication       ⏳
Advanced analytics              ⏳
Multi-panel inspection          ⏳
Cloud object storage            ⏳
Offline/mobile workflow         ⏳

```

---

## Project Positioning

> **Multimodal AI + Computer Vision + Deterministic Compliance Inspection Platform**

The project combines:

```text
Vision AI
   ↓
Semantic Understanding

Computer Vision
   ↓
Physical Measurement

Deterministic Rules
   ↓
Compliance Evaluation

FastAPI
   ↓
Backend Orchestration

MongoDB Atlas
   ↓
Persistence + History

Streamlit
   ↓
Inspection Interface

```

It is therefore **not primarily a LangChain/LangGraph project**. Its technical strength comes from combining multimodal AI with classical computer vision and deterministic compliance logic.

---

## Regulatory Disclaimer

This application is an **inspection-support system**, not a replacement for authorized regulatory judgment.

`NOT VISIBLE` indicates that a declaration was not detected in the supplied image. `NOT ASSESSABLE` indicates insufficient evidence or measurement confidence for a reliable automated conclusion. Final legal interpretation and enforcement decisions remain with the authorized Legal Metrology Officer.

The implemented SIH ruleset should be verified against authoritative and current regulatory material before real-world enforcement use.

---

## Status

**Core platform implemented and locally verified.**

```text
Semantic extraction          ✅
CV measurement               ✅
Rule 6/7/8                   ✅
Unified inspection           ✅
PDF reporting                ✅
MongoDB persistence          ✅
Inspection history           ✅
LMO review                   ✅
Streamlit frontend           ✅
Regression suite             ✅
Live Atlas verification      ✅

```

