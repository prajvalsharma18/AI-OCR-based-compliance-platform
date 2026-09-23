"""FastAPI application entrypoint for SIH PS 26034 - Module 1.

Provides semantic extraction endpoints for packaged commodity label images using OpenAI.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.compliance import router as compliance_router
from app.api.extraction import router as extraction_router
from app.api.inspection import router as inspection_router
from app.api.measurement import router as measurement_router
from app.api.report import router as report_router
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    # Startup verification
    model = settings.LLM_MODEL
    api_key_present = bool(settings.OPENAI_API_KEY.strip())
    print("[*] Starting SIH PS 26034 Compliance Checker API")
    print(f"[*] Configured Vision Model: {model}")
    print(f"[*] OpenAI API Key Configured: {'YES' if api_key_present else 'NO (Set OPENAI_API_KEY in .env)'}")
    yield
    print("[*] Shutting down Compliance Checker API")


app = FastAPI(
    title="SIH 2026 PS 26034 — Automated Compliance Checker API",
    description=(
        "Automated Compliance Checker for Packaged Commodities (SIH PS 26034). "
        "Includes Module 1 (Vision-based semantic label extraction), "
        "Module 2A (Calibrated numeral size measurement), and "
        "Module 3 (Deterministic Rule 7 numeral-height compliance evaluation)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Formats FastAPI request validation errors into standard error envelope."""
    error_messages = []
    for err in exc.errors():
        loc = " -> ".join(str(p) for p in err.get("loc", []))
        msg = err.get("msg", "")
        error_messages.append(f"{loc}: {msg}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT if hasattr(status, "HTTP_422_UNPROCESSABLE_CONTENT") else 422,
        content={
            "success": False,
            "error": {
                "code": "REQUEST_VALIDATION_ERROR",
                "message": "; ".join(error_messages),
            },
        },
    )


# Register API Routers
app.include_router(inspection_router)
app.include_router(extraction_router)
app.include_router(measurement_router)
app.include_router(compliance_router)
app.include_router(report_router)


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint. Verifies server is running without calling external APIs."""
    return {
        "status": "healthy",
        "module": "1: Vision-Based Semantic Label Extraction & 2A: Calibrated Numeral Size Measurement & 3: Rule 7 Numeral-Height & 3B: Rule 8 Spatial Clearance",
        "modules": [
            "1: Vision-Based Semantic Label Extraction",
            "2A: Calibrated Numeral Size Measurement",
            "3: Rule 7 Numeral-Height Evaluation Engine",
            "3B: Rule 8 Spatial Clearance Evaluation Engine",
        ],
        "ps_number": "26034",
        "model": settings.LLM_MODEL,
    }
