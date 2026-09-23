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
from app.api.history import router as history_router
from app.config import settings
from app.db.indexes import ensure_indexes
from app.db.mongo import check_mongo_connection, close_mongo_client, get_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    # Startup verification
    model = settings.LLM_MODEL
    api_key_present = bool(settings.OPENAI_API_KEY.strip())
    print("[*] Starting SIH PS 26034 Compliance Checker API")
    print(f"[*] Configured Vision Model: {model}")
    print(f"[*] OpenAI API Key Configured: {'YES' if api_key_present else 'NO (Set OPENAI_API_KEY in .env)'}")

    # MongoDB connection and index initialization
    mongo_status = check_mongo_connection()
    print(f"[*] MongoDB Persistence Layer: {mongo_status.get('status', 'unknown').upper()}")
    if mongo_status.get("status") == "connected":
        db = get_database()
        ensure_indexes(db)
    elif mongo_status.get("status") == "disabled":
        print("[*] MongoDB is disabled via MONGODB_ENABLED=false.")
    else:
        print(f"[*] Note: MongoDB is not connected ({mongo_status.get('message', 'server offline')}). Filesystem storage will be used.")

    yield
    print("[*] Shutting down Compliance Checker API")
    close_mongo_client()


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
app.include_router(history_router)


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint. Verifies server and database status without calling external APIs."""
    mongo_health = check_mongo_connection()
    return {
        "status": "healthy",
        "database": {
            "mongodb": mongo_health.get("status", "unknown"),
            "details": mongo_health.get("message") or f"Database: {mongo_health.get('database')}",
        },
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
