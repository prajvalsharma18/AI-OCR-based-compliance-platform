"""Optional live integration test against the real OpenAI Vision API.

Runs ONLY when explicitly enabled:
    RUN_OPENAI_INTEGRATION_TESTS=true
and a valid OPENAI_API_KEY is configured in the environment or .env.
"""

import io
import os
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.config import settings
from app.main import app

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_OPENAI_INTEGRATION_TESTS", "").lower() != "true"
    or not (os.getenv("OPENAI_API_KEY", "").strip() or settings.OPENAI_API_KEY.strip()),
    reason="Requires RUN_OPENAI_INTEGRATION_TESTS=true and valid OPENAI_API_KEY",
)


def create_sample_packaging_image() -> bytes:
    """Creates an image with visible text declarations for live LLM extraction test."""
    img = Image.new("RGB", (600, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Simulated label text
    draw.text((20, 30), "TATA SAMPANN BASMATI RICE", fill=(0, 0, 0))
    draw.text((20, 80), "Net Qty: 500 g", fill=(0, 0, 0))
    draw.text((20, 130), "MRP Rs. 120.00 (inclusive of all taxes)", fill=(0, 0, 0))
    draw.text((20, 180), "Pkd: 08/2026", fill=(0, 0, 0))
    draw.text((20, 230), "Mfg by: Tata Consumer Products Ltd., Pune 411001", fill=(0, 0, 0))
    draw.text((20, 280), "Consumer Care: care@tataconsumer.com, 1800-108-4488", fill=(0, 0, 0))

    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_live_openai_label_extraction():
    """Live integration test verifying OpenAI Responses API and Pydantic validation."""
    client = TestClient(app)
    image_bytes = create_sample_packaging_image()

    response = client.post(
        "/api/v1/extract",
        files={"image": ("test_packaging.jpg", image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "data" in data
    assert data["data"]["net_quantity"]["status"] in ("present", "uncertain")
    assert data["data"]["generic_name"]["status"] in ("present", "uncertain")
