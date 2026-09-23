"""Unit and integration tests for FastAPI extraction endpoint and ExtractionService."""

import io
import json
from unittest.mock import AsyncMock, patch
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services.vision_llm import (
    VisionLLMAPIError,
    VisionLLMConfigurationError,
)
from tests.test_schema import sample_valid_extraction_dict


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


def create_dummy_image_bytes(format_name: str = "JPEG", size=(200, 200)) -> bytes:
    """Generates an in-memory valid image for testing."""
    image = Image.new("RGB", size, color=(240, 240, 240))
    buffer = io.BytesIO()
    image.save(buffer, format=format_name)
    return buffer.getvalue()


def test_health_check(client):
    """Test health endpoint returns 200 without calling OpenAI."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["ps_number"] == "26034"
    assert "model" in data


@pytest.mark.asyncio
async def test_extract_success(client):
    """Test successful image extraction with mocked OpenAI Vision LLM response."""
    valid_data = sample_valid_extraction_dict()
    valid_json_str = json.dumps(valid_data)
    dummy_image = create_dummy_image_bytes("JPEG")

    with patch("app.services.vision_llm.VisionLLMClient.extract", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = valid_json_str

        response = client.post(
            "/api/v1/extract",
            files={"image": ("test_pack.jpg", dummy_image, "image/jpeg")},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        sem = body["data"]["semantic_extraction"]
        assert sem["generic_name"]["value"] == "Basmati Rice"
        assert sem["net_quantity"]["value"] == 500.0
        assert sem["mrp"]["value"] == 120.0
        assert body["data"]["image"]["coordinate_system"] == "normalized_0_1"
        assert mock_llm.call_count == 1


@pytest.mark.asyncio
async def test_extract_missing_and_uncertain_fields(client):
    """Test extraction with missing and uncertain declaration statuses."""
    data = sample_valid_extraction_dict()
    data["mrp"]["status"] = "missing"
    data["mrp"]["value"] = None
    data["mrp"]["raw_text"] = None
    data["consumer_care"]["status"] = "uncertain"
    data["consumer_care"]["phone"] = None
    data["consumer_care"]["email"] = None

    dummy_image = create_dummy_image_bytes("PNG")

    with patch("app.services.vision_llm.VisionLLMClient.extract", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = json.dumps(data)

        response = client.post(
            "/api/v1/extract",
            files={"image": ("pack.png", dummy_image, "image/png")},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["data"]["semantic_extraction"]["mrp"]["status"] == "missing"
        assert body["data"]["semantic_extraction"]["consumer_care"]["status"] == "uncertain"


@pytest.mark.asyncio
async def test_extract_retry_on_initial_malformed_json(client):
    """Test that malformed initial output triggers a repair retry and succeeds."""
    valid_data = sample_valid_extraction_dict()
    valid_json_str = json.dumps(valid_data)
    malformed_json_str = "```json { invalid json string here ... "

    dummy_image = create_dummy_image_bytes("PNG")

    with patch("app.services.vision_llm.VisionLLMClient.extract", new_callable=AsyncMock) as mock_llm:
        # First call returns malformed JSON, second call returns repaired valid JSON
        mock_llm.side_effect = [malformed_json_str, valid_json_str]

        response = client.post(
            "/api/v1/extract",
            files={"image": ("biscuit.png", dummy_image, "image/png")},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["data"]["semantic_extraction"]["package"]["product_category"] == "rice"
        assert mock_llm.call_count == 2


@pytest.mark.asyncio
async def test_extract_pydantic_validation_failure_triggers_repair(client):
    """Test that Pydantic validation failure (e.g. invalid confidence) triggers repair."""
    invalid_data = sample_valid_extraction_dict()
    invalid_data["mrp"]["confidence"] = 1.5  # Exceeds max 1.0

    valid_data = sample_valid_extraction_dict()
    dummy_image = create_dummy_image_bytes("JPEG")

    with patch("app.services.vision_llm.VisionLLMClient.extract", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [json.dumps(invalid_data), json.dumps(valid_data)]

        response = client.post(
            "/api/v1/extract",
            files={"image": ("sample.jpg", dummy_image, "image/jpeg")},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert mock_llm.call_count == 2


@pytest.mark.asyncio
async def test_extract_fails_after_repair_retry(client):
    """Test that when both attempts produce invalid JSON, a structured 422 error is returned."""
    invalid_json_str = "{ 'unquoted_key': invalid }"
    dummy_image = create_dummy_image_bytes("JPEG")

    with patch("app.services.vision_llm.VisionLLMClient.extract", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = [invalid_json_str, invalid_json_str]

        response = client.post(
            "/api/v1/extract",
            files={"image": ("oil.jpg", dummy_image, "image/jpeg")},
        )

        assert response.status_code == 422
        body = response.json()
        assert body["success"] is False
        assert body["error"]["code"] == "EXTRACTION_SCHEMA_VALIDATION_FAILED"
        assert mock_llm.call_count == 2


def test_extract_unsupported_media_type(client):
    """Test rejection of unsupported file types (e.g. text/plain)."""
    response = client.post(
        "/api/v1/extract",
        files={"image": ("notes.txt", b"plain text content", "text/plain")},
    )

    assert response.status_code == 415
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_extract_corrupted_image(client):
    """Test rejection of corrupted image payload."""
    corrupted_bytes = b"Not real image binary data"

    response = client.post(
        "/api/v1/extract",
        files={"image": ("corrupted.jpg", corrupted_bytes, "image/jpeg")},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "CORRUPTED_IMAGE"


def test_extract_empty_file(client):
    """Test rejection of empty file (0 bytes)."""
    response = client.post(
        "/api/v1/extract",
        files={"image": ("empty.png", b"", "image/png")},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "EMPTY_FILE"


def test_extract_missing_file_payload(client):
    """Test error response when image field is missing from request."""
    response = client.post("/api/v1/extract")
    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "REQUEST_VALIDATION_ERROR"


@pytest.mark.asyncio
async def test_extract_missing_api_key(client):
    """Test error response when OPENAI_API_KEY is not set."""
    dummy_image = create_dummy_image_bytes("JPEG")

    with patch("app.services.vision_llm.VisionLLMClient.extract", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = VisionLLMConfigurationError(
            code="MISSING_API_KEY",
            message="OPENAI_API_KEY is not configured.",
        )

        response = client.post(
            "/api/v1/extract",
            files={"image": ("sample.jpg", dummy_image, "image/jpeg")},
        )

        assert response.status_code == 500
        body = response.json()
        assert body["success"] is False
        assert body["error"]["code"] == "MISSING_API_KEY"


@pytest.mark.asyncio
async def test_extract_upstream_llm_failure(client):
    """Test handling when OpenAI Vision returns an API error."""
    dummy_image = create_dummy_image_bytes("JPEG")

    with patch("app.services.vision_llm.VisionLLMClient.extract", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = VisionLLMAPIError(
            code="OPENAI_RATE_LIMIT",
            message="OpenAI rate limit reached or quota exceeded.",
        )

        response = client.post(
            "/api/v1/extract",
            files={"image": ("sample.jpg", dummy_image, "image/jpeg")},
        )

        assert response.status_code == 502
        body = response.json()
        assert body["success"] is False
        assert body["error"]["code"] == "OPENAI_RATE_LIMIT"
        assert "rate limit" in body["error"]["message"].lower()


@pytest.mark.asyncio
async def test_extract_oversized_image(client):
    """Test rejection when uploaded image exceeds MAX_IMAGE_SIZE_MB limit."""
    dummy_image = create_dummy_image_bytes("JPEG")

    with patch("app.config.settings.MAX_IMAGE_SIZE_MB", 0.0001):  # ~100 bytes
        response = client.post(
            "/api/v1/extract",
            files={"image": ("large.jpg", dummy_image, "image/jpeg")},
        )

        assert response.status_code == 413
        body = response.json()
        assert body["success"] is False
        assert body["error"]["code"] == "FILE_TOO_LARGE"
