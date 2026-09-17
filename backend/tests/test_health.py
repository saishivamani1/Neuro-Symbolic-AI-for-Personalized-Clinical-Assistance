"""
tests/test_health.py

Stage 1 test: verify the /api/v1/health endpoint returns HTTP 200
with the expected JSON structure.

Run with:
    pytest tests/test_health.py -v
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a test client for the FastAPI application."""
    return TestClient(app, raise_server_exceptions=True)


class TestHealthEndpoint:
    """Tests for GET /api/v1/health."""

    def test_status_code_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/health")
        assert response.status_code == 200, response.text

    def test_response_has_status_field(self, client: TestClient) -> None:
        data = client.get("/api/v1/health").json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded")

    def test_response_has_services(self, client: TestClient) -> None:
        data = client.get("/api/v1/health").json()
        assert "services" in data
        services = data["services"]
        assert "api" in services
        assert "vector_store" in services
        assert "knowledge_graph" in services
        assert "llm" in services

    def test_api_service_is_up(self, client: TestClient) -> None:
        data = client.get("/api/v1/health").json()
        assert data["services"]["api"]["status"] == "up"

    def test_response_has_version(self, client: TestClient) -> None:
        data = client.get("/api/v1/health").json()
        assert "version" in data

    def test_response_has_environment(self, client: TestClient) -> None:
        data = client.get("/api/v1/health").json()
        assert "environment" in data

    def test_root_endpoint(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "docs" in data
        assert "health" in data

    def test_chat_endpoint_validates_request(self, client: TestClient) -> None:
        """Confirm chat endpoint validates missing payload correctly."""
        response = client.post("/api/v1/chat", json={})
        assert response.status_code == 422


class TestTextUtils:
    """Tests for app.utils.text helpers."""

    def test_normalise_text_collapses_whitespace(self) -> None:
        from app.utils.text import normalise_text

        result = normalise_text("hello   world\n\t!")
        assert result == "hello world !"

    def test_truncate_text(self) -> None:
        from app.utils.text import truncate_text

        assert truncate_text("abc", 5) == "abc"
        assert truncate_text("abcdef", 3) == "abc…"

    def test_clean_medical_text_removes_page_numbers(self) -> None:
        from app.utils.text import clean_medical_text

        text = "Some clinical text.\nPage 3 of 45\nMore text here please."
        result = clean_medical_text(text)
        assert "Page 3 of 45" not in result
        assert "clinical" in result


class TestIDUtils:
    """Tests for app.utils.ids helpers."""

    def test_generate_document_id_is_deterministic(self) -> None:
        from app.utils.ids import generate_document_id

        id1 = generate_document_id("report.pdf", "abc123")
        id2 = generate_document_id("report.pdf", "abc123")
        assert id1 == id2
        assert id1.startswith("doc_")

    def test_generate_document_id_different_for_different_content(self) -> None:
        from app.utils.ids import generate_document_id

        id1 = generate_document_id("report.pdf", "abc123")
        id2 = generate_document_id("report.pdf", "xyz999")
        assert id1 != id2

    def test_generate_chunk_id_format(self) -> None:
        from app.utils.ids import generate_chunk_id

        chunk_id = generate_chunk_id("doc_abc123", 7)
        assert chunk_id == "doc_abc123_chunk_0007"

    def test_content_hash_returns_hex(self) -> None:
        from app.utils.ids import content_hash

        digest = content_hash(b"test data")
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)
