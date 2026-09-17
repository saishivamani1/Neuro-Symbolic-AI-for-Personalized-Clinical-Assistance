"""
tests/test_rag.py

Comprehensive tests for Stage 2 RAG Pipeline components:
1. PDF Ingestion & PyMuPDF Loader (app.rag.loaders)
2. Medical Text Cleaning & Normalisation (app.utils.text)
3. Page-Aware Chunking & Metadata Preservation (app.rag.chunking)
4. BGE Dense Embeddings (app.rag.embeddings)
5. ChromaDB Persistent Vector Store (app.rag.vector_store)
6. Semantic Retrieval Service (app.rag.retriever)
7. End-to-End Pipeline (app.rag.pipeline)
8. API Endpoints (/api/v1/documents/upload, /api/v1/rag/search, /api/v1/rag/stats)
"""

from __future__ import annotations

import io
import shutil
import tempfile
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import InvalidDocumentError
from app.main import app
from app.rag.chunking import MedicalChunker
from app.rag.embeddings import BGEEmbeddings
from app.rag.loaders import PDFLoader
from app.rag.pipeline import DocumentPipeline
from app.rag.retriever import RAGRetriever
from app.rag.vector_store import ChromaVectorStore
from app.schemas.rag import PageContent


# --------------------------------------------------------------------------- #
# Helpers & Fixtures
# --------------------------------------------------------------------------- #


def create_in_memory_pdf(pages_text: List[str]) -> bytes:
    """Create a minimal PDF in memory with specified text per page."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(50, 50, 550, 750), text, fontsize=11)
    pdf_bytes = doc.write()
    doc.close()
    return pdf_bytes


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """Fixture providing a 3-page synthetic medical PDF."""
    pages = [
        (
            "Clinical Practice Guidelines for Hypertension\n"
            "Page 1 of 3\n"
            "----------------------------------------\n"
            "Hypertension is defined as persistent blood pressure >= 140/90 mmHg.\n"
            "First-line treatment includes ACE inhibitors such as Lisinopril 10-40mg daily,\n"
            "or ARBs like Losartan for renal protection."
        ),
        (
            "Type 2 Diabetes Mellitus Clinical Protocols\n"
            "Page 2 of 3\n"
            "----------------------------------------\n"
            "Metformin is the standard first-line oral antidiabetic drug.\n"
            "Caution: Metformin is contraindicated in severe renal failure when eGFR < 30 mL/min."
        ),
        (
            "Drug Interactions and Safety Monitoring\n"
            "Page 3 of 3\n"
            "----------------------------------------\n"
            "Co-administration of ACE inhibitors and Spironolactone increases hyperkalemia risk.\n"
            "Serum potassium must be monitored closely."
        ),
    ]
    return create_in_memory_pdf(pages)


class MockFastEmbeddings(BGEEmbeddings):
    """Fast deterministic mock embeddings for rapid unit testing without model download."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__(model_name="mock-model", device="cpu")
        self.dim = dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for t in texts:
            val = float(len(t) % 100) / 100.0
            results.append([val] * self.dim)
        return results

    def embed_documents_batched(
        self, texts: List[str], batch_size: int = 64, progress_callback=None
    ):
        total = len(texts)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = self.embed_documents(texts[start:end])
            if progress_callback:
                progress_callback(end, total, 1)
            yield (start, end, batch)

    def embed_query(self, text: str) -> List[float]:
        val = float(len(text) % 100) / 100.0
        return [val] * self.dim


@pytest.fixture
def temp_chroma_dir():
    """Fixture providing a clean temporary directory for ChromaDB."""
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def vector_store(temp_chroma_dir) -> ChromaVectorStore:
    """Fixture providing an isolated ChromaVectorStore instance."""
    return ChromaVectorStore(
        persist_directory=temp_chroma_dir,
        collection_name="test_collection",
    )


@pytest.fixture
def client() -> TestClient:
    """FastAPI TestClient."""
    return TestClient(app, raise_server_exceptions=True)


# --------------------------------------------------------------------------- #
# 1. PDF Loader Tests
# --------------------------------------------------------------------------- #


class TestPDFLoader:
    def test_load_from_bytes_success(self, sample_pdf_bytes: bytes) -> None:
        loader = PDFLoader(apply_cleaning=True)
        doc_id, file_hash, pages = loader.load_from_bytes(
            sample_pdf_bytes, filename="hypertension_guidelines.pdf"
        )

        assert doc_id.startswith("doc_")
        assert len(file_hash) == 64
        assert len(pages) == 3
        assert pages[0].page_number == 1
        assert "Hypertension" in pages[0].text
        assert pages[1].page_number == 2
        assert "Metformin" in pages[1].text
        assert pages[2].page_number == 3
        assert "Spironolactone" in pages[2].text

    def test_load_from_bytes_cleaning_applied(self, sample_pdf_bytes: bytes) -> None:
        loader = PDFLoader(apply_cleaning=True)
        _, _, pages = loader.load_from_bytes(sample_pdf_bytes, "test.pdf")
        # Page numbers like "Page 1 of 3" and divider "---" should be cleaned
        for page in pages:
            assert "Page 1 of 3" not in page.text
            assert "----------------------------------------" not in page.text

    def test_load_from_bytes_empty_raises(self) -> None:
        loader = PDFLoader()
        with pytest.raises(InvalidDocumentError):
            loader.load_from_bytes(b"", "empty.pdf")

    def test_load_from_bytes_invalid_format_raises(self) -> None:
        loader = PDFLoader()
        with pytest.raises(InvalidDocumentError):
            loader.load_from_bytes(b"This is just plain text, not a PDF", "fake.pdf")

    def test_load_from_path(self, sample_pdf_bytes: bytes, tmp_path: Path) -> None:
        pdf_file = tmp_path / "sample.pdf"
        pdf_file.write_bytes(sample_pdf_bytes)

        loader = PDFLoader()
        doc_id, file_hash, pages = loader.load_from_path(pdf_file)
        assert len(pages) == 3
        assert doc_id.startswith("doc_")


# --------------------------------------------------------------------------- #
# 2. Medical Chunker Tests
# --------------------------------------------------------------------------- #


class TestMedicalChunker:
    def test_page_aware_chunking_preserves_provenance(self) -> None:
        chunker = MedicalChunker(chunk_size=100, chunk_overlap=20)
        pages = [
            PageContent(
                page_number=1,
                text="Hypertension guideline paragraph one. Blood pressure management is essential.",
                char_count=77,
            ),
            PageContent(
                page_number=2,
                text="Diabetes guideline on page two. Metformin dosing protocols.",
                char_count=59,
            ),
        ]

        chunks = chunker.chunk_pages(
            pages=pages,
            document_id="doc_test123",
            filename="guidelines.pdf",
        )

        assert len(chunks) >= 2
        for idx, chunk in enumerate(chunks):
            assert chunk.document_id == "doc_test123"
            assert chunk.filename == "guidelines.pdf"
            assert chunk.chunk_id == f"doc_test123_chunk_{idx:04d}"
            assert chunk.page_number in (1, 2)
            assert chunk.metadata["page_number"] == chunk.page_number
            assert len(chunk.text) > 0

    def test_empty_page_handling(self) -> None:
        chunker = MedicalChunker()
        pages = [
            PageContent(page_number=1, text="", char_count=0),
            PageContent(page_number=2, text="Valid content on page 2", char_count=23),
        ]
        chunks = chunker.chunk_pages(pages, "doc_empty", "test.pdf")
        assert len(chunks) == 1
        assert chunks[0].page_number == 2


# --------------------------------------------------------------------------- #
# 3. Chroma Vector Store Tests
# --------------------------------------------------------------------------- #


class TestChromaVectorStore:
    def test_add_chunks_and_query(self, vector_store: ChromaVectorStore) -> None:
        chunker = MedicalChunker()
        pages = [
            PageContent(
                page_number=1,
                text="Hypertension first-line medications include ACE inhibitors and ARBs.",
                char_count=68,
            )
        ]
        chunks = chunker.chunk_pages(pages, "doc_htn", "htn.pdf")

        # Mock embeddings
        embeddings = [[0.1] * 384 for _ in chunks]
        added = vector_store.add_chunks(chunks=chunks, embeddings=embeddings)

        assert added == len(chunks)
        assert vector_store.count() == len(chunks)

        # Query
        query_vec = [0.1] * 384
        res = vector_store.query(query_embeddings=[query_vec], top_k=5)
        assert len(res["ids"][0]) == 1
        assert res["ids"][0][0] == chunks[0].chunk_id

    def test_delete_document(self, vector_store: ChromaVectorStore) -> None:
        chunker = MedicalChunker()
        pages1 = [PageContent(page_number=1, text="Doc 1 content", char_count=13)]
        pages2 = [PageContent(page_number=1, text="Doc 2 content", char_count=13)]
        chunks1 = chunker.chunk_pages(pages1, "doc_1", "doc1.pdf")
        chunks2 = chunker.chunk_pages(pages2, "doc_2", "doc2.pdf")

        vector_store.add_chunks(chunks1, [[0.2] * 64] * len(chunks1))
        vector_store.add_chunks(chunks2, [[0.3] * 64] * len(chunks2))
        assert vector_store.count() == 2

        deleted = vector_store.delete_document("doc_1")
        assert deleted == 1
        assert vector_store.count() == 1


# --------------------------------------------------------------------------- #
# 4. RAG Retriever & Pipeline Integration Tests
# --------------------------------------------------------------------------- #


class TestRAGRetrieverAndPipeline:
    def test_end_to_end_pipeline_and_retrieval(
        self, sample_pdf_bytes: bytes, temp_chroma_dir: str
    ) -> None:
        mock_embeds = MockFastEmbeddings()
        store = ChromaVectorStore(
            persist_directory=temp_chroma_dir,
            collection_name="pipeline_test_col",
        )

        pipeline = DocumentPipeline(
            loader=PDFLoader(apply_cleaning=True),
            chunker=MedicalChunker(chunk_size=200, chunk_overlap=30),
            embeddings_service=mock_embeds,
            vector_store=store,
        )

        # Ingest
        result = pipeline.ingest_bytes(sample_pdf_bytes, filename="clinical_protocols.pdf")
        assert result.status == "success"
        assert result.page_count == 3
        assert result.chunk_count >= 3
        assert store.count() == result.chunk_count

        # Retrieve
        retriever = RAGRetriever(
            vector_store=store,
            embeddings_service=mock_embeds,
            default_top_k=3,
        )
        search_res = retriever.retrieve(query="What is the first line treatment for hypertension?")

        assert search_res.total_results > 0
        first_match = search_res.results[0]
        assert first_match.document_id == result.document_id
        assert first_match.filename == "clinical_protocols.pdf"
        assert first_match.page_number in (1, 2, 3)
        assert first_match.score >= 0.0


# --------------------------------------------------------------------------- #
# 5. API Endpoints Tests
# --------------------------------------------------------------------------- #


class TestAPIEndpoints:
    def test_upload_invalid_extension(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", b"plain text", "text/plain")},
        )
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert data["error"]["error_code"] == "INVALID_DOCUMENT"

    def test_upload_empty_pdf(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False

    def test_rag_stats_endpoint(self, client: TestClient) -> None:
        response = client.get("/api/v1/rag/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "total_chunks" in data["data"]
        assert "collection_name" in data["data"]

    def test_rag_search_get_endpoint(self, client: TestClient) -> None:
        response = client.get("/api/v1/rag/search?query=hypertension&top_k=2")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "results" in data["data"]
        assert "total_results" in data["data"]

    def test_rag_search_post_endpoint(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/rag/search",
            json={"query": "Type 2 diabetes Metformin", "top_k": 3},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["query"] == "Type 2 diabetes Metformin"

    def test_upload_valid_pdf_and_delete(
        self, client: TestClient, sample_pdf_bytes: bytes
    ) -> None:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("clinical_test.pdf", sample_pdf_bytes, "application/pdf")},
            data={"save_to_disk": "false"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        doc_id = data["data"]["document_id"]
        assert doc_id.startswith("doc_")
        assert data["data"]["page_count"] == 3

        # Delete document
        del_resp = client.delete(f"/api/v1/documents/{doc_id}")
        assert del_resp.status_code == 200
        del_data = del_resp.json()
        assert del_data["success"] is True
        assert del_data["data"]["document_id"] == doc_id
