"""
app/schemas/rag.py

Pydantic schemas for RAG pipeline requests, responses, and internal models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata associated with an ingested document."""

    document_id: str = Field(description="Unique deterministic ID for the document.")
    filename: str = Field(description="Original filename.")
    content_hash: str = Field(description="SHA-256 hash of the document content.")
    page_count: int = Field(default=0, description="Total number of pages extracted.")
    total_characters: int = Field(default=0, description="Total character count across all pages.")


class PageContent(BaseModel):
    """Extracted text and metadata for a single PDF page."""

    page_number: int = Field(description="1-indexed page number.")
    text: str = Field(description="Raw or cleaned text of the page.")
    char_count: int = Field(default=0, description="Character count for this page.")


class DocumentChunk(BaseModel):
    """A single page-aware chunk produced by the chunker."""

    chunk_id: str = Field(description="Unique ID in format {document_id}_chunk_{index:04d}.")
    document_id: str = Field(description="Parent document identifier.")
    filename: str = Field(description="Original source filename.")
    page_number: int = Field(description="Page number this chunk originated from.")
    chunk_index: int = Field(description="Zero-based index of this chunk within the document.")
    text: str = Field(description="Cleaned text content of the chunk.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary extra metadata.")


class DocumentIngestionResult(BaseModel):
    """Summary of a document ingestion run."""

    document_id: str
    filename: str
    content_hash: str
    page_count: int
    chunk_count: int
    status: str = "success"
    message: str = "Document successfully ingested."


class RAGSearchRequest(BaseModel):
    """Query payload for semantic retrieval."""

    query: str = Field(..., min_length=1, description="Medical query or question.")
    top_k: Optional[int] = Field(default=None, ge=1, le=50, description="Number of results to retrieve (defaults to settings.top_k).")
    document_id: Optional[str] = Field(default=None, description="Optional document ID to restrict search to.")


class RAGSearchResultChunk(BaseModel):
    """A retrieved chunk with relevance score and metadata."""

    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    content: str
    score: float = Field(description="Relevance or similarity score (e.g. cosine distance/similarity).")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGSearchResponse(BaseModel):
    """Response payload for RAG retrieval."""

    query: str
    total_results: int
    results: List[RAGSearchResultChunk]
