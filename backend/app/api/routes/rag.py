"""
app/api/routes/rag.py

Temporary testing and inspection endpoints for RAG retrieval.

Endpoints:
- POST /api/v1/rag/search: Semantic similarity retrieval endpoint
- GET  /api/v1/rag/search: Query-param based retrieval endpoint (browser/curl friendly)
- GET  /api/v1/rag/stats: Vector store statistics and chunk counts
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query, status

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.retriever import get_retriever
from app.rag.vector_store import get_vector_store
from app.schemas.rag import RAGSearchRequest, RAGSearchResponse
from app.schemas.response import APIResponse

router = APIRouter()
logger = get_logger(__name__)


@router.post(
    "/rag/search",
    response_model=APIResponse[RAGSearchResponse],
    status_code=status.HTTP_200_OK,
    tags=["RAG"],
    summary="Semantic similarity search over ingested medical documents",
)
async def rag_search_post(request: RAGSearchRequest) -> APIResponse[RAGSearchResponse]:
    """Search for relevant medical guideline chunks using dense vector similarity."""
    retriever = get_retriever()
    response = retriever.retrieve(
        query=request.query,
        top_k=request.top_k,
        document_id=request.document_id,
    )
    return APIResponse(
        success=True,
        data=response,
    )


@router.get(
    "/rag/search",
    response_model=APIResponse[RAGSearchResponse],
    status_code=status.HTTP_200_OK,
    tags=["RAG"],
    summary="Semantic similarity search (GET version for testing)",
)
async def rag_search_get(
    query: str = Query(..., min_length=1, description="Search query"),
    top_k: Optional[int] = Query(default=None, ge=1, le=50, description="Top K results"),
    document_id: Optional[str] = Query(default=None, description="Filter by document ID"),
) -> APIResponse[RAGSearchResponse]:
    """Query parameters version of semantic similarity retrieval for fast manual testing."""
    retriever = get_retriever()
    response = retriever.retrieve(
        query=query,
        top_k=top_k,
        document_id=document_id,
    )
    return APIResponse(
        success=True,
        data=response,
    )


@router.get(
    "/rag/stats",
    response_model=APIResponse[dict],
    tags=["RAG"],
    summary="RAG vector store collection statistics",
)
async def rag_stats() -> APIResponse[dict]:
    """Return count of stored chunks and collection metadata."""
    settings = get_settings()
    vector_store = get_vector_store()
    chunk_count = vector_store.count()
    return APIResponse(
        success=True,
        data={
            "collection_name": settings.chroma_collection_name,
            "persist_directory": settings.chroma_persist_directory,
            "total_chunks": chunk_count,
            "embedding_model": settings.embedding_model,
        },
    )
