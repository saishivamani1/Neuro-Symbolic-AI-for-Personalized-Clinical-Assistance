"""
app/rag/retriever.py

Semantic retrieval service for the Neuro-Symbolic Healthcare Intelligence Platform.

Converts medical queries into dense embeddings, performs similarity search against
the ChromaDB vector store, ranks results, and packages chunks with complete
provenance metadata (document_id, page_number, chunk_index, similarity score).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.core.exceptions import RetrievalError
from app.core.logging import get_logger
from app.rag.embeddings import BGEEmbeddings, get_embeddings_service
from app.rag.vector_store import ChromaVectorStore, get_vector_store
from app.schemas.rag import RAGSearchRequest, RAGSearchResponse, RAGSearchResultChunk
from app.utils.text import normalise_text

logger = get_logger(__name__)


class RAGRetriever:
    """Retrieves relevant medical document chunks given a natural language query."""

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        embeddings_service: Optional[BGEEmbeddings] = None,
        default_top_k: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self.vector_store = vector_store or get_vector_store()
        self.embeddings_service = embeddings_service or get_embeddings_service()
        self.default_top_k = default_top_k or settings.top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None,
        min_score: float = 0.0,
    ) -> RAGSearchResponse:
        """Retrieve the most relevant chunks for a clinical query.

        Parameters
        ----------
        query : str
            The medical or clinical question / search term.
        top_k : Optional[int]
            Number of top results to return.
        document_id : Optional[str]
            Optional filter to restrict retrieval to a specific document.
        min_score : float
            Minimum similarity threshold (0.0 to 1.0) to filter out irrelevant chunks.

        Returns
        -------
        RAGSearchResponse
            Structured response containing list of ranked chunks with scores and metadata.
        """
        cleaned_query = normalise_text(query)
        if not cleaned_query:
            return RAGSearchResponse(query=query, total_results=0, results=[])

        k = top_k or self.default_top_k

        # If vector store is empty, return empty result list
        if self.vector_store.count() == 0:
            logger.warning("RAG retrieval invoked on empty vector store collection.")
            return RAGSearchResponse(query=cleaned_query, total_results=0, results=[])

        # Generate query embedding
        query_embedding = self.embeddings_service.embed_query(cleaned_query)

        # Build metadata filter if requested
        where_filter: Optional[Dict[str, Any]] = None
        if document_id:
            where_filter = {"document_id": document_id}

        # Query ChromaDB
        raw_results = self.vector_store.query(
            query_embeddings=[query_embedding],
            top_k=k,
            where=where_filter,
        )

        results: List[RAGSearchResultChunk] = []

        if raw_results and "ids" in raw_results and raw_results["ids"]:
            ids_list = raw_results["ids"][0]
            docs_list = raw_results["documents"][0] if "documents" in raw_results else []
            metas_list = raw_results["metadatas"][0] if "metadatas" in raw_results else []
            distances_list = raw_results["distances"][0] if "distances" in raw_results else []

            for i, chunk_id in enumerate(ids_list):
                doc_text = docs_list[i] if i < len(docs_list) else ""
                meta = metas_list[i] if i < len(metas_list) else {}
                dist = distances_list[i] if i < len(distances_list) else 0.0

                # In ChromaDB cosine distance: similarity = 1.0 - (distance / 2) or 1.0 - distance
                # Cosine distance is in [0, 2], so similarity score is bounded in [-1, 1] or [0, 1]
                score = round(max(0.0, 1.0 - dist), 4)

                if score < min_score:
                    continue

                chunk_obj = RAGSearchResultChunk(
                    chunk_id=chunk_id,
                    document_id=meta.get("document_id", ""),
                    filename=meta.get("filename", "unknown.pdf"),
                    page_number=meta.get("page_number", 1),
                    chunk_index=meta.get("chunk_index", i),
                    content=doc_text,
                    score=score,
                    metadata=meta,
                )
                results.append(chunk_obj)

        logger.info(
            "Retrieved %d chunks for query: '%s' (top_k=%d)",
            len(results),
            cleaned_query[:40],
            k,
            extra={"query": cleaned_query[:40], "result_count": len(results)},
        )

        return RAGSearchResponse(
            query=cleaned_query,
            total_results=len(results),
            results=results,
        )


_cached_retriever: Optional[RAGRetriever] = None


def get_retriever() -> RAGRetriever:
    """Singleton getter for RAGRetriever."""
    global _cached_retriever
    if _cached_retriever is None:
        _cached_retriever = RAGRetriever()
    return _cached_retriever
