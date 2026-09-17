"""
app/core/exceptions.py

Domain-specific exception hierarchy for the platform.

All custom exceptions derive from NeuroSymbolicBaseError, which itself
inherits from Exception.  FastAPI exception handlers are registered in
main.py and convert these to structured JSON HTTP responses.

Exception design rules
-----------------------
* Carry a human-readable ``detail`` message.
* Carry an optional ``context`` dict for machine-readable extras.
* NEVER embed patient PII in exception messages.
"""

from __future__ import annotations

from typing import Any


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #


class NeuroSymbolicBaseError(Exception):
    """Root exception for the platform.

    All handlers that catch this class will produce a structured JSON
    error response rather than a plain 500.
    """

    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        detail: str = "An unexpected error occurred.",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(detail)
        self.detail = detail
        self.context: dict[str, Any] = context or {}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


class ConfigurationError(NeuroSymbolicBaseError):
    """Raised when the application is misconfigured (e.g. missing env var)."""

    status_code = 500
    error_code = "CONFIGURATION_ERROR"


class ResourceNotFoundError(NeuroSymbolicBaseError):
    """Raised when a requested resource (e.g., document, patient, rule) is not found."""

    status_code = 404
    error_code = "RESOURCE_NOT_FOUND"

    def __init__(
        self,
        detail: str = "Resource not found.",
        resource_type: str | None = None,
        resource_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        ctx = context or {}
        if resource_type:
            ctx["resource_type"] = resource_type
        if resource_id:
            ctx["resource_id"] = resource_id
        super().__init__(detail=detail, context=ctx)


# --------------------------------------------------------------------------- #
# Document / RAG
# --------------------------------------------------------------------------- #


class DocumentProcessingError(NeuroSymbolicBaseError):
    """Raised when a document cannot be parsed, chunked, or embedded."""

    status_code = 422
    error_code = "DOCUMENT_PROCESSING_ERROR"


class InvalidDocumentError(DocumentProcessingError):
    """Raised when the uploaded file is not a valid PDF or is empty."""

    status_code = 400
    error_code = "INVALID_DOCUMENT"


class EmbeddingError(NeuroSymbolicBaseError):
    """Raised when the embedding model fails to encode text."""

    status_code = 500
    error_code = "EMBEDDING_ERROR"


class VectorStoreError(NeuroSymbolicBaseError):
    """Raised when ChromaDB is unavailable or returns an unexpected error."""

    status_code = 503
    error_code = "VECTOR_STORE_UNAVAILABLE"


class RetrievalError(NeuroSymbolicBaseError):
    """Raised when the RAG retriever returns no evidence."""

    status_code = 404
    error_code = "RETRIEVAL_NO_RESULTS"


# --------------------------------------------------------------------------- #
# Knowledge Graph
# --------------------------------------------------------------------------- #


class KnowledgeGraphError(NeuroSymbolicBaseError):
    """Raised when Neo4j is unavailable or returns an unexpected error."""

    status_code = 503
    error_code = "KNOWLEDGE_GRAPH_UNAVAILABLE"


class GraphQueryError(NeuroSymbolicBaseError):
    """Raised when a Cypher query fails due to malformed data."""

    status_code = 422
    error_code = "GRAPH_QUERY_ERROR"


class MalformedGraphDataError(NeuroSymbolicBaseError):
    """Raised when entity/relationship data fails validation before ingestion."""

    status_code = 422
    error_code = "MALFORMED_GRAPH_DATA"


# --------------------------------------------------------------------------- #
# Symbolic Reasoning
# --------------------------------------------------------------------------- #


class RuleEngineError(NeuroSymbolicBaseError):
    """Raised when the symbolic rule engine encounters an unexpected failure."""

    status_code = 500
    error_code = "RULE_ENGINE_ERROR"


# --------------------------------------------------------------------------- #
# Patient
# --------------------------------------------------------------------------- #


class InvalidPatientDataError(NeuroSymbolicBaseError):
    """Raised when the patient schema fails validation."""

    status_code = 422
    error_code = "INVALID_PATIENT_DATA"


# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #


class LLMError(NeuroSymbolicBaseError):
    """Raised when the LLM provider returns an error or is unreachable."""

    status_code = 503
    error_code = "LLM_UNAVAILABLE"


class LLMConfigurationError(LLMError):
    """Raised when the LLM is not configured (e.g. missing API key)."""

    status_code = 500
    error_code = "LLM_CONFIGURATION_ERROR"
