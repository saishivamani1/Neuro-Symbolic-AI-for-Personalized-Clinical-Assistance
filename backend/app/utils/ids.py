"""
app/utils/ids.py

Deterministic and random identifier generation utilities.

Centralising ID generation means the format can be changed in one place
without hunting through the codebase.
"""

from __future__ import annotations

import hashlib
import uuid


def generate_request_id() -> str:
    """Return a new random UUID4 string for request tracing."""
    return str(uuid.uuid4())


def generate_document_id(filename: str, content_hash: str | None = None) -> str:
    """Generate a deterministic document identifier.

    If ``content_hash`` is provided the ID is derived from the file name
    and content, making it idempotent (uploading the same file twice
    produces the same ID and can be detected for deduplication).

    Parameters
    ----------
    filename:
        Original filename of the uploaded document.
    content_hash:
        Optional SHA-256 hex digest of the file content.

    Returns
    -------
    str
        Document ID prefixed with ``doc_``.
    """
    if content_hash:
        combined = f"{filename}:{content_hash}"
        digest = hashlib.sha256(combined.encode()).hexdigest()[:16]
        return f"doc_{digest}"
    return f"doc_{uuid.uuid4().hex[:16]}"


def generate_chunk_id(document_id: str, chunk_index: int) -> str:
    """Return a unique identifier for a document chunk.

    Parameters
    ----------
    document_id:
        Parent document identifier.
    chunk_index:
        Zero-based chunk position within the document.

    Returns
    -------
    str
        Chunk ID in the form ``{document_id}_chunk_{index:04d}``.
    """
    return f"{document_id}_chunk_{chunk_index:04d}"


def content_hash(data: bytes) -> str:
    """Return the SHA-256 hex digest of ``data``.

    Parameters
    ----------
    data:
        Raw file bytes.

    Returns
    -------
    str
        64-character hex digest.
    """
    return hashlib.sha256(data).hexdigest()
