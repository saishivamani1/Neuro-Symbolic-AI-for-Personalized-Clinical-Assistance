"""
app/api/routes/documents.py

POST /api/v1/documents/upload — Medical PDF ingestion endpoint.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.exceptions import InvalidDocumentError
from app.core.logging import get_logger
from app.rag.pipeline import get_document_pipeline
from app.rag.vector_store import get_vector_store
from app.schemas.rag import DocumentIngestionResult
from app.schemas.response import APIResponse

router = APIRouter()
logger = get_logger(__name__)


@router.post(
    "/documents/upload",
    response_model=APIResponse[DocumentIngestionResult],
    status_code=status.HTTP_200_OK,
    tags=["Documents"],
    summary="Upload and ingest a medical PDF into the RAG vector store",
)
async def upload_document(
    file: UploadFile = File(..., description="Medical PDF file to ingest"),
    save_to_disk: bool = Form(default=True, description="Save a copy of the PDF to the data/documents directory"),
) -> APIResponse[DocumentIngestionResult]:
    """Upload a medical guideline or clinical PDF document.

    The endpoint:
    1. Validates the uploaded file is a valid PDF.
    2. Extracts text page-by-page using PyMuPDF.
    3. Cleans clinical headers, footers, and formatting.
    4. Chunks text using page-aware splitting.
    5. Computes BGE-large dense vector embeddings.
    6. Stores chunks and metadata in ChromaDB.
    """
    if not file.filename:
        raise InvalidDocumentError(
            detail="No filename provided in upload request.",
            context={"filename": None},
        )

    if not file.filename.lower().endswith(".pdf"):
        raise InvalidDocumentError(
            detail=f"Only PDF files are supported. Received: '{file.filename}'",
            context={"filename": file.filename},
        )

    try:
        content = await file.read()
    except Exception as exc:
        raise InvalidDocumentError(
            detail=f"Failed to read uploaded file: {exc}",
            context={"filename": file.filename},
        ) from exc

    if len(content) == 0:
        raise InvalidDocumentError(
            detail=f"Uploaded file '{file.filename}' is empty (0 bytes).",
            context={"filename": file.filename},
        )

    # Optionally persist raw PDF to data/documents/
    if save_to_disk:
        settings = get_settings()
        doc_dir = Path(settings.documents_directory)
        doc_dir.mkdir(parents=True, exist_ok=True)
        saved_path = doc_dir / file.filename
        try:
            saved_path.write_bytes(content)
            logger.info("Saved copy of %s to %s", file.filename, saved_path)
        except Exception as exc:
            logger.warning("Could not save copy to disk: %s", str(exc))

    pipeline = get_document_pipeline()
    result = pipeline.ingest_bytes(data=content, filename=file.filename)

    return APIResponse(
        success=True,
        data=result,
    )


@router.delete(
    "/documents/{document_id}",
    response_model=APIResponse[dict],
    tags=["Documents"],
    summary="Delete a document and its chunks from the vector store",
)
async def delete_document(document_id: str) -> APIResponse[dict]:
    """Purge an ingested document and its vector embeddings from ChromaDB."""
    vector_store = get_vector_store()
    deleted_count = vector_store.delete_document(document_id=document_id)
    return APIResponse(
        success=True,
        data={"document_id": document_id, "deleted_chunks": deleted_count},
    )
