"""
app/rag/loaders.py

PyMuPDF-based medical PDF document loader.

Extracts text from PDF documents page-by-page, preserves page number metadata,
applies medical text cleaning, and produces structured page objects.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Tuple, Union

try:
    import pymupdf as fitz
except ImportError:
    import fitz  # Fallback

from app.core.exceptions import DocumentProcessingError, InvalidDocumentError
from app.core.logging import get_logger
from app.schemas.rag import PageContent
from app.utils.ids import content_hash, generate_document_id
from app.utils.text import clean_medical_text

logger = get_logger(__name__)


class PDFLoader:
    """Loads and extracts page-aware medical text from PDF files or raw bytes."""

    def __init__(self, apply_cleaning: bool = True) -> None:
        self.apply_cleaning = apply_cleaning

    def load_from_bytes(
        self, data: bytes, filename: str = "document.pdf"
    ) -> Tuple[str, str, List[PageContent]]:
        """Load and extract pages from raw PDF bytes.

        Parameters
        ----------
        data : bytes
            Raw PDF file bytes.
        filename : str
            Original filename for ID generation and logging.

        Returns
        -------
        Tuple[str, str, List[PageContent]]
            (document_id, file_hash, list_of_pages)
        """
        if not data or len(data) == 0:
            raise InvalidDocumentError(
                detail=f"Uploaded file '{filename}' is empty.",
                context={"filename": filename, "size_bytes": 0},
            )

        # Quick magic bytes check for PDF
        if not data.startswith(b"%PDF-"):
            raise InvalidDocumentError(
                detail=f"File '{filename}' does not appear to be a valid PDF format.",
                context={"filename": filename},
            )

        file_hash = content_hash(data)
        doc_id = generate_document_id(filename=filename, content_hash=file_hash)

        try:
            doc = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:
            logger.error("Failed to open PDF %s: %s", filename, str(exc))
            raise DocumentProcessingError(
                detail=f"Could not open PDF file '{filename}': {exc}",
                context={"filename": filename, "error": str(exc)},
            ) from exc

        pages = self._extract_pages(doc, filename)
        doc.close()

        if not pages:
            raise DocumentProcessingError(
                detail=f"PDF '{filename}' contains no extractable text.",
                context={"filename": filename, "document_id": doc_id},
            )

        logger.info(
            "Extracted %d pages from %s (doc_id: %s)",
            len(pages),
            filename,
            doc_id,
            extra={"document_id": doc_id, "page_count": len(pages)},
        )

        return doc_id, file_hash, pages

    def load_from_path(
        self, file_path: Union[str, Path]
    ) -> Tuple[str, str, List[PageContent]]:
        """Load and extract pages from a file path on disk.

        Parameters
        ----------
        file_path : Union[str, Path]
            Path to the PDF file.

        Returns
        -------
        Tuple[str, str, List[PageContent]]
            (document_id, file_hash, list_of_pages)
        """
        path = Path(file_path)
        if not path.exists():
            raise InvalidDocumentError(
                detail=f"File not found at path: {path}",
                context={"path": str(path)},
            )

        if path.suffix.lower() != ".pdf":
            raise InvalidDocumentError(
                detail=f"Expected a PDF file, but received: {path.name}",
                context={"filename": path.name, "extension": path.suffix},
            )

        try:
            data = path.read_bytes()
        except Exception as exc:
            raise DocumentProcessingError(
                detail=f"Error reading file '{path.name}': {exc}",
                context={"filename": path.name, "error": str(exc)},
            ) from exc

        return self.load_from_bytes(data, filename=path.name)

    def _extract_pages(self, doc: fitz.Document, filename: str) -> List[PageContent]:
        """Extract and clean text from each page in a PyMuPDF document."""
        pages: List[PageContent] = []

        if doc.is_encrypted:
            raise DocumentProcessingError(
                detail=f"Encrypted PDFs are not supported: '{filename}'.",
                context={"filename": filename},
            )

        for page_idx in range(len(doc)):
            page_num = page_idx + 1
            try:
                page = doc.load_page(page_idx)
                raw_text = page.get_text("text") or ""
                
                cleaned_text = (
                    clean_medical_text(raw_text) if self.apply_cleaning else raw_text.strip()
                )

                if cleaned_text:
                    pages.append(
                        PageContent(
                            page_number=page_num,
                            text=cleaned_text,
                            char_count=len(cleaned_text),
                        )
                    )
            except Exception as exc:
                logger.warning(
                    "Error extracting page %d from %s: %s",
                    page_num,
                    filename,
                    str(exc),
                )

        return pages
