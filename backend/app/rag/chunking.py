"""
app/rag/chunking.py

Page-aware medical text chunking using LangChain's RecursiveCharacterTextSplitter.

Preserves document ID, page numbers, chunk IDs, and associated metadata for every
generated chunk so that retrieval can trace evidence directly back to the exact
page and document source.
"""

from __future__ import annotations

from typing import List, Optional

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        # Fallback minimal splitter if langchain is not yet imported
        class RecursiveCharacterTextSplitter:  # type: ignore[no-redef]
            def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64, separators: Optional[List[str]] = None) -> None:
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap

            def split_text(self, text: str) -> List[str]:
                if not text:
                    return []
                chunks = []
                start = 0
                while start < len(text):
                    end = min(start + self.chunk_size, len(text))
                    chunks.append(text[start:end])
                    if end >= len(text):
                        break
                    start += self.chunk_size - self.chunk_overlap
                return chunks

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.rag import DocumentChunk, PageContent
from app.utils.ids import generate_chunk_id

logger = get_logger(__name__)


class MedicalChunker:
    """Splits extracted document pages into contextual chunks while maintaining page provenance."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
    ) -> None:
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        # Medical text split priority: paragraphs -> sentences -> clauses -> words
        self.separators = separators or ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_pages(
        self,
        pages: List[PageContent],
        document_id: str,
        filename: str,
    ) -> List[DocumentChunk]:
        """Split a list of pages into indexed chunks with preserved page metadata.

        Parameters
        ----------
        pages : List[PageContent]
            List of pages extracted from the PDF.
        document_id : str
            The document ID.
        filename : str
            Source document filename.

        Returns
        -------
        List[DocumentChunk]
            Ordered list of chunks with unique IDs and provenance metadata.
        """
        chunks: List[DocumentChunk] = []
        global_chunk_index = 0

        for page in pages:
            if not page.text or not page.text.strip():
                continue

            page_chunks = self.splitter.split_text(page.text)

            for chunk_text in page_chunks:
                clean_chunk = chunk_text.strip()
                if not clean_chunk:
                    continue

                chunk_id = generate_chunk_id(document_id, global_chunk_index)

                metadata = {
                    "document_id": document_id,
                    "filename": filename,
                    "page_number": page.page_number,
                    "chunk_index": global_chunk_index,
                    "char_count": len(clean_chunk),
                }

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        filename=filename,
                        page_number=page.page_number,
                        chunk_index=global_chunk_index,
                        text=clean_chunk,
                        metadata=metadata,
                    )
                )
                global_chunk_index += 1

        logger.info(
            "Chunked document %s into %d chunks across %d pages",
            document_id,
            len(chunks),
            len(pages),
            extra={
                "document_id": document_id,
                "chunk_count": len(chunks),
                "page_count": len(pages),
            },
        )

        return chunks
