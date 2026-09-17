"""
app/rag/pipeline.py

End-to-end medical document ingestion pipeline.

Supports:
- PDF documents
- Structured medical JSON datasets
- Incremental ingestion
- SHA-256 duplicate detection
- BGE-large embeddings
- Batched embedding
- Incremental ChromaDB indexing
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional, Union

from app.core.exceptions import DocumentProcessingError
from app.core.logging import get_logger

from app.rag.chunking import MedicalChunker

from app.rag.embeddings import (
    BGEEmbeddings,
    get_embeddings_service,
)

from app.rag.json_loader import JSONDiseaseLoader
from app.rag.loaders import PDFLoader

from app.rag.vector_store import (
    ChromaVectorStore,
    get_vector_store,
)

from app.schemas.rag import DocumentIngestionResult


logger = get_logger(__name__)


# ============================================================
# DEFAULT EMBEDDING BATCH SIZE
# ============================================================

DEFAULT_EMBEDDING_BATCH_SIZE = 64


# ============================================================
# DOCUMENT PIPELINE
# ============================================================

class DocumentPipeline:
    """
    Orchestrates ingestion of PDF and JSON medical documents.

    Large datasets are embedded and persisted in batches.
    """

    def __init__(
        self,
        loader: Optional[PDFLoader] = None,
        chunker: Optional[MedicalChunker] = None,
        embeddings_service: Optional[BGEEmbeddings] = None,
        vector_store: Optional[ChromaVectorStore] = None,
        json_loader: Optional[JSONDiseaseLoader] = None,
        embedding_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    ) -> None:

        self.loader = (
            loader
            or PDFLoader(
                apply_cleaning=True
            )
        )

        self.chunker = (
            chunker
            or MedicalChunker()
        )

        self.embeddings_service = (
            embeddings_service
            or get_embeddings_service()
        )

        self.vector_store = (
            vector_store
            or get_vector_store()
        )

        self.json_loader = (
            json_loader
            or JSONDiseaseLoader()
        )

        self.embedding_batch_size = (
            embedding_batch_size
        )

        if self.embedding_batch_size <= 0:

            raise ValueError(
                "embedding_batch_size must be "
                "greater than zero."
            )

    # ========================================================
    # DUPLICATE CHECK
    # ========================================================

    def _is_duplicate(
        self,
        content_hash: str,
    ) -> bool:

        return (
            self.vector_store.document_hash_exists(
                content_hash
            )
        )

    # ========================================================
    # BYTES HASH
    # ========================================================

    @staticmethod
    def _hash_bytes(
        data: bytes,
    ) -> str:

        return hashlib.sha256(
            data
        ).hexdigest()

    # ========================================================
    # BATCH EMBEDDING + STORAGE
    # ========================================================

    def _embed_and_store_chunks(
        self,
        chunks,
        filename: str,
    ) -> int:
        """
        Embed chunks in batches and immediately persist
        each batch to ChromaDB.
        """

        if not chunks:
            return 0

        total_chunks = len(chunks)

        chunk_texts = [
            chunk.text
            for chunk in chunks
        ]

        batch_count = (
            (
                total_chunks
                + self.embedding_batch_size
                - 1
            )
            // self.embedding_batch_size
        )

        logger.info(
            "Embedding %d chunks for %s "
            "using batch size %d "
            "(%d batches)",
            total_chunks,
            filename,
            self.embedding_batch_size,
            batch_count,
        )

        stored_count = 0

        def progress(
            completed: int,
            total: int,
            batch_number: int,
        ) -> None:

            logger.info(
                "Embedding progress: "
                "%d/%d chunks (batch %d/%d)",
                completed,
                total,
                batch_number,
                batch_count,
            )

            print(
                f"    Embedding progress: "
                f"{completed}/{total} "
                f"(batch {batch_number}/{batch_count})"
            )

        for (
            start,
            end,
            embeddings,
        ) in self.embeddings_service.embed_documents_batched(
            chunk_texts,
            batch_size=self.embedding_batch_size,
            progress_callback=progress,
        ):

            batch_chunks = chunks[start:end]

            self.vector_store.add_chunks(
                chunks=batch_chunks,
                embeddings=embeddings,
            )

            stored_count += len(batch_chunks)

            print(
                f"    Stored: "
                f"{stored_count}/{total_chunks} chunks"
            )

        logger.info(
            "Completed embedding and storage for %s: "
            "%d/%d chunks",
            filename,
            stored_count,
            total_chunks,
        )

        return stored_count

    # ========================================================
    # PDF INGESTION
    # ========================================================

    def ingest_bytes(
        self,
        data: bytes,
        filename: str = "document.pdf",
    ) -> DocumentIngestionResult:
        """
        Ingest a PDF from raw bytes.
        """

        logger.info(
            "Starting ingestion for uploaded file: "
            "%s (%d bytes)",
            filename,
            len(data),
        )

        content_hash = self._hash_bytes(
            data
        )

        # ----------------------------------------------------
        # Duplicate detection
        # ----------------------------------------------------

        if self._is_duplicate(
            content_hash
        ):

            existing = (
                self.vector_store.get_document_by_hash(
                    content_hash
                )
            )

            existing_metadata = (
                existing.get(
                    "metadata",
                    {},
                )
                if existing
                else {}
            )

            existing_document_id = (
                existing_metadata.get(
                    "document_id"
                )
                or "existing"
            )

            logger.info(
                "Skipping duplicate document: %s",
                filename,
            )

            return DocumentIngestionResult(
                document_id=existing_document_id,
                filename=filename,
                content_hash=content_hash,
                page_count=0,
                chunk_count=0,
                status="skipped",
                message=(
                    "Document already exists "
                    "in the vector store."
                ),
            )

        # ----------------------------------------------------
        # PDF loading
        # ----------------------------------------------------

        try:

            (
                doc_id,
                file_hash,
                pages,
            ) = self.loader.load_from_bytes(
                data,
                filename=filename,
            )

            if file_hash:
                content_hash = file_hash

        except Exception as exc:

            logger.error(
                "PDF loading failed for %s: %s",
                filename,
                str(exc),
            )

            raise DocumentProcessingError(
                detail=(
                    f"Failed to load PDF "
                    f"'{filename}': {exc}"
                ),
                context={
                    "filename": filename
                },
            ) from exc

        # ----------------------------------------------------
        # Second duplicate check
        # ----------------------------------------------------

        if self._is_duplicate(
            content_hash
        ):

            existing = (
                self.vector_store.get_document_by_hash(
                    content_hash
                )
            )

            metadata = (
                existing.get(
                    "metadata",
                    {},
                )
                if existing
                else {}
            )

            return DocumentIngestionResult(
                document_id=metadata.get(
                    "document_id",
                    doc_id,
                ),
                filename=filename,
                content_hash=content_hash,
                page_count=0,
                chunk_count=0,
                status="skipped",
                message=(
                    "Document already exists "
                    "in the vector store."
                ),
            )

        # ----------------------------------------------------
        # Chunk
        # ----------------------------------------------------

        chunks = self.chunker.chunk_pages(
            pages,
            document_id=doc_id,
            filename=filename,
        )

        if not chunks:

            raise DocumentProcessingError(
                detail=(
                    f"No chunks could be generated "
                    f"from document '{filename}'."
                ),
                context={
                    "document_id": doc_id,
                    "filename": filename,
                },
            )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        for chunk in chunks:

            chunk.metadata[
                "content_hash"
            ] = content_hash

            chunk.metadata[
                "document_type"
            ] = "pdf"

        # ----------------------------------------------------
        # Embedding + ChromaDB
        # ----------------------------------------------------

        self._embed_and_store_chunks(
            chunks=chunks,
            filename=filename,
        )

        logger.info(
            "Completed ingestion for %s: "
            "%d pages -> %d chunks",
            filename,
            len(pages),
            len(chunks),
        )

        return DocumentIngestionResult(
            document_id=doc_id,
            filename=filename,
            content_hash=content_hash,
            page_count=len(pages),
            chunk_count=len(chunks),
            status="success",
            message=(
                f"Successfully ingested "
                f"{len(chunks)} chunks across "
                f"{len(pages)} pages."
            ),
        )

    # ========================================================
    # FILE INGESTION
    # ========================================================

    def ingest_file(
        self,
        file_path: Union[str, Path],
    ) -> DocumentIngestionResult:

        path = Path(file_path)

        if not path.exists():

            raise FileNotFoundError(
                f"File not found: {path}"
            )

        extension = (
            path.suffix.lower()
        )

        if extension == ".pdf":

            data = path.read_bytes()

            return self.ingest_bytes(
                data=data,
                filename=path.name,
            )

        if extension == ".json":

            return self.ingest_json_file(
                path
            )

        raise ValueError(
            f"Unsupported file type: "
            f"{path.suffix}. "
            "Supported types: .pdf, .json"
        )

    # ========================================================
    # JSON INGESTION
    # ========================================================

    def ingest_json_file(
        self,
        file_path: Union[str, Path],
    ) -> DocumentIngestionResult:
        """
        Ingest structured medical disease JSON.

        The JSON loader creates section-level chunks.
        Embeddings and ChromaDB insertion are performed
        incrementally in batches.
        """

        path = Path(file_path)

        logger.info(
            "Starting JSON ingestion: %s",
            path.name,
        )

        print()
        print(
            f"[*] JSON ingestion: {path.name}"
        )

        # ----------------------------------------------------
        # Calculate hash before parsing
        # ----------------------------------------------------

        raw_data = path.read_bytes()

        content_hash = self._hash_bytes(
            raw_data
        )

        # ----------------------------------------------------
        # Duplicate check
        # ----------------------------------------------------

        if self._is_duplicate(
            content_hash
        ):

            existing = (
                self.vector_store.get_document_by_hash(
                    content_hash
                )
            )

            metadata = (
                existing.get(
                    "metadata",
                    {},
                )
                if existing
                else {}
            )

            document_id = metadata.get(
                "document_id",
                f"json_{content_hash[:24]}",
            )

            logger.info(
                "Skipping duplicate JSON: %s",
                path.name,
            )

            return DocumentIngestionResult(
                document_id=document_id,
                filename=path.name,
                content_hash=content_hash,
                page_count=0,
                chunk_count=0,
                status="skipped",
                message=(
                    "JSON dataset already exists "
                    "in the vector store."
                ),
            )

        # ----------------------------------------------------
        # Parse + create chunks
        # ----------------------------------------------------

        (
            document_id,
            loader_hash,
            chunks,
        ) = self.json_loader.load(
            path
        )

        content_hash = loader_hash

        if not chunks:

            raise DocumentProcessingError(
                detail=(
                    f"No chunks generated "
                    f"from JSON '{path.name}'."
                ),
                context={
                    "filename": path.name
                },
            )

        logger.info(
            "JSON parsed successfully: "
            "%d chunks generated from %s",
            len(chunks),
            path.name,
        )

        print(
            f"    Generated chunks: "
            f"{len(chunks)}"
        )

        # ----------------------------------------------------
        # Embedding + ChromaDB
        # ----------------------------------------------------

        stored_count = (
            self._embed_and_store_chunks(
                chunks=chunks,
                filename=path.name,
            )
        )

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        logger.info(
            "JSON ingestion completed: "
            "%s -> %d chunks",
            path.name,
            stored_count,
        )

        return DocumentIngestionResult(
            document_id=document_id,
            filename=path.name,
            content_hash=content_hash,
            page_count=0,
            chunk_count=stored_count,
            status="success",
            message=(
                f"Successfully ingested "
                f"{stored_count} disease "
                f"knowledge chunks."
            ),
        )

    # ========================================================
    # DIRECTORY INGESTION
    # ========================================================

    def ingest_directory(
        self,
        dir_path: Union[str, Path],
    ) -> List[DocumentIngestionResult]:
        """
        Scan directory for supported files.

        Supported:
        - .pdf
        - .PDF
        - .json
        - .JSON
        """

        path = Path(dir_path)

        if (
            not path.exists()
            or not path.is_dir()
        ):

            logger.warning(
                "Ingest directory does not exist "
                "or is not a directory: %s",
                path,
            )

            return []

        files = []

        files.extend(
            path.glob("*.pdf")
        )

        files.extend(
            path.glob("*.PDF")
        )

        files.extend(
            path.glob("*.json")
        )

        files.extend(
            path.glob("*.JSON")
        )

        files = sorted(
            files,
            key=lambda p: p.name.lower(),
        )

        logger.info(
            "Found %d supported file(s) "
            "in directory %s",
            len(files),
            path,
        )

        results = []

        for file_path in files:

            try:

                result = self.ingest_file(
                    file_path
                )

                results.append(
                    result
                )

            except Exception as exc:

                logger.error(
                    "Failed to ingest %s: %s",
                    file_path.name,
                    str(exc),
                )

                results.append(
                    DocumentIngestionResult(
                        document_id="error",
                        filename=file_path.name,
                        content_hash="",
                        page_count=0,
                        chunk_count=0,
                        status="error",
                        message=str(exc),
                    )
                )

        return results


# ============================================================
# SINGLETON
# ============================================================

_cached_pipeline: Optional[
    DocumentPipeline
] = None


def get_document_pipeline() -> DocumentPipeline:
    """
    Singleton getter for DocumentPipeline.
    """

    global _cached_pipeline

    if _cached_pipeline is None:

        _cached_pipeline = (
            DocumentPipeline()
        )

    return _cached_pipeline