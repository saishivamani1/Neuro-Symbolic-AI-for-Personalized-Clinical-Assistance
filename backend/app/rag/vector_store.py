"""
app/rag/vector_store.py

Persistent ChromaDB vector store wrapper for medical document embeddings
and metadata.

Supports:
- Collection creation
- Batch chunk insertion
- Similarity search
- Metadata filtering
- Document retrieval
- Document deletion
- Content-hash duplicate detection
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb

from app.core.config import get_settings
from app.core.exceptions import VectorStoreError
from app.core.logging import get_logger
from app.schemas.rag import DocumentChunk

logger = get_logger(__name__)


class ChromaVectorStore:
    """Manages persistent storage and querying of ChromaDB."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        settings = get_settings()

        self.persist_directory = (
            persist_directory
            or settings.chroma_persist_directory
        )

        self.collection_name = (
            collection_name
            or settings.chroma_collection_name
        )

        Path(self.persist_directory).mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory
            )

            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            logger.info(
                "ChromaDB initialized at %s "
                "(collection: %s, existing items: %d)",
                self.persist_directory,
                self.collection_name,
                self._collection.count(),
            )

        except Exception as exc:
            logger.error(
                "Failed to initialize ChromaDB: %s",
                str(exc),
            )

            raise VectorStoreError(
                detail=(
                    "Failed to initialize ChromaDB "
                    f"persistent store: {exc}"
                ),
                context={
                    "persist_directory":
                        self.persist_directory
                },
            ) from exc

    @property
    def collection(self):
        """Return the active Chroma collection."""
        return self._collection

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> int:
        """Add or upsert document chunks."""

        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                detail=(
                    f"Chunk count ({len(chunks)}) does not "
                    f"match embedding count ({len(embeddings)})."
                ),
                context={
                    "chunk_count": len(chunks),
                    "embedding_count": len(embeddings),
                },
            )

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks:

            ids.append(chunk.chunk_id)
            documents.append(chunk.text)

            meta = {
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "char_count": len(chunk.text),
            }

            for key, value in chunk.metadata.items():

                if isinstance(
                    value,
                    (str, int, float, bool),
                ):
                    meta[key] = value

                elif isinstance(value, list):
                    # Chroma metadata cannot directly store lists.
                    meta[key] = ", ".join(
                        str(item)
                        for item in value
                    )

            metadatas.append(meta)

        try:

            self._collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            logger.info(
                "Successfully added/upserted %d chunks "
                "to collection %s",
                len(chunks),
                self.collection_name,
            )

            return len(chunks)

        except Exception as exc:

            logger.error(
                "Failed to insert chunks into ChromaDB: %s",
                str(exc),
            )

            raise VectorStoreError(
                detail=(
                    "Failed to upsert chunks into "
                    f"vector store: {exc}"
                ),
                context={
                    "collection": self.collection_name,
                    "error": str(exc),
                },
            ) from exc

    def document_hash_exists(
        self,
        content_hash: str,
    ) -> bool:
        """
        Check whether a document with the same content hash
        already exists in ChromaDB.
        """

        if not content_hash:
            return False

        try:

            result = self._collection.get(
                where={
                    "content_hash": content_hash
                },
                include=["metadatas"],
            )

            ids = result.get("ids", [])

            return bool(ids)

        except Exception as exc:

            logger.error(
                "Failed duplicate check for hash %s: %s",
                content_hash[:16],
                str(exc),
            )

            raise VectorStoreError(
                detail=(
                    "Failed to check document "
                    f"duplicate status: {exc}"
                ),
                context={
                    "content_hash":
                        content_hash
                },
            ) from exc

    def get_document_by_hash(
        self,
        content_hash: str,
    ) -> Optional[Dict[str, Any]]:
        """Return stored metadata for a matching content hash."""

        if not content_hash:
            return None

        try:

            result = self._collection.get(
                where={
                    "content_hash": content_hash
                },
                include=["metadatas"],
            )

            ids = result.get("ids", [])
            metadatas = result.get(
                "metadatas",
                [],
            )

            if not ids:
                return None

            return {
                "chunk_id": ids[0],
                "metadata": (
                    metadatas[0]
                    if metadatas
                    else {}
                ),
            }

        except Exception as exc:

            raise VectorStoreError(
                detail=(
                    "Failed to retrieve document "
                    f"by hash: {exc}"
                ),
                context={
                    "content_hash":
                        content_hash
                },
            ) from exc

    def query(
        self,
        query_embeddings: List[List[float]],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute similarity search."""

        try:

            results = self._collection.query(
                query_embeddings=query_embeddings,
                n_results=min(
                    top_k,
                    max(1, self.count()),
                ),
                where=where,
                include=[
                    "documents",
                    "metadatas",
                    "distances",
                ],
            )

            return results

        except Exception as exc:

            logger.error(
                "ChromaDB similarity query failed: %s",
                str(exc),
            )

            raise VectorStoreError(
                detail=(
                    f"Similarity query failed: {exc}"
                ),
                context={
                    "where": where,
                    "top_k": top_k,
                },
            ) from exc

    def count(self) -> int:
        """Return total number of chunks."""

        try:
            return self._collection.count()

        except Exception as exc:

            raise VectorStoreError(
                detail=(
                    "Failed to get collection count: "
                    f"{exc}"
                )
            ) from exc

    def delete_document(
        self,
        document_id: str,
    ) -> int:
        """Delete all chunks belonging to a document."""

        try:

            before = self.count()

            self._collection.delete(
                where={
                    "document_id":
                        document_id
                }
            )

            after = self.count()

            deleted_count = max(
                0,
                before - after,
            )

            logger.info(
                "Deleted %d chunks for document %s",
                deleted_count,
                document_id,
            )

            return deleted_count

        except Exception as exc:

            logger.error(
                "Failed to delete document %s: %s",
                document_id,
                str(exc),
            )

            raise VectorStoreError(
                detail=(
                    f"Failed to delete document "
                    f"'{document_id}': {exc}"
                ),
                context={
                    "document_id":
                        document_id
                },
            ) from exc

    def get_document_chunks(
        self,
        document_id: str,
    ) -> Dict[str, Any]:
        """Fetch all chunks for a document."""

        try:

            return self._collection.get(
                where={
                    "document_id":
                        document_id
                },
                include=[
                    "documents",
                    "metadatas",
                ],
            )

        except Exception as exc:

            raise VectorStoreError(
                detail=(
                    "Failed to retrieve chunks "
                    f"for document '{document_id}': "
                    f"{exc}"
                ),
                context={
                    "document_id":
                        document_id
                },
            ) from exc


_cached_vector_store: Optional[
    ChromaVectorStore
] = None


def get_vector_store(
    persist_directory: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> ChromaVectorStore:
    """Singleton getter for ChromaVectorStore."""

    global _cached_vector_store

    if _cached_vector_store is None:

        _cached_vector_store = ChromaVectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

    return _cached_vector_store