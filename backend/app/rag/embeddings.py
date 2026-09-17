"""
app/rag/embeddings.py

Embedding generation using BGE-large
(BAAI/bge-large-en-v1.5) or configured HuggingFace model.

Supports:
- SentenceTransformer embeddings
- CPU/CUDA device support
- Normalized embeddings
- Batched document embedding
- Progress reporting
- Singleton caching
"""

from __future__ import annotations

from typing import Callable, List, Optional

from app.core.config import get_settings
from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger

logger = get_logger(__name__)


# ============================================================
# Base class for LangChain compatibility
# ============================================================

try:
    from langchain_core.embeddings import Embeddings

except ImportError:

    class Embeddings:  # type: ignore[no-redef]

        def embed_documents(
            self,
            texts: List[str],
        ) -> List[List[float]]:
            raise NotImplementedError

        def embed_query(
            self,
            text: str,
        ) -> List[float]:
            raise NotImplementedError


# ============================================================
# BGE Embeddings
# ============================================================

class BGEEmbeddings(Embeddings):
    """
    BGE-large / SentenceTransformers embeddings wrapper.

    Supports normal document embedding as well as batched
    embedding for large medical datasets.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ) -> None:

        settings = get_settings()

        self.model_name = (
            model_name
            or settings.embedding_model
        )

        self.device = (
            device
            or settings.embedding_device
        )

        self.normalize_embeddings = (
            normalize_embeddings
        )

        self._model = None

    # ========================================================
    # MODEL
    # ========================================================

    def _get_model(self):
        """
        Lazily load SentenceTransformer model.
        """

        if self._model is None:

            logger.info(
                "Loading embedding model: %s on device: %s",
                self.model_name,
                self.device,
            )

            try:

                from sentence_transformers import (
                    SentenceTransformer,
                )

                self._model = SentenceTransformer(
                    model_name_or_path=self.model_name,
                    device=self.device,
                )

                logger.info(
                    "Embedding model %s loaded successfully",
                    self.model_name,
                )

            except Exception as exc:

                logger.error(
                    "Failed to load embedding model %s: %s",
                    self.model_name,
                    str(exc),
                )

                raise EmbeddingError(
                    detail=(
                        "Could not load embedding model "
                        f"'{self.model_name}': {exc}"
                    ),
                    context={
                        "model_name": self.model_name,
                        "device": self.device,
                    },
                ) from exc

        return self._model

    # ========================================================
    # DOCUMENT EMBEDDING
    # ========================================================

    def embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Compute embeddings for a list of document texts.

        This method preserves the original API behavior.
        For large datasets, use embed_documents_batched().
        """

        if not texts:
            return []

        try:

            model = self._get_model()

            embeddings = model.encode(
                texts,
                normalize_embeddings=(
                    self.normalize_embeddings
                ),
                show_progress_bar=False,
            )

            return embeddings.tolist()

        except Exception as exc:

            if isinstance(exc, EmbeddingError):
                raise

            logger.error(
                "Error generating document embeddings: %s",
                str(exc),
            )

            raise EmbeddingError(
                detail=(
                    "Error computing document embeddings: "
                    f"{exc}"
                ),
                context={
                    "count": len(texts),
                    "model": self.model_name,
                },
            ) from exc

    # ========================================================
    # BATCHED DOCUMENT EMBEDDING
    # ========================================================

    def embed_documents_batched(
        self,
        texts: List[str],
        batch_size: int = 64,
        progress_callback: Optional[
            Callable[[int, int, int], None]
        ] = None,
    ):
        """
        Generate embeddings in batches.

        Parameters
        ----------
        texts:
            Document texts to embed.

        batch_size:
            Number of texts passed to SentenceTransformer
            at one time.

        progress_callback:
            Optional callback receiving:

                completed_chunks,
                total_chunks,
                batch_number

        Yields
        ------
        tuple[int, int, List[List[float]]]

            start_index,
            end_index,
            embeddings

        Notes
        -----
        This is intentionally implemented as a generator so
        the pipeline can embed and immediately persist each
        batch instead of keeping every embedding in memory.
        """

        if not texts:
            return

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        total = len(texts)

        model = self._get_model()

        batch_number = 0

        for start in range(
            0,
            total,
            batch_size,
        ):

            end = min(
                start + batch_size,
                total,
            )

            batch_texts = texts[start:end]

            batch_number += 1

            try:

                logger.info(
                    "Embedding batch %d: "
                    "chunks %d-%d of %d",
                    batch_number,
                    start + 1,
                    end,
                    total,
                )

                embeddings = model.encode(
                    batch_texts,
                    normalize_embeddings=(
                        self.normalize_embeddings
                    ),
                    show_progress_bar=False,
                )

                batch_embeddings = (
                    embeddings.tolist()
                )

            except Exception as exc:

                logger.error(
                    "Embedding failed for batch %d "
                    "(chunks %d-%d): %s",
                    batch_number,
                    start + 1,
                    end,
                    str(exc),
                )

                raise EmbeddingError(
                    detail=(
                        f"Error computing embeddings "
                        f"for batch {batch_number}: {exc}"
                    ),
                    context={
                        "batch_number": batch_number,
                        "start": start,
                        "end": end,
                        "total": total,
                        "model": self.model_name,
                    },
                ) from exc

            completed = end

            if progress_callback:

                progress_callback(
                    completed,
                    total,
                    batch_number,
                )

            yield (
                start,
                end,
                batch_embeddings,
            )

    # ========================================================
    # QUERY EMBEDDING
    # ========================================================

    def embed_query(
        self,
        text: str,
    ) -> List[float]:
        """
        Compute embedding for a single query.

        BGE-large-en-v1.5 works directly without
        an instruction prefix.
        """

        if not text:
            return []

        try:

            model = self._get_model()

            embedding = model.encode(
                text,
                normalize_embeddings=(
                    self.normalize_embeddings
                ),
                show_progress_bar=False,
            )

            return embedding.tolist()

        except Exception as exc:

            if isinstance(exc, EmbeddingError):
                raise

            logger.error(
                "Error generating query embedding: %s",
                str(exc),
            )

            raise EmbeddingError(
                detail=(
                    "Error computing query embedding: "
                    f"{exc}"
                ),
                context={
                    "query": text[:50],
                    "model": self.model_name,
                },
            ) from exc


# ============================================================
# Singleton
# ============================================================

_cached_embeddings_instance: Optional[
    BGEEmbeddings
] = None


def get_embeddings_service(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
) -> BGEEmbeddings:
    """
    Singleton getter for BGEEmbeddings.
    """

    global _cached_embeddings_instance

    if _cached_embeddings_instance is None:

        _cached_embeddings_instance = (
            BGEEmbeddings(
                model_name=model_name,
                device=device,
            )
        )

    return _cached_embeddings_instance