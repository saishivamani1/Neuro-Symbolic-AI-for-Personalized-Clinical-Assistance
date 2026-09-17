"""
scripts/ingest_documents.py

CLI for incremental ingestion of medical PDF and JSON documents.

Usage:

    cd backend

    python scripts/ingest_documents.py \
        --dir ./data/documents

    python scripts/ingest_documents.py \
        --file ./data/documents/diseases.json

    python scripts/ingest_documents.py \
        --file ./data/documents/guideline.pdf

    python scripts/ingest_documents.py \
        --dir ./data/documents \
        --verbose
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


# ============================================================
# Add backend directory
# ============================================================

backend_dir = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if str(backend_dir) not in sys.path:

    sys.path.insert(
        0,
        str(backend_dir),
    )


from app.core.config import get_settings
from app.core.logging import (
    configure_logging,
    get_logger,
)

from app.rag.pipeline import (
    DocumentPipeline,
)

from app.rag.vector_store import (
    ChromaVectorStore,
)


logger = get_logger(__name__)


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Incrementally ingest medical "
            "PDF and JSON documents."
        )
    )

    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help=(
            "Directory containing medical "
            "PDF/JSON files."
        ),
    )

    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help=(
            "Single PDF or JSON file to ingest."
        ),
    )

    parser.add_argument(
        "--persist-dir",
        type=str,
        default=None,
        help=(
            "Custom ChromaDB persistence directory."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )

    args = parser.parse_args()

    configure_logging()

    settings = get_settings()

    print("=" * 75)
    print(
        "  Neuro-Symbolic Healthcare "
        "Intelligence Platform"
    )
    print(
        "  Stage 2: Incremental Medical "
        "Document Ingestion"
    )
    print("=" * 75)
    print()

    print(
        f"Embedding Model:   "
        f"{settings.embedding_model}"
    )

    print(
        f"Vector Store Path: "
        f"{args.persist_dir or settings.chroma_persist_directory}"
    )

    print(
        f"Collection Name:   "
        f"{settings.chroma_collection_name}"
    )

    print(
        f"Chunk Size / Lap:  "
        f"{settings.chunk_size} / "
        f"{settings.chunk_overlap}"
    )

    print(
        "Supported Types:   PDF, JSON"
    )

    print()

    # ========================================================
    # Create vector store
    # ========================================================

    if args.persist_dir:

        custom_store = (
            ChromaVectorStore(
                persist_directory=
                    args.persist_dir
            )
        )

        pipeline = DocumentPipeline(
            vector_store=custom_store
        )

    else:

        pipeline = DocumentPipeline()

    t0 = time.monotonic()

    # ========================================================
    # Single file
    # ========================================================

    if args.file:

        target_file = Path(
            args.file
        )

        if not target_file.exists():

            print(
                "[-] Error: Specified file "
                f"not found: {target_file}"
            )

            sys.exit(1)

        extension = (
            target_file.suffix.lower()
        )

        if extension not in {
            ".pdf",
            ".json",
        }:

            print(
                "[-] Error: Unsupported file type: "
                f"{target_file.suffix}"
            )

            print(
                "    Supported types: .pdf, .json"
            )

            sys.exit(1)

        print(
            f"[*] Processing: "
            f"{target_file.name}"
        )

        try:

            result = (
                pipeline.ingest_file(
                    target_file
                )
            )

            print()

            if result.status == "skipped":

                print(
                    "[=] SKIPPED: "
                    f"{result.filename}"
                )

                print(
                    "    Reason: "
                    f"{result.message}"
                )

            elif result.status == "success":

                print(
                    "[+] SUCCESS: "
                    f"{result.filename}"
                )

                print(
                    f"    Document ID: "
                    f"{result.document_id}"
                )

                print(
                    f"    Chunks:      "
                    f"{result.chunk_count}"
                )

                print(
                    f"    Hash:        "
                    f"{result.content_hash[:16]}..."
                )

            else:

                print(
                    f"[!] {result.status.upper()}: "
                    f"{result.filename}"
                )

                print(
                    f"    {result.message}"
                )

        except Exception as exc:

            print(
                "[-] FAILED: "
                f"{target_file.name}"
            )

            print(
                f"    {exc}"
            )

            sys.exit(1)

    # ========================================================
    # Directory
    # ========================================================

    else:

        target_dir = (
            args.dir
            or Path(
                settings.documents_directory
            )
        )

        target_dir = Path(
            target_dir
        )

        if not target_dir.exists():

            print(
                f"[*] Directory {target_dir} "
                "does not exist. Creating it..."
            )

            target_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

        print(
            f"[*] Scanning directory: "
            f"{target_dir}"
        )

        results = (
            pipeline.ingest_directory(
                target_dir
            )
        )

        if not results:

            print(
                "[!] No PDF or JSON documents "
                f"found in {target_dir}"
            )

            print(
                "    Place medical documents "
                "there and re-run."
            )

            sys.exit(0)

        print()

        print("-" * 90)

        print(
            f"{'Filename':<40} | "
            f"{'Chunks':<8} | "
            f"{'Status':<12}"
        )

        print("-" * 90)

        total_chunks = 0
        success_count = 0
        skipped_count = 0
        error_count = 0

        for result in results:

            total_chunks += (
                result.chunk_count
            )

            if result.status == "success":
                success_count += 1

            elif result.status == "skipped":
                skipped_count += 1

            elif result.status == "error":
                error_count += 1

            print(
                f"{result.filename[:40]:<40} | "
                f"{result.chunk_count:<8} | "
                f"{result.status:<12}"
            )

        print("-" * 90)

        print(
            f"Processed: {len(results)} file(s)"
        )

        print(
            f"New:       {success_count}"
        )

        print(
            f"Skipped:   {skipped_count}"
        )

        print(
            f"Errors:    {error_count}"
        )

        print(
            f"New chunks: {total_chunks}"
        )

    # ========================================================
    # Final statistics
    # ========================================================

    elapsed = round(
        time.monotonic() - t0,
        2,
    )

    store = pipeline.vector_store

    print()

    print(
        f"[*] Ingestion completed "
        f"in {elapsed}s."
    )

    print(
        "[*] Total items in ChromaDB "
        f"collection: {store.count()}"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()