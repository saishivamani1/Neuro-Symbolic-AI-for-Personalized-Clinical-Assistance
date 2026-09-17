"""
scripts/build_knowledge_graph.py

Standalone CLI script: initialize Neo4j schema and seed medical knowledge graph.

Usage:
    cd backend
    python scripts/build_knowledge_graph.py [--verify-only]

Requirements:
    Neo4j must be running (e.g. via `docker compose up -d neo4j`).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.knowledge_graph.builder import seed_sample_knowledge_graph
from app.knowledge_graph.connection import get_neo4j_manager
from app.knowledge_graph.service import get_kg_service

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize constraints and seed synthetic Medical Knowledge Graph into Neo4j."
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify Neo4j connectivity and print graph stats without seeding.",
    )
    args = parser.parse_args()

    configure_logging()
    settings = get_settings()

    print("=" * 70)
    print("  Neuro-Symbolic Healthcare Intelligence Platform")
    print("  Stage 3: Medical Knowledge Graph Seeder (Neo4j)")
    print("=" * 70)
    print()
    print(f"Neo4j URI:      {settings.neo4j_uri}")
    print(f"Neo4j Database: {settings.neo4j_database}")
    print(f"Neo4j Username: {settings.neo4j_username}")
    print()

    manager = get_neo4j_manager()

    print("[*] Testing Neo4j connectivity...")
    if not manager.verify_connectivity():
        print("[-] Error: Could not connect to Neo4j database.")
        print("    Ensure Neo4j is running (e.g. `docker compose up -d neo4j`)")
        print("    and verify your credentials in .env.")
        sys.exit(1)

    print("[+] Neo4j connection: OK")
    print()

    kg_service = get_kg_service()

    if args.verify_only:
        stats = kg_service.get_stats()
        print("[*] Current Knowledge Graph Statistics:")
        print(f"    Total Entities:      {stats.total_entities}")
        print(f"    Total Relationships: {stats.total_relationships}")
        print("    Entities by Type:")
        for k, v in stats.entity_counts_by_type.items():
            print(f"      {k}: {v}")
        print("=" * 70)
        sys.exit(0)

    print("[*] Initializing constraints and seeding synthetic medical ontology...")
    t0 = time.monotonic()
    result = seed_sample_knowledge_graph(service=kg_service)
    elapsed = round(time.monotonic() - t0, 2)

    print()
    print("Entities created:")
    for ent_type, count in result["entities_by_type"].items():
        print(f"  {ent_type}: {count}")

    print()
    print("Relationships created:")
    for rel_type, count in result["relationships_by_type"].items():
        print(f"  {rel_type}: {count}")

    stats = kg_service.get_stats()
    print()
    print(f"[*] Total Graph Entities:      {stats.total_entities}")
    print(f"[*] Total Graph Relationships: {stats.total_relationships}")
    print(f"[*] Time elapsed:              {elapsed}s")
    print()
    print("[+] Knowledge graph initialization: SUCCESS")
    print("=" * 70)


if __name__ == "__main__":
    main()
