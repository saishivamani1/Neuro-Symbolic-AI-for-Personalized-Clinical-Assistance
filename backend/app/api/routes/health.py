"""
app/api/routes/health.py

Health-check endpoint — GET /api/v1/health

Returns the operational status of:
  * The FastAPI application itself
  * ChromaDB (vector store)
  * Neo4j (knowledge graph)
  * LLM provider (configuration check only — no live call)

This endpoint is intentionally lightweight so that container
orchestrators (Kubernetes liveness/readiness probes, Docker
health checks) can poll it rapidly without side-effects.
"""

from __future__ import annotations

import time

from fastapi import APIRouter

from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.response import HealthResponse, HealthServiceStatus

router = APIRouter()
logger = get_logger(__name__)


def _check_vector_store() -> HealthServiceStatus:
    """Probe ChromaDB by checking the vector store singleton."""
    try:
        from app.rag.vector_store import get_vector_store  # noqa: PLC0415

        store = get_vector_store()
        store.client.list_collections()
        return HealthServiceStatus(status="up")
    except ImportError:
        return HealthServiceStatus(
            status="down", detail="chromadb package not installed"
        )
    except Exception as exc:  # noqa: BLE001
        return HealthServiceStatus(status="down", detail=str(exc))


def _check_knowledge_graph() -> HealthServiceStatus:
    """Probe Neo4j with a lightweight connectivity check."""
    try:
        from app.knowledge_graph.connection import get_neo4j_manager  # noqa: PLC0415

        manager = get_neo4j_manager()
        if manager.verify_connectivity():
            return HealthServiceStatus(status="up")
        return HealthServiceStatus(
            status="down", detail="Could not establish connection to Neo4j"
        )
    except ImportError:
        return HealthServiceStatus(
            status="down", detail="neo4j package not installed"
        )
    except Exception as exc:  # noqa: BLE001
        return HealthServiceStatus(status="down", detail=str(exc))


def _check_llm() -> HealthServiceStatus:
    """Check whether the LLM is configured (no live API call)."""
    settings = get_settings()
    provider = settings.llm_provider

    if provider == "openai":
        if not settings.openai_api_key:
            return HealthServiceStatus(
                status="unconfigured", detail="OPENAI_API_KEY not set"
            )
        return HealthServiceStatus(status="configured", detail=f"provider={provider}, model={settings.llm_model}")

    if provider == "groq":
        api_key = settings.groq_api_key or settings.openai_api_key
        if not api_key:
            return HealthServiceStatus(
                status="unconfigured", detail="GROQ_API_KEY not set"
            )
        return HealthServiceStatus(status="configured", detail=f"provider={provider}, model={settings.groq_model}")

    if provider == "ollama":
        return HealthServiceStatus(
            status="configured",
            detail=f"provider={provider}, model={settings.ollama_model}, base_url={settings.ollama_base_url}",
        )

    if provider == "huggingface":
        if not settings.hf_api_key:
            return HealthServiceStatus(
                status="unconfigured", detail="HF_API_KEY not set"
            )
        return HealthServiceStatus(status="configured", detail=f"provider={provider}, model={settings.hf_model}")

    if provider in ("gemini", "google"):
        api_key = settings.google_api_key or settings.gemini_api_key
        if not api_key:
            return HealthServiceStatus(
                status="unconfigured", detail="GOOGLE_API_KEY / GEMINI_API_KEY not set"
            )
        return HealthServiceStatus(status="configured", detail=f"provider={provider}, model={settings.gemini_model}")

    return HealthServiceStatus(status="unconfigured", detail=f"Unknown provider: {provider}")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Platform health check",
    tags=["Health"],
)
async def health_check() -> HealthResponse:
    """Return the operational status of all platform services.

    This endpoint is **safe to call frequently** — it performs only
    lightweight probes and never modifies any state.
    """
    t0 = time.monotonic()
    settings = get_settings()

    vector_store_status = _check_vector_store()
    kg_status = _check_knowledge_graph()
    llm_status = _check_llm()

    all_up = all(
        s.status in ("up", "configured")
        for s in [vector_store_status, kg_status, llm_status]
    )
    overall = "healthy" if all_up else "degraded"

    latency_ms = (time.monotonic() - t0) * 1000
    logger.info(
        "Health check completed",
        extra={"overall": overall, "latency_ms": round(latency_ms, 2)},
    )

    return HealthResponse(
        status=overall,
        version=settings.app_version,
        environment=settings.app_env,
        services={
            "api": HealthServiceStatus(status="up"),
            "vector_store": vector_store_status,
            "knowledge_graph": kg_status,
            "llm": llm_status,
        },
    )
