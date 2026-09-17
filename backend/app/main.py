"""
app/main.py

FastAPI application factory for the Neuro-Symbolic Healthcare Intelligence
Platform.

Responsibilities
----------------
* Create the FastAPI application instance.
* Configure CORS.
* Register all API routers under /api/v1.
* Register global exception handlers that convert domain exceptions into
  structured JSON HTTP responses.
* Trigger logging configuration at startup.
* Expose lifespan events (startup / shutdown) for resource initialisation
  and cleanup.

Design note
-----------
Do NOT place business logic here.  This file is strictly the application
wiring layer.  All logic lives in dedicated service / route modules.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import (
    assistant,
    chat,
    documents,
    health,
    knowledge_graph,
    patients,
    rag,
)
from app.core.config import get_settings
from app.core.exceptions import NeuroSymbolicBaseError
from app.core.logging import configure_logging, get_logger

# --------------------------------------------------------------------------- #
# Initialise logging before anything else
# --------------------------------------------------------------------------- #
configure_logging()
logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Lifespan — startup / shutdown
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application-level resources across the full server lifetime.

    Everything before ``yield`` runs at startup; everything after runs at
    shutdown.
    """
    settings = get_settings()
    logger.info(
        "Starting %s v%s [%s]",
        settings.app_name,
        settings.app_version,
        settings.app_env,
    )

    # Future stages will initialise ChromaDB, Neo4j, and embedding model
    # connections here so that they are ready before the first request.

    yield  # ← server is running

    logger.info("Shutting down %s", settings.app_name)
    try:
        from app.knowledge_graph.connection import get_neo4j_manager

        get_neo4j_manager().close()
    except Exception as exc:
        logger.warning("Error closing Neo4j driver during shutdown: %s", str(exc))


# --------------------------------------------------------------------------- #
# Application factory
# --------------------------------------------------------------------------- #


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application.

    Returns
    -------
    FastAPI
        The fully configured application instance.
    """
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Neuro-Symbolic Healthcare Intelligence Platform — "
            "combining RAG, Medical Knowledge Graph, and Symbolic Rule-Based "
            "Reasoning with an LLM to produce explainable, evidence-grounded "
            "clinical decision support."
        ),
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------ #
    # CORS
    # ------------------------------------------------------------------ #
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],          # Tighten in production.
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ #
    # Request-ID middleware
    # ------------------------------------------------------------------ #
    @app.middleware("http")
    async def add_request_id(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        t0 = time.monotonic()
        response = await call_next(request)
        latency_ms = (time.monotonic() - t0) * 1000
        response.headers["X-Request-ID"] = request_id
        logger.debug(
            "HTTP %s %s → %s",
            request.method,
            request.url.path,
            response.status_code,
            extra={"request_id": request_id, "latency_ms": round(latency_ms, 2)},
        )
        return response

    # ------------------------------------------------------------------ #
    # Exception handlers
    # ------------------------------------------------------------------ #
    @app.exception_handler(NeuroSymbolicBaseError)
    async def domain_exception_handler(
        request: Request, exc: NeuroSymbolicBaseError
    ) -> JSONResponse:
        """Convert domain exceptions into structured JSON error responses."""
        logger.error(
            "Domain error: %s — %s",
            exc.error_code,
            exc.detail,
            extra={"error_code": exc.error_code, "context": exc.context},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "success": False,
                "error": {
                    "error_code": exc.error_code,
                    "message": exc.detail,
                    "context": exc.context,
                },
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Catch-all handler for unexpected exceptions."""
        logger.exception("Unhandled exception: %s", str(exc))
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": {
                    "error_code": "INTERNAL_ERROR",
                    "message": "An unexpected internal error occurred.",
                    "context": {},
                },
            },
        )

    # ------------------------------------------------------------------ #
    # Routers
    # ------------------------------------------------------------------ #
    API_PREFIX = "/api/v1"

    app.include_router(health.router, prefix=API_PREFIX)
    app.include_router(assistant.router, prefix=API_PREFIX)
    app.include_router(chat.router, prefix=API_PREFIX)
    app.include_router(documents.router, prefix=API_PREFIX)
    app.include_router(patients.router, prefix=API_PREFIX)
    app.include_router(knowledge_graph.router, prefix=API_PREFIX)
    app.include_router(rag.router, prefix=API_PREFIX)

    # ------------------------------------------------------------------ #
    # Root redirect
    # ------------------------------------------------------------------ #
    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse(
            content={
                "name": settings.app_name,
                "version": settings.app_version,
                "docs": "/api/docs",
                "health": "/api/v1/health",
            }
        )

    return app


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
app = create_app()
