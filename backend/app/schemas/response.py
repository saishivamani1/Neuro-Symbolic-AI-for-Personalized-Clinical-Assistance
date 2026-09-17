"""
app/schemas/response.py

Shared response envelope schemas used across all API endpoints.

These schemas ensure every API response is consistent and carries
machine-readable status information alongside the payload.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

DataT = TypeVar("DataT")


class ErrorDetail(BaseModel):
    """Structured error payload returned by exception handlers."""

    error_code: str = Field(description="Machine-readable error code.")
    message: str = Field(description="Human-readable error description.")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional extra information about the error.",
    )


class APIResponse(BaseModel, Generic[DataT]):
    """Generic API response envelope.

    Every successful endpoint response is wrapped in this model so that
    the client always receives a consistent structure:

        {
            "success": true,
            "data": { ... }
        }
    """

    success: bool = Field(default=True)
    data: DataT | None = None


class HealthServiceStatus(BaseModel):
    """Status of a single backing service."""

    status: str = Field(description="'up', 'down', 'configured', 'unconfigured'.")
    detail: str | None = Field(default=None)


class HealthResponse(BaseModel):
    """Response model for GET /api/v1/health."""

    status: str = Field(description="Overall platform health: 'healthy' or 'degraded'.")
    version: str
    environment: str
    services: dict[str, HealthServiceStatus]
