"""
app/api/routes/assistant.py

Clinical Decision Support Assistant endpoints.

Endpoints:
- POST /api/v1/assistant/evaluate: Synthesize multi-source clinical evidence (Patient + RAG + KG + Symbolic Reasoning + LLM)
- POST /api/v1/assistant/audit: Retrieve complete deterministic audit trail and reasoning trace
- GET  /api/v1/assistant/status: Health and operational status of all neuro-symbolic components
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Query, status

from app.clinical.orchestrator import get_clinical_orchestrator
from app.core.logging import get_logger
from app.llm.schemas import (
    AssistantAuditRequest,
    AssistantAuditResponse,
    AssistantEvaluationRequest,
    AssistantEvaluationResponse,
)
from app.schemas.response import APIResponse

router = APIRouter()
logger = get_logger(__name__)


@router.post(
    "/assistant/evaluate",
    response_model=APIResponse[AssistantEvaluationResponse],
    status_code=status.HTTP_200_OK,
    tags=["Clinical Assistant"],
    summary="Evaluate patient using LangChain LLM grounded by RAG, KG, and Symbolic Reasoning",
)
async def evaluate_with_assistant(
    request: AssistantEvaluationRequest,
    include_audit: bool = Query(
        default=False,
        description="Include deterministic execution audit records and full reasoning traces.",
    ),
) -> APIResponse[AssistantEvaluationResponse]:
    """Execute end-to-end neuro-symbolic clinical evaluation with structured LLM synthesis."""
    if include_audit:
        request.include_audit = True

    orchestrator = get_clinical_orchestrator()
    result = orchestrator.evaluate(request)
    return APIResponse(success=True, data=result)


@router.post(
    "/assistant/audit",
    response_model=APIResponse[AssistantAuditResponse],
    status_code=status.HTTP_200_OK,
    tags=["Clinical Assistant"],
    summary="Retrieve full deterministic audit trace of patient evaluation",
)
async def audit_assistant_evaluation(
    request: AssistantAuditRequest,
) -> APIResponse[AssistantAuditResponse]:
    """Return full deterministic audit record including RAG chunks, KG facts, rule traces, and validation status."""
    orchestrator = get_clinical_orchestrator()
    result = orchestrator.audit(request)
    return APIResponse(success=True, data=result)


@router.get(
    "/assistant/status",
    response_model=APIResponse[Dict[str, Any]],
    tags=["Clinical Assistant"],
    summary="Get operational status of all neuro-symbolic assistant components",
)
async def get_assistant_status() -> APIResponse[Dict[str, Any]]:
    """Return current operational status of patient_service, rag, knowledge_graph, reasoning, llm, and safety."""
    orchestrator = get_clinical_orchestrator()
    status_data = orchestrator.get_status()
    return APIResponse(success=True, data=status_data)
