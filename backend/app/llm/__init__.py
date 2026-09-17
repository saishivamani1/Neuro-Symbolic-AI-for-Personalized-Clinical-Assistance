"""
app/llm/__init__.py

LangChain + LLM Clinical Assistance Package.

Exports:
- LLMFactory, get_llm_factory
- ClinicalContextAssembler, get_context_assembler
- ClinicalGuardrailValidator
- build_clinical_assistant_chain
- ClinicalAssistantService, get_assistant_service
- Schemas: ClinicalAssistantResponse, AssistantEvaluationRequest, AssistantEvaluationResponse
"""

from app.llm.chains import build_clinical_assistant_chain
from app.llm.context import ClinicalContextAssembler, get_context_assembler
from app.llm.guardrails import ClinicalGuardrailValidator
from app.llm.models import LLMFactory, get_llm_factory
from app.llm.schemas import (
    AssistantEvaluationMetadata,
    AssistantEvaluationRequest,
    AssistantEvaluationResponse,
    ClinicalAssistantResponse,
    ClinicalFindingItem,
    EvidenceItem,
    EvidenceSourceCitation,
    KnowledgeGraphFindingItem,
    NextStepItem,
    RiskFlagItem,
    RuleBasedFindingItem,
    UncertaintyItem,
)
from app.llm.service import ClinicalAssistantService, get_assistant_service

__all__ = [
    "LLMFactory",
    "get_llm_factory",
    "ClinicalContextAssembler",
    "get_context_assembler",
    "ClinicalGuardrailValidator",
    "build_clinical_assistant_chain",
    "ClinicalAssistantService",
    "get_assistant_service",
    "ClinicalAssistantResponse",
    "AssistantEvaluationRequest",
    "AssistantEvaluationResponse",
    "AssistantEvaluationMetadata",
    "ClinicalFindingItem",
    "RiskFlagItem",
    "RuleBasedFindingItem",
    "KnowledgeGraphFindingItem",
    "EvidenceItem",
    "UncertaintyItem",
    "NextStepItem",
    "EvidenceSourceCitation",
]
