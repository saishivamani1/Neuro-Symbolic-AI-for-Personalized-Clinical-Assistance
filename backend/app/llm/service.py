"""
app/llm/service.py

Clinical Assistant Service orchestrating RAG, Knowledge Graph, Symbolic Reasoning,
and LangChain LLM synthesis.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Union

from app.api.routes.patients import SAMPLE_PATIENTS
from app.core.config import get_settings
from app.core.exceptions import LLMError, ResourceNotFoundError
from app.core.logging import get_logger
from app.llm.chains import build_clinical_assistant_chain
from app.llm.context import ClinicalContextAssembler, get_context_assembler
from app.llm.models import LLMFactory, get_llm_factory
from app.llm.schemas import (
    AssistantEvaluationMetadata,
    AssistantEvaluationRequest,
    AssistantEvaluationResponse,
    ClinicalAssistantResponse,
    ClinicalFindingItem,
    EvidenceItem,
    EvidenceSourceCitation,
    NextStepItem,
    RiskFlagItem,
    RuleBasedFindingItem,
    UncertaintyItem,
)
from app.reasoning.models import PatientProfile

logger = get_logger(__name__)


class ClinicalAssistantService:
    """Orchestrates end-to-end neuro-symbolic clinical evaluation."""

    def __init__(
        self,
        context_assembler: Optional[ClinicalContextAssembler] = None,
        llm_factory: Optional[LLMFactory] = None,
    ) -> None:
        self.context_assembler = context_assembler or get_context_assembler()
        self.llm_factory = llm_factory or get_llm_factory()

    def _resolve_patient(
        self, request: AssistantEvaluationRequest
    ) -> PatientProfile:
        """Resolve PatientProfile from inline payload or registered sample ID."""
        if request.patient is not None:
            return request.patient

        if not request.patient_id:
            raise ResourceNotFoundError(
                detail="Either 'patient_id' or inline 'patient' object must be provided.",
                resource_type="Patient",
            )

        if request.patient_id in SAMPLE_PATIENTS:
            return SAMPLE_PATIENTS[request.patient_id]

        # Generate standard fallback patient if arbitrary ID is provided
        return PatientProfile(
            patient_id=request.patient_id,
            name=f"Patient {request.patient_id}",
            vitals={"systolic_bp": 142.0, "diastolic_bp": 90.0},
            lab_results={"egfr": 26.0, "hba1c": 8.4},
            conditions=["Type 2 Diabetes Mellitus"],
            current_medications=["Metformin"],
            symptoms=["Fatigue"],
        )

    def _build_deterministic_fallback_response(
        self, context: Dict[str, Any], error_msg: str
    ) -> ClinicalAssistantResponse:
        """Construct structured clinical response purely from deterministic symbolic findings."""
        patient_facts = context.get("patient_facts", {})
        pid = patient_facts.get("patient_id", "unknown_patient")
        symbolic = context.get("symbolic_reasoning", {})
        triggered_rules = symbolic.get("triggered_rules", [])
        derived_facts = symbolic.get("derived_facts", [])
        critical_alerts = symbolic.get("critical_alerts", [])

        # Build rule-based findings
        rule_findings: list[RuleBasedFindingItem] = []
        for r in triggered_rules:
            rule_findings.append(
                RuleBasedFindingItem(
                    rule_id=r.get("rule_id", ""),
                    rule_name=r.get("name", ""),
                    category=r.get("category", ""),
                    severity=r.get("severity", ""),
                    conclusion=r.get("description", ""),
                    actionable_guidance=r.get("description", ""),
                    guideline_source=r.get("guideline_source", ""),
                    page_number=r.get("page_number"),
                )
            )

        # Build risk flags
        risk_flags: list[RiskFlagItem] = []
        for alert in critical_alerts:
            prov = alert.get("provenance", {})
            risk_flags.append(
                RiskFlagItem(
                    flag=alert.get("name", "Critical Alert"),
                    severity="critical",
                    clinical_implication=alert.get("explanation", ""),
                    evidence=[
                        EvidenceSourceCitation(
                            type="rule",
                            rule_id=alert.get("supporting_rules", [""])[0],
                            document=prov.get("guideline"),
                            page=prov.get("page"),
                        )
                    ],
                )
            )

        summary = (
            f"Deterministic clinical evaluation for patient {pid}. "
            f"Triggered {len(triggered_rules)} evidence-grounded rules with {len(critical_alerts)} critical alerts. "
            f"[Note: LLM synthesis generation offline/unconfigured ({error_msg})]."
        )

        return ClinicalAssistantResponse(
            patient_id=pid,
            summary=summary,
            clinical_findings=[
                ClinicalFindingItem(
                    finding=f"Patient {pid} with {len(patient_facts.get('conditions', []))} conditions and {len(patient_facts.get('current_medications', []))} medications.",
                    category="condition",
                    severity="info",
                )
            ],
            risk_flags=risk_flags,
            rule_based_findings=rule_findings,
            knowledge_graph_findings=[],
            evidence=[
                EvidenceItem(
                    claim=alert.get("explanation", ""),
                    sources=[
                        EvidenceSourceCitation(
                            type="rule",
                            rule_id=alert.get("supporting_rules", [""])[0],
                        )
                    ],
                )
                for alert in critical_alerts
            ],
            uncertainties=[
                UncertaintyItem(
                    gap_type="llm_offline",
                    description=f"LLM generation was skipped: {error_msg}",
                    impact_on_decision="Response contains purely deterministic rule engine deductions.",
                )
            ],
            recommended_next_steps=[
                NextStepItem(
                    action="Review deterministic symbolic findings and critical safety alerts.",
                    priority="immediate" if critical_alerts else "routine",
                    rationale="Verified clinical rules triggered for patient profile.",
                )
            ],
        )

    def evaluate_patient(
        self, request: AssistantEvaluationRequest
    ) -> AssistantEvaluationResponse:
        """Run full neuro-symbolic clinical assistant pipeline."""
        t0 = time.monotonic()
        settings = get_settings()

        # 1. Resolve patient profile
        patient = self._resolve_patient(request)

        # 2. Assemble context (RAG + KG + Symbolic Reasoning)
        context = self.context_assembler.assemble(
            patient=patient,
            include_rag=True,
            include_kg=True,
        )

        symbolic = context.get("symbolic_reasoning", {})
        rules_eval = symbolic.get("rules_evaluated_count", 0)
        rules_trig = symbolic.get("triggered_rules_count", 0)
        crit_alerts = len(symbolic.get("critical_alerts", []))
        rag_count = len(context.get("rag_evidence", []))

        # 3. Invoke LangChain LLM Chain or Fallback
        assistant_output: ClinicalAssistantResponse
        if not self.llm_factory.is_configured():
            logger.info(
                "LLM is unconfigured. Returning deterministic symbolic reasoning fallback."
            )
            assistant_output = self._build_deterministic_fallback_response(
                context=context, error_msg=f"{settings.llm_provider.upper()} unconfigured"
            )
        else:
            try:
                chain = build_clinical_assistant_chain(
                    model=self.llm_factory.get_model()
                )
                assistant_output = chain.invoke(
                    {"context": context, "question": request.question}
                )
            except Exception as exc:
                logger.error(
                    "LLM generation failed, degrading gracefully to symbolic fallback: %s",
                    str(exc),
                )
                assistant_output = self._build_deterministic_fallback_response(
                    context=context, error_msg=str(exc)
                )

        elapsed_ms = round((time.monotonic() - t0) * 1000, 2)

        # 4. Assemble audit data if requested
        audit_data: Optional[Dict[str, Any]] = None
        if request.include_audit:
            audit_data = {
                "triggered_rules": symbolic.get("triggered_rules", []),
                "derived_facts": symbolic.get("derived_facts", []),
                "critical_alerts": symbolic.get("critical_alerts", []),
                "reasoning_trace": symbolic.get("reasoning_trace", []),
                "knowledge_graph_facts": context.get("knowledge_graph_facts", []),
                "rag_evidence": context.get("rag_evidence", []),
            }

        metadata = AssistantEvaluationMetadata(
            model=self.llm_factory._resolve_model_name(),
            provider=settings.llm_provider,
            rag_results_count=rag_count,
            rules_evaluated_count=rules_eval,
            rules_triggered_count=rules_trig,
            critical_alerts_count=crit_alerts,
            execution_time_ms=elapsed_ms,
            audit_data=audit_data,
        )

        return AssistantEvaluationResponse(
            patient_id=assistant_output.patient_id,
            summary=assistant_output.summary,
            clinical_findings=assistant_output.clinical_findings,
            risk_flags=assistant_output.risk_flags,
            rule_based_findings=assistant_output.rule_based_findings,
            knowledge_graph_findings=assistant_output.knowledge_graph_findings,
            evidence=assistant_output.evidence,
            uncertainties=assistant_output.uncertainties,
            recommended_next_steps=assistant_output.recommended_next_steps,
            safety_notice=assistant_output.safety_notice,
            metadata=metadata,
        )


_cached_assistant_service: Optional[ClinicalAssistantService] = None


def get_assistant_service() -> ClinicalAssistantService:
    """Singleton getter for ClinicalAssistantService."""
    global _cached_assistant_service
    if _cached_assistant_service is None:
        _cached_assistant_service = ClinicalAssistantService()
    return _cached_assistant_service
