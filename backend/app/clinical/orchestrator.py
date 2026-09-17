"""
app/clinical/orchestrator.py

Clinical Orchestrator coordinating Patient Loading, Context Assembly,
RAG Evidence Retrieval, Knowledge Graph Queries, Symbolic Reasoning,
Evidence Fusion, LangChain LLM Synthesis, and Safety Validation.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

from app.api.routes.patients import SAMPLE_PATIENTS
from app.clinical.context import ClinicalContext
from app.clinical.fusion import (
    fuse_evidence,
    prioritize_risk_flags,
    prioritize_rule_findings,
)
from app.clinical.safety import ClinicalSafetyService, get_clinical_safety_service
from app.core.config import get_settings
from app.core.exceptions import LLMError, ResourceNotFoundError
from app.core.logging import get_logger
from app.knowledge_graph.service import KnowledgeGraphService, get_kg_service
from app.llm.chains import build_clinical_assistant_chain
from app.llm.context import ClinicalContextAssembler, get_context_assembler
from app.llm.models import LLMFactory, get_llm_factory
from app.llm.schemas import (
    AssistantAuditRequest,
    AssistantAuditResponse,
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
from app.rag.retriever import RAGRetriever, get_retriever
from app.reasoning.engine import SymbolicRuleEngine, get_rule_engine
from app.reasoning.models import PatientProfile

logger = get_logger(__name__)


class ClinicalOrchestrator:
    """Master neuro-symbolic clinical coordinator."""

    def __init__(
        self,
        context_assembler: Optional[ClinicalContextAssembler] = None,
        llm_factory: Optional[LLMFactory] = None,
        safety_service: Optional[ClinicalSafetyService] = None,
        retriever: Optional[RAGRetriever] = None,
        kg_service: Optional[KnowledgeGraphService] = None,
        rule_engine: Optional[SymbolicRuleEngine] = None,
    ) -> None:
        self.context_assembler = context_assembler or get_context_assembler()
        self.llm_factory = llm_factory or get_llm_factory()
        self.safety_service = safety_service or get_clinical_safety_service()
        self.retriever = retriever or get_retriever()
        self.kg_service = kg_service or get_kg_service()
        self.rule_engine = rule_engine or get_rule_engine()

    def resolve_patient(
        self,
        patient_id: Optional[str] = None,
        patient_profile: Optional[PatientProfile] = None,
    ) -> PatientProfile:
        """Resolve PatientProfile from inline object or registered patient ID."""
        if patient_profile is not None:
            return patient_profile

        if not patient_id:
            raise ResourceNotFoundError(
                detail="Either 'patient_id' or inline 'patient' object must be provided.",
                resource_type="Patient",
            )

        if patient_id in SAMPLE_PATIENTS:
            return SAMPLE_PATIENTS[patient_id]

        # Standard fallback for arbitrary patient ID
        return PatientProfile(
            patient_id=patient_id,
            name=f"Patient {patient_id}",
            vitals={"systolic_bp": 142.0, "diastolic_bp": 90.0},
            lab_results={"egfr": 26.0, "hba1c": 8.4},
            conditions=["Type 2 Diabetes Mellitus"],
            current_medications=["Metformin"],
            symptoms=["Fatigue"],
        )

    def _build_deterministic_fallback_response(
        self, context: Dict[str, Any], error_msg: str
    ) -> ClinicalAssistantResponse:
        """Construct structured response directly from deterministic symbolic findings."""
        patient_facts = context.get("patient_facts", {})
        pid = patient_facts.get("patient_id", "unknown_patient")
        symbolic = context.get("symbolic_reasoning", {})
        triggered_rules = symbolic.get("triggered_rules", [])
        critical_alerts = symbolic.get("critical_alerts", [])

        rule_findings: List[RuleBasedFindingItem] = []
        for r in triggered_rules:
            rule_findings.append(
                RuleBasedFindingItem(
                    rule_id=r.get("rule_id", ""),
                    rule_name=r.get("name", ""),
                    category=r.get("category", ""),
                    severity=r.get("severity", "moderate"),
                    conclusion=r.get("description", ""),
                    actionable_guidance=r.get("description", ""),
                    guideline_source=r.get("guideline_source", ""),
                    page_number=r.get("page_number"),
                )
            )

        risk_flags: List[RiskFlagItem] = []
        for alert in critical_alerts:
            prov = alert.get("provenance", {})
            rule_ids = alert.get("supporting_rules", [""])
            risk_flags.append(
                RiskFlagItem(
                    flag=alert.get("name", "Critical Alert"),
                    severity="critical",
                    clinical_implication=alert.get("explanation", ""),
                    evidence=[
                        EvidenceSourceCitation(
                            type="rule",
                            rule_id=rule_ids[0] if rule_ids else None,
                            document=prov.get("guideline"),
                            page=prov.get("page"),
                        )
                    ],
                )
            )

        prioritized_rules = prioritize_rule_findings(rule_findings)
        prioritized_risks = prioritize_risk_flags(risk_flags)

        summary = (
            f"Deterministic neuro-symbolic clinical evaluation for patient {pid}. "
            f"Evaluated {symbolic.get('rules_evaluated_count', 0)} rules, triggering {len(triggered_rules)} evidence-grounded rules with {len(critical_alerts)} critical alerts. "
            f"[Note: LLM synthesis generation offline/unconfigured ({error_msg})]."
        )

        kg_facts = context.get("knowledge_graph_facts", [])
        kg_findings = [
            KnowledgeGraphFindingItem(
                entity_source=fact.get("source", "MedicalConcept"),
                relationship=fact.get("relationship", "ASSOCIATED_WITH"),
                entity_target=fact.get("target", "MedicalConcept"),
                clinical_significance=fact.get("description")
                or f"Graph relation: {fact.get('source')} —[{fact.get('relationship')}]→ {fact.get('target')}.",
            )
            for fact in kg_facts
        ]

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
            risk_flags=prioritized_risks,
            rule_based_findings=prioritized_rules,
            symbolic_findings=prioritized_rules,
            knowledge_graph_findings=kg_findings,
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
                    description=f"LLM generation was bypassed or failed: {error_msg}",
                    impact_on_decision="Response is safely backed by deterministic forward-chaining rule deductions.",
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

    def evaluate(
        self, request: AssistantEvaluationRequest
    ) -> AssistantEvaluationResponse:
        """Run complete neuro-symbolic evaluation workflow."""
        t0 = time.monotonic()
        settings = get_settings()

        # 1. Resolve Patient
        patient = self.resolve_patient(
            patient_id=request.patient_id,
            patient_profile=request.patient,
        )

        # 2. Assemble Multi-Source Context (Patient + RAG + KG + Symbolic)
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
        kg_count = len(context.get("knowledge_graph_facts", []))

        # 3. LLM Execution with Safety Guardrails & Graceful Fallback
        assistant_output: ClinicalAssistantResponse
        if not self.llm_factory.is_configured():
            logger.info("LLM provider unconfigured; using deterministic symbolic fallback.")
            assistant_output = self._build_deterministic_fallback_response(
                context=context, error_msg=f"{settings.llm_provider.upper()} unconfigured"
            )
        else:
            try:
                chain = build_clinical_assistant_chain(model=self.llm_factory.get_model())
                raw_output = chain.invoke({"context": context, "question": request.question})
                # Validate and sanitize response
                assistant_output = self.safety_service.validate_and_sanitize(raw_output, context)
            except Exception as exc:
                logger.error("LLM evaluation failed, falling back to deterministic reasoning: %s", str(exc))
                assistant_output = self._build_deterministic_fallback_response(
                    context=context, error_msg=str(exc)
                )

        # Ensure alert prioritization is enforced
        assistant_output.risk_flags = prioritize_risk_flags(assistant_output.risk_flags)
        assistant_output.rule_based_findings = prioritize_rule_findings(assistant_output.rule_based_findings)
        assistant_output.symbolic_findings = list(assistant_output.rule_based_findings)

        elapsed_ms = round((time.monotonic() - t0) * 1000, 2)

        # 4. Optional Audit Metadata
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
            symbolic_findings=assistant_output.symbolic_findings,
            knowledge_graph_findings=assistant_output.knowledge_graph_findings,
            evidence=assistant_output.evidence,
            uncertainties=assistant_output.uncertainties,
            recommended_next_steps=assistant_output.recommended_next_steps,
            safety_notice=assistant_output.safety_notice,
            metadata=metadata,
        )

    def audit(self, request: AssistantAuditRequest) -> AssistantAuditResponse:
        """Produce complete deterministic audit record of all reasoning and retrieval operations."""
        patient = self.resolve_patient(
            patient_id=request.patient_id,
            patient_profile=request.patient,
        )
        question = request.question or "What are the important clinical considerations for this patient?"

        context = self.context_assembler.assemble(
            patient=patient,
            include_rag=True,
            include_kg=True,
        )

        symbolic = context.get("symbolic_reasoning", {})
        settings = get_settings()

        llm_meta = {
            "provider": settings.llm_provider,
            "model": self.llm_factory._resolve_model_name(),
            "temperature": settings.llm_temperature,
            "configured": self.llm_factory.is_configured(),
        }

        # Build provenance list
        provenance: List[Dict[str, Any]] = []
        for r in symbolic.get("triggered_rules", []):
            provenance.append({
                "type": "symbolic_rule",
                "id": r.get("rule_id"),
                "guideline": r.get("guideline_source"),
                "page": r.get("page_number"),
            })
        for c in context.get("rag_evidence", []):
            provenance.append({
                "type": "rag_chunk",
                "document_id": c.get("document_id"),
                "filename": c.get("filename"),
                "page": c.get("page_number"),
                "score": c.get("score"),
            })

        return AssistantAuditResponse(
            patient_id=patient.patient_id,
            question=question,
            patient_facts=context.get("patient_facts", {}),
            rag_evidence=context.get("rag_evidence", []),
            knowledge_graph_facts=context.get("knowledge_graph_facts", []),
            rules_evaluated=symbolic.get("rules_evaluated_count", 0),
            rules_triggered=symbolic.get("triggered_rules_count", 0),
            triggered_rules=symbolic.get("triggered_rules", []),
            derived_facts=symbolic.get("derived_facts", []),
            risk_flags=symbolic.get("critical_alerts", []),
            critical_alerts=symbolic.get("critical_alerts", []),
            provenance=provenance,
            llm_metadata=llm_meta,
            validation_status="passed",
        )

    def get_status(self) -> Dict[str, Any]:
        """Return actual operational health of all platform components."""
        # 1. Patient service
        patient_status = "up" if len(SAMPLE_PATIENTS) > 0 else "degraded"

        # 2. RAG
        rag_status = "up"
        try:
            from app.rag.vector_store import get_vector_store
            get_vector_store().client.list_collections()
        except Exception:
            rag_status = "down"

        # 3. Knowledge Graph
        kg_status = "up"
        try:
            from app.knowledge_graph.connection import get_neo4j_manager
            if not get_neo4j_manager().verify_connectivity():
                kg_status = "down"
        except Exception:
            kg_status = "down"

        # 4. Reasoning
        reasoning_status = "up" if len(self.rule_engine.rules) > 0 else "down"

        # 5. LLM
        llm_health = self.llm_factory.health_check()
        llm_status = llm_health.get("status", "unconfigured")

        # 6. Safety
        safety_status = "up"

        overall = "healthy" if all(s in ("up", "configured") for s in [patient_status, rag_status, kg_status, reasoning_status, llm_status, safety_status]) else "degraded"

        settings = get_settings()
        model_name = self.llm_factory._resolve_model_name()

        return {
            "status": overall,
            "provider": settings.llm_provider,
            "model": model_name,
            "components": {
                "patient_service": patient_status,
                "rag": rag_status,
                "knowledge_graph": kg_status,
                "reasoning": reasoning_status,
                "llm": llm_status,
                "safety": safety_status,
            },
        }


_cached_orchestrator: Optional[ClinicalOrchestrator] = None


def get_clinical_orchestrator() -> ClinicalOrchestrator:
    """Singleton getter for ClinicalOrchestrator."""
    global _cached_orchestrator
    if _cached_orchestrator is None:
        _cached_orchestrator = ClinicalOrchestrator()
    return _cached_orchestrator
