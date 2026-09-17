"""
app/llm/schemas.py

Strongly typed Pydantic models for LangChain LLM structured outputs,
grounded evidence citations, clinical assistant requests, and responses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from app.reasoning.models import PatientProfile


class EvidenceSourceCitation(BaseModel):
    """Citation linking a clinical statement to its authoritative provenance."""

    type: Literal["rag", "rule", "knowledge_graph", "patient_fact"] = Field(
        ..., description="Provenance source type."
    )
    document: Optional[str] = Field(
        default=None, description="Document name for RAG evidence."
    )
    page: Optional[int] = Field(
        default=None, description="Document page number if available."
    )
    section: Optional[str] = Field(
        default=None, description="Document or guideline section name."
    )
    rule_id: Optional[str] = Field(
        default=None, description="Clinical rule ID for symbolic deductions."
    )
    entity: Optional[str] = Field(
        default=None, description="Entity or relationship for Knowledge Graph facts."
    )


class ClinicalFindingItem(BaseModel):
    """An individual clinical observation or abnormality."""

    finding: str = Field(..., description="Description of the finding.")
    category: str = Field(
        ...,
        description="Category: vital_abnormality, lab_abnormality, symptom, condition, medication.",
    )
    severity: Literal["critical", "high", "moderate", "low", "info"] = Field(
        default="moderate"
    )
    evidence: List[EvidenceSourceCitation] = Field(
        default_factory=list, description="Grounding source for this finding."
    )


class RiskFlagItem(BaseModel):
    """Identified clinical risk or safety concern."""

    flag: str = Field(..., description="Name of the risk condition or interaction.")
    severity: Literal["critical", "high", "moderate", "low"] = Field(...)
    clinical_implication: str = Field(
        ..., description="Potential clinical consequences or pathophysiological risks."
    )
    evidence: List[EvidenceSourceCitation] = Field(default_factory=list)


class RuleBasedFindingItem(BaseModel):
    """A deterministic deduction transferred directly from the Symbolic Reasoning Engine."""

    rule_id: str = Field(..., description="ID of the triggered symbolic rule.")
    rule_name: str = Field(..., description="Name of the triggered rule.")
    category: str = Field(..., description="Clinical category.")
    severity: str = Field(..., description="Clinical severity level.")
    conclusion: str = Field(..., description="Exact derived clinical fact or alert.")
    actionable_guidance: str = Field(
        ..., description="Actionable clinical guidance derived from the rule."
    )
    guideline_source: str = Field(
        ..., description="Authoritative guideline citation."
    )
    page_number: Optional[int] = Field(default=None)


class KnowledgeGraphFindingItem(BaseModel):
    """A structured relationship fact from the Medical Knowledge Graph."""

    entity_source: str = Field(..., description="Source entity name/type.")
    relationship: str = Field(..., description="Directed relationship type.")
    entity_target: str = Field(..., description="Target entity name/type.")
    clinical_significance: str = Field(
        ..., description="Clinical relevance to the patient context."
    )


class EvidenceItem(BaseModel):
    """Major clinical claim with explicit grounding citations."""

    claim: str = Field(..., description="The clinical synthesis statement.")
    sources: List[EvidenceSourceCitation] = Field(
        default_factory=list, description="Grounding citations for this claim."
    )


class UncertaintyItem(BaseModel):
    """Identified data gap or diagnostic ambiguity requiring clarification."""

    gap_type: str = Field(
        ...,
        description="Type: missing_laboratory_value, missing_history, unconfirmed_diagnosis, etc.",
    )
    description: str = Field(
        ..., description="Explanation of what information is missing."
    )
    impact_on_decision: str = Field(
        ..., description="How this data gap affects clinical decision support."
    )


class NextStepItem(BaseModel):
    """Actionable recommendation for the attending clinician."""

    action: str = Field(..., description="Specific recommended clinical action.")
    priority: Literal["immediate", "urgent", "routine", "monitoring"] = Field(...)
    rationale: str = Field(..., description="Clinical rationale for this step.")
    supporting_guideline: Optional[str] = Field(
        default=None, description="Guideline supporting this recommendation."
    )


class ClinicalAssistantResponse(BaseModel):
    """Strictly structured clinical assistance response synthesized by the LLM."""

    patient_id: str = Field(..., description="Identifier of the evaluated patient.")
    summary: str = Field(
        ...,
        description="Comprehensive, evidence-grounded clinical synthesis paragraph.",
    )
    clinical_findings: List[ClinicalFindingItem] = Field(
        default_factory=list, description="Extracted patient findings and abnormalities."
    )
    risk_flags: List[RiskFlagItem] = Field(
        default_factory=list, description="Identified clinical risks and safety alerts."
    )
    rule_based_findings: List[RuleBasedFindingItem] = Field(
        default_factory=list,
        description="Deterministic symbolic reasoning deductions (must preserve all critical alerts).",
    )
    symbolic_findings: List[RuleBasedFindingItem] = Field(
        default_factory=list,
        description="Deterministic symbolic reasoning deductions (synonym for rule_based_findings).",
    )
    knowledge_graph_findings: List[KnowledgeGraphFindingItem] = Field(
        default_factory=list,
        description="Relevant graph relationships (contraindications, indications).",
    )
    evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="Synthesized claims paired with supporting citations.",
    )
    uncertainties: List[UncertaintyItem] = Field(
        default_factory=list,
        description="Clinical data gaps or missing diagnostic tests.",
    )
    recommended_next_steps: List[NextStepItem] = Field(
        default_factory=list,
        description="Prioritized, evidence-grounded recommended actions.",
    )
    safety_notice: str = Field(
        default=(
            "This output is generated for clinical decision support and research purposes only. "
            "It is not an autonomous medical diagnosis or prescription. All recommendations "
            "must be verified by a licensed healthcare professional."
        )
    )


class AssistantEvaluationMetadata(BaseModel):
    """Execution metadata and performance metrics."""

    model: str = Field(..., description="LLM model identifier.")
    provider: str = Field(..., description="LLM provider (e.g., openai).")
    rag_results_count: int = Field(default=0)
    rules_evaluated_count: int = Field(default=0)
    rules_triggered_count: int = Field(default=0)
    critical_alerts_count: int = Field(default=0)
    execution_time_ms: float = Field(default=0.0)
    audit_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Deterministic reasoning trace and raw facts when include_audit=True.",
    )


class AssistantEvaluationRequest(BaseModel):
    """API payload for requesting clinical assistant evaluation."""

    patient_id: Optional[str] = Field(
        default=None,
        description="ID of a synthetic or pre-existing patient profile.",
    )
    patient: Optional[PatientProfile] = Field(
        default=None,
        description="Inline synthetic patient profile if evaluating custom clinical data.",
    )
    question: str = Field(
        default="What are the important clinical considerations and safety recommendations for this patient?",
        description="Clinical query or instruction for the assistant.",
    )
    include_audit: bool = Field(
        default=False,
        description="When True, exposes deterministic execution records and full reasoning traces.",
    )


class AssistantEvaluationResponse(BaseModel):
    """Complete API response wrapping structured LLM synthesis and metadata."""

    patient_id: str
    summary: str
    clinical_findings: List[ClinicalFindingItem] = Field(default_factory=list)
    risk_flags: List[RiskFlagItem] = Field(default_factory=list)
    rule_based_findings: List[RuleBasedFindingItem] = Field(default_factory=list)
    symbolic_findings: List[RuleBasedFindingItem] = Field(default_factory=list)
    knowledge_graph_findings: List[KnowledgeGraphFindingItem] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    uncertainties: List[UncertaintyItem] = Field(default_factory=list)
    recommended_next_steps: List[NextStepItem] = Field(default_factory=list)
    safety_notice: str
    metadata: AssistantEvaluationMetadata


class AssistantAuditRequest(BaseModel):
    """API payload for requesting deterministic clinical audit trail."""

    patient_id: Optional[str] = Field(default=None)
    patient: Optional[PatientProfile] = Field(default=None)
    question: Optional[str] = Field(
        default="What are the important clinical considerations for this patient?"
    )


class AssistantAuditResponse(BaseModel):
    """Deterministic audit record covering all reasoning, retrieval, and validation layers."""

    patient_id: str
    question: str
    patient_facts: Dict[str, Any]
    rag_evidence: List[Dict[str, Any]]
    knowledge_graph_facts: List[Dict[str, Any]]
    rules_evaluated: int
    rules_triggered: int
    triggered_rules: List[Dict[str, Any]]
    derived_facts: List[Dict[str, Any]]
    risk_flags: List[Dict[str, Any]]
    critical_alerts: List[Dict[str, Any]]
    provenance: List[Dict[str, Any]]
    llm_metadata: Dict[str, Any]
    validation_status: str
