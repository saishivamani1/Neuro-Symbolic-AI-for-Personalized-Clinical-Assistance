"""
app/clinical/context.py

Structured clinical context model unifying multi-source inputs for the
neuro-symbolic clinical decision assistance workflow.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.reasoning.models import PatientProfile


class ClinicalContext(BaseModel):
    """Unified structured context containing all neural and symbolic evidence."""

    patient: PatientProfile = Field(
        ..., description="Patient clinical profile containing vitals, labs, and history."
    )
    question: str = Field(
        default="What are the important clinical considerations and safety recommendations for this patient?",
        description="User query or clinical evaluation question.",
    )
    patient_facts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extracted and normalized patient facts (conditions, labs, vitals, meds).",
    )
    rag_evidence: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Relevant clinical guideline chunks retrieved from ChromaDB with provenance.",
    )
    knowledge_graph_facts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Structured relations (contraindications, indications, symptoms) from Neo4j.",
    )
    symbolic_findings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Triggered deterministic rules and clinical assertions.",
    )
    derived_facts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Facts derived via forward chaining inference.",
    )
    risk_flags: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Clinical risk flags categorized by severity.",
    )
    critical_alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="High-priority safety and contraindication alerts that cannot be overridden.",
    )
    provenance: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Unified provenance list tracking source IDs and references.",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert clinical context to dictionary for LLM prompt ingestion."""
        return {
            "patient": self.patient.model_dump(),
            "question": self.question,
            "patient_facts": self.patient_facts,
            "rag_evidence": self.rag_evidence,
            "knowledge_graph_facts": self.knowledge_graph_facts,
            "symbolic_findings": self.symbolic_findings,
            "derived_facts": self.derived_facts,
            "risk_flags": self.risk_flags,
            "critical_alerts": self.critical_alerts,
            "provenance": self.provenance,
        }
