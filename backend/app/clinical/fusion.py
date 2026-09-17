"""
app/clinical/fusion.py

Evidence Fusion and Deterministic Alert Prioritization.

Combines heterogeneous evidence from:
- PATIENT_DATA (Patient profile, labs, vitals)
- RAG (Guideline passages from ChromaDB vector store)
- KNOWLEDGE_GRAPH (Triples and traversals from Neo4j)
- SYMBOLIC_RULE (Inference deductions and contraindications)

Applies strict deterministic severity hierarchy:
CRITICAL > HIGH > MODERATE > LOW > INFO
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.llm.schemas import (
    ClinicalFindingItem,
    EvidenceItem,
    EvidenceSourceCitation,
    KnowledgeGraphFindingItem,
    RiskFlagItem,
    RuleBasedFindingItem,
)

logger = get_logger(__name__)

SEVERITY_RANKS = {
    "critical": 5,
    "high": 4,
    "moderate": 3,
    "low": 2,
    "info": 1,
}


class UnifiedEvidence(BaseModel):
    """Normalized evidence record from any subsystem."""

    source_type: Literal["PATIENT_DATA", "RAG", "KNOWLEDGE_GRAPH", "SYMBOLIC_RULE"]
    source_id: Optional[str] = None
    claim: str
    severity: Optional[Literal["critical", "high", "moderate", "low", "info"]] = None
    document: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    score: Optional[float] = None
    source_entity: Optional[str] = None
    relationship: Optional[str] = None
    target_entity: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


def prioritize_risk_flags(flags: List[RiskFlagItem]) -> List[RiskFlagItem]:
    """Sort risk flags deterministically by severity (CRITICAL -> HIGH -> MODERATE -> LOW)."""
    return sorted(
        flags,
        key=lambda f: SEVERITY_RANKS.get(f.severity.lower(), 0),
        reverse=True,
    )


def prioritize_rule_findings(
    findings: List[RuleBasedFindingItem],
) -> List[RuleBasedFindingItem]:
    """Sort symbolic findings deterministically by severity."""
    return sorted(
        findings,
        key=lambda r: SEVERITY_RANKS.get(r.severity.lower(), 0),
        reverse=True,
    )


def fuse_evidence(
    patient_facts: Dict[str, Any],
    rag_evidence: List[Dict[str, Any]],
    kg_facts: List[Dict[str, Any]],
    symbolic_findings: List[Dict[str, Any]],
    critical_alerts: List[Dict[str, Any]],
) -> List[UnifiedEvidence]:
    """Fuse multi-source evidence into a unified, traceable evidence collection."""
    fused: List[UnifiedEvidence] = []

    # 1. Critical Alerts & Symbolic Rules
    for alert in critical_alerts:
        prov = alert.get("provenance", {})
        rule_ids = alert.get("supporting_rules", [])
        rule_id = rule_ids[0] if rule_ids else None
        fused.append(
            UnifiedEvidence(
                source_type="SYMBOLIC_RULE",
                source_id=rule_id or alert.get("fact_id"),
                claim=alert.get("explanation") or alert.get("name", "Critical Alert"),
                severity="critical",
                document=prov.get("guideline"),
                page=prov.get("page"),
                metadata={"fact_id": alert.get("fact_id"), "rule_ids": rule_ids},
            )
        )

    for rule in symbolic_findings:
        sev = rule.get("severity", "moderate").lower()
        if sev not in SEVERITY_RANKS:
            sev = "moderate"
        fused.append(
            UnifiedEvidence(
                source_type="SYMBOLIC_RULE",
                source_id=rule.get("rule_id"),
                claim=rule.get("description") or rule.get("name", ""),
                severity=sev,  # type: ignore[arg-type]
                document=rule.get("guideline_source"),
                page=rule.get("page_number"),
                metadata={"category": rule.get("category")},
            )
        )

    # 2. Knowledge Graph Triples
    for fact in kg_facts:
        fused.append(
            UnifiedEvidence(
                source_type="KNOWLEDGE_GRAPH",
                source_id=f"{fact.get('source')}->{fact.get('relationship')}->{fact.get('target')}",
                claim=f"{fact.get('source')} is {fact.get('relationship')} with {fact.get('target')}",
                source_entity=fact.get("source"),
                relationship=fact.get("relationship"),
                target_entity=fact.get("target"),
                metadata={"source_label": fact.get("source_label"), "target_label": fact.get("target_label")},
            )
        )

    # 3. RAG Evidence
    for chunk in rag_evidence:
        fused.append(
            UnifiedEvidence(
                source_type="RAG",
                source_id=chunk.get("chunk_id"),
                claim=chunk.get("content", "")[:300],
                document=chunk.get("filename") or chunk.get("document_id"),
                page=chunk.get("page_number"),
                chunk_id=chunk.get("chunk_id"),
                score=chunk.get("score"),
                metadata={"document_id": chunk.get("document_id")},
            )
        )

    # 4. Patient Facts
    for cond in patient_facts.get("conditions", []):
        fused.append(
            UnifiedEvidence(
                source_type="PATIENT_DATA",
                source_id=f"condition_{cond}",
                claim=f"Diagnosed with {cond}",
                metadata={"category": "condition"},
            )
        )
    for med in patient_facts.get("current_medications", []):
        fused.append(
            UnifiedEvidence(
                source_type="PATIENT_DATA",
                source_id=f"medication_{med}",
                claim=f"Currently prescribed {med}",
                metadata={"category": "medication"},
            )
        )
    for lab, val in patient_facts.get("lab_results", {}).items():
        fused.append(
            UnifiedEvidence(
                source_type="PATIENT_DATA",
                source_id=f"lab_{lab}",
                claim=f"Lab value {lab}: {val}",
                metadata={"category": "lab_result"},
            )
        )

    return fused
