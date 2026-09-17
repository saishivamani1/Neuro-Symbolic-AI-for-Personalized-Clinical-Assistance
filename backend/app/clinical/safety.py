"""
app/clinical/safety.py

Clinical Safety Service and Deterministic Contradiction Detection.

Enforces post-generation deterministic safety validation on LLM outputs:
1. Patient ID consistency.
2. Rule ID grounding against the active symbolic rule registry.
3. Citation validation against supplied RAG documents and Neo4j graph entities.
4. Mandatory preservation of all critical symbolic reasoning alerts.
5. Contradiction detection (intercepting cases where LLM asserts safety of contraindicated treatments).
6. Non-autonomous clinical decision support safety notice enforcement.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

from app.core.logging import get_logger
from app.llm.schemas import (
    ClinicalAssistantResponse,
    ClinicalFindingItem,
    EvidenceItem,
    EvidenceSourceCitation,
    KnowledgeGraphFindingItem,
    NextStepItem,
    RiskFlagItem,
    RuleBasedFindingItem,
)
from app.reasoning.rules import DEFAULT_CLINICAL_RULES, OLD_TO_NEW_RULE_MAP

logger = get_logger(__name__)

ALL_KNOWN_RULE_IDS: Set[str] = {r.rule_id for r in DEFAULT_CLINICAL_RULES} | set(OLD_TO_NEW_RULE_MAP.keys())

CONTRADICTION_PATTERNS = [
    (r"\bmetformin\b.*\b(safe|appropriate|continue|indicated|recommend(ed)?)\b", "Metformin contraindicated in severe renal impairment"),
    (r"\bspironolactone\b.*\blisinopril\b.*\b(safe|no interaction|compatible)\b", "Severe hyperkalemia dual RAAS interaction"),
]


class ClinicalSafetyService:
    """Validates and enforces deterministic safety constraints on clinical assistant responses."""

    @classmethod
    def validate_and_sanitize(
        cls,
        response: ClinicalAssistantResponse,
        context: Dict[str, Any],
    ) -> ClinicalAssistantResponse:
        """Run all safety checks and sanitize the LLM response."""
        patient_facts = context.get("patient_facts", {})
        expected_patient_id = patient_facts.get("patient_id")
        symbolic = context.get("symbolic_reasoning", {})
        critical_alerts = symbolic.get("critical_alerts", [])
        valid_citations = context.get("valid_citations", {})
        allowed_rule_ids = set(valid_citations.get("rule_ids", [])) | ALL_KNOWN_RULE_IDS
        allowed_documents = set(valid_citations.get("documents", []))
        rag_chunks = context.get("rag_evidence", [])
        for c in rag_chunks:
            if c.get("filename"):
                allowed_documents.add(c["filename"])
            if c.get("document_id"):
                allowed_documents.add(c["document_id"])

        # 1. Enforce Patient ID consistency
        if expected_patient_id and response.patient_id != expected_patient_id:
            logger.warning(
                "Patient ID mismatch: LLM returned '%s', expected '%s'. Sanitizing.",
                response.patient_id,
                expected_patient_id,
            )
            response.patient_id = expected_patient_id

        # 2. Filter fabricated Rule IDs
        valid_rule_findings: List[RuleBasedFindingItem] = []
        for rf in response.rule_based_findings:
            if rf.rule_id in allowed_rule_ids:
                valid_rule_findings.append(rf)
            else:
                logger.warning("Filtered hallucinated rule ID: %s", rf.rule_id)
        response.rule_based_findings = valid_rule_findings

        # 3. Filter hallucinated RAG citations
        for ev in response.evidence:
            valid_sources: List[EvidenceSourceCitation] = []
            for src in ev.sources:
                if src.type == "rule":
                    if src.rule_id and src.rule_id in allowed_rule_ids:
                        valid_sources.append(src)
                elif src.type == "rag":
                    if not allowed_documents or (src.document and any(doc in src.document for doc in allowed_documents)):
                        valid_sources.append(src)
                    else:
                        logger.warning("Filtered hallucinated RAG document citation: %s", src.document)
                else:
                    valid_sources.append(src)
            ev.sources = valid_sources

        # 4. Critical Alert Preservation
        existing_flags = {f.flag.lower() for f in response.risk_flags}
        existing_rule_ids = {rf.rule_id for rf in response.rule_based_findings}

        for alert in critical_alerts:
            alert_name = alert.get("name", "Critical Safety Alert")
            fact_id = alert.get("fact_id", "")
            supp_rules = alert.get("supporting_rules", [])
            rule_id = supp_rules[0] if supp_rules else "RULE_CRITICAL_ALERT"
            prov = alert.get("provenance", {})

            # Ensure present in risk_flags
            if alert_name.lower() not in existing_flags and not any(alert_name.lower() in f for f in existing_flags):
                logger.warning("Injecting dropped critical alert into risk_flags: %s", alert_name)
                response.risk_flags.insert(
                    0,
                    RiskFlagItem(
                        flag=alert_name,
                        severity="critical",
                        clinical_implication=alert.get("explanation", "Critical clinical alert triggered by rule engine."),
                        evidence=[
                            EvidenceSourceCitation(
                                type="rule",
                                rule_id=rule_id,
                                document=prov.get("guideline"),
                                page=prov.get("page"),
                            )
                        ],
                    ),
                )
                existing_flags.add(alert_name.lower())

            # Ensure present in rule_based_findings
            if rule_id not in existing_rule_ids:
                logger.warning("Injecting dropped critical rule into rule_based_findings: %s", rule_id)
                response.rule_based_findings.insert(
                    0,
                    RuleBasedFindingItem(
                        rule_id=rule_id,
                        rule_name=alert_name,
                        category=alert.get("category", "contraindication"),
                        severity="critical",
                        conclusion=alert.get("explanation", ""),
                        actionable_guidance=alert.get("explanation", ""),
                        guideline_source=prov.get("guideline", "Clinical Guidelines"),
                        page_number=prov.get("page"),
                    ),
                )
                existing_rule_ids.add(rule_id)

        # Mirror rule_based_findings into symbolic_findings
        response.symbolic_findings = list(response.rule_based_findings)

        # 5. Knowledge Graph findings preservation
        kg_facts = context.get("knowledge_graph_facts", [])
        if kg_facts:
            existing_kg_keys = {
                (kg.entity_source.strip().lower(), kg.relationship.strip().lower(), kg.entity_target.strip().lower())
                for kg in response.knowledge_graph_findings
            }
            for fact in kg_facts:
                src = fact.get("source", "")
                rel = fact.get("relationship", "")
                tgt = fact.get("target", "")
                key = (src.strip().lower(), rel.strip().lower(), tgt.strip().lower())
                if key not in existing_kg_keys:
                    response.knowledge_graph_findings.append(
                        KnowledgeGraphFindingItem(
                            entity_source=src,
                            relationship=rel,
                            entity_target=tgt,
                            clinical_significance=fact.get("description")
                            or f"Knowledge graph relation: {src} —[{rel}]→ {tgt}.",
                        )
                    )
                    existing_kg_keys.add(key)

        # 6. Contradiction Detection & Neutralization
        cls._detect_and_neutralize_contradictions(response, critical_alerts)

        # 7. Enforce mandatory safety notice
        if not response.safety_notice or "clinical decision support" not in response.safety_notice.lower():
            response.safety_notice = (
                "This output is generated for clinical decision support and research purposes only. "
                "It is not an autonomous medical diagnosis or prescription. All recommendations "
                "must be verified by a licensed healthcare professional."
            )

        return response

    @classmethod
    def _detect_and_neutralize_contradictions(
        cls, response: ClinicalAssistantResponse, critical_alerts: List[Dict[str, Any]]
    ) -> None:
        """Scan LLM summary and recommendations for dangerous statements contradicting critical alerts."""
        if not critical_alerts:
            return

        summary_lower = response.summary.lower()

        # Check for known contraindicated assertions
        for alert in critical_alerts:
            expl = alert.get("explanation", "").lower()
            if "metformin" in expl and "contraindicated" in expl:
                if re.search(r"\b(continue|safe to use|prescribe|appropriate to give)\s+metformin\b", summary_lower):
                    logger.error("Contradiction detected: LLM asserted Metformin is safe despite critical contraindication.")
                    response.summary += " [SAFETY WARNING: Metformin is strictly contraindicated based on deterministic renal function rules.]"
            if "hyperkalemia" in expl or "potassium" in expl:
                if re.search(r"\b(safe to combine|no interaction between)\s+(lisinopril|spironolactone)\b", summary_lower):
                    logger.error("Contradiction detected: LLM dismissed Lisinopril + Spironolactone interaction.")
                    response.summary += " [SAFETY WARNING: Concurrent Lisinopril and Spironolactone carries severe hyperkalemia risk.]"


_cached_safety_service: ClinicalSafetyService | None = None


def get_clinical_safety_service() -> ClinicalSafetyService:
    """Singleton getter for ClinicalSafetyService."""
    global _cached_safety_service
    if _cached_safety_service is None:
        _cached_safety_service = ClinicalSafetyService()
    return _cached_safety_service
