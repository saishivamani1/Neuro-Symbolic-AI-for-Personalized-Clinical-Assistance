"""
app/reasoning/service.py

High-level Clinical Reasoning Service.

Coordinates patient working memory assembly (enriching with Knowledge Graph relations
and RAG retrieval) and executes the forward-chaining symbolic rule engine.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from app.core.logging import get_logger
from app.reasoning.engine import SymbolicRuleEngine, get_rule_engine
from app.reasoning.evidence import EvidenceAssembler, get_evidence_assembler
from app.reasoning.models import ClinicalRule, PatientProfile, ReasoningResult

logger = get_logger(__name__)


class ReasoningService:
    """Orchestrates symbolic clinical reasoning and evidence synthesis."""

    def __init__(
        self,
        rule_engine: Optional[SymbolicRuleEngine] = None,
        evidence_assembler: Optional[EvidenceAssembler] = None,
    ) -> None:
        self.engine = rule_engine or get_rule_engine()
        self.assembler = evidence_assembler or get_evidence_assembler()

    def evaluate_patient(
        self,
        patient: Union[PatientProfile, Dict[str, Any]],
        include_kg: bool = True,
        include_rag: bool = True,
    ) -> ReasoningResult:
        """Execute evidence-enriched clinical inference for a patient.

        Parameters
        ----------
        patient : Union[PatientProfile, Dict[str, Any]]
            Patient profile.
        include_kg : bool
            Whether to enrich with Knowledge Graph facts.
        include_rag : bool
            Whether to enrich with RAG evidence chunks.

        Returns
        -------
        ReasoningResult
            Symbolic reasoning output with fired rules, derived facts,
            and step-by-step explainability trace.
        """
        # 1. Assemble augmented working memory
        working_memory = self.assembler.assemble_patient_working_memory(
            patient=patient,
            include_kg=include_kg,
            include_rag=include_rag,
        )

        # 2. Execute forward-chaining symbolic reasoning
        result = self.engine.evaluate(working_memory)
        return result

    def list_rules(self) -> List[ClinicalRule]:
        """Return all currently active clinical rules."""
        return self.engine.rules

    def get_rule(self, rule_id: str) -> Optional[ClinicalRule]:
        """Fetch a specific rule by its rule_id or legacy alias."""
        from app.reasoning.rules import OLD_TO_NEW_RULE_MAP

        resolved_id = OLD_TO_NEW_RULE_MAP.get(rule_id, rule_id)
        for rule in self.engine.rules:
            if rule.rule_id in (resolved_id, rule_id):
                return rule
        return None

    def register_rule(self, rule: ClinicalRule) -> None:
        """Dynamically add or update a clinical rule."""
        self.engine.register_rule(rule)


_cached_reasoning_service: Optional[ReasoningService] = None


def get_reasoning_service() -> ReasoningService:
    """Singleton getter for ReasoningService."""
    global _cached_reasoning_service
    if _cached_reasoning_service is None:
        _cached_reasoning_service = ReasoningService()
    return _cached_reasoning_service
