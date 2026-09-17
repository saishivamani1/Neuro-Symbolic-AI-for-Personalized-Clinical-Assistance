"""
app/reasoning/__init__.py

Symbolic Clinical Reasoning Engine Package.

Exports:
- PatientProfile, ClinicalRule, RuleCondition, DerivedFact, ReasoningTraceStep, ReasoningResult
- RuleCategory, RuleSeverity, RuleOperator, ConditionMatchDetail
- DEFAULT_CLINICAL_RULES
- SymbolicRuleEngine, get_rule_engine
- EvidenceAssembler, get_evidence_assembler
- ReasoningService, get_reasoning_service
"""

from app.reasoning.engine import SymbolicRuleEngine, get_rule_engine
from app.reasoning.evidence import EvidenceAssembler, get_evidence_assembler
from app.reasoning.models import (
    ClinicalRule,
    ConditionMatchDetail,
    DerivedFact,
    PatientProfile,
    ReasoningResult,
    ReasoningTraceStep,
    RuleCategory,
    RuleCondition,
    RuleOperator,
    RuleSeverity,
)
from app.reasoning.rules import DEFAULT_CLINICAL_RULES
from app.reasoning.service import ReasoningService, get_reasoning_service

__all__ = [
    "PatientProfile",
    "ClinicalRule",
    "RuleCondition",
    "DerivedFact",
    "ReasoningTraceStep",
    "ReasoningResult",
    "RuleCategory",
    "RuleSeverity",
    "RuleOperator",
    "ConditionMatchDetail",
    "DEFAULT_CLINICAL_RULES",
    "SymbolicRuleEngine",
    "get_rule_engine",
    "EvidenceAssembler",
    "get_evidence_assembler",
    "ReasoningService",
    "get_reasoning_service",
]
