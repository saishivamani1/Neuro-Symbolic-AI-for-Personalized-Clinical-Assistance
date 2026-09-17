"""
app/clinical/__init__.py

Stage 6 Clinical Orchestration and Neuro-Symbolic Integration Package.

Exports:
- ClinicalOrchestrator, get_clinical_orchestrator
- ClinicalSafetyService, get_clinical_safety_service
- ClinicalContext
- UnifiedEvidence, fuse_evidence, prioritize_risk_flags, prioritize_rule_findings
"""

from app.clinical.context import ClinicalContext
from app.clinical.fusion import (
    UnifiedEvidence,
    fuse_evidence,
    prioritize_risk_flags,
    prioritize_rule_findings,
)
from app.clinical.orchestrator import (
    ClinicalOrchestrator,
    get_clinical_orchestrator,
)
from app.clinical.safety import (
    ClinicalSafetyService,
    get_clinical_safety_service,
)

__all__ = [
    "ClinicalOrchestrator",
    "get_clinical_orchestrator",
    "ClinicalSafetyService",
    "get_clinical_safety_service",
    "ClinicalContext",
    "UnifiedEvidence",
    "fuse_evidence",
    "prioritize_risk_flags",
    "prioritize_rule_findings",
]
