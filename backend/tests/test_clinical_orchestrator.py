"""
tests/test_clinical_orchestrator.py

Unit and integration tests for ClinicalOrchestrator, evidence fusion, and alert prioritization.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from app.clinical.context import ClinicalContext
from app.clinical.fusion import (
    fuse_evidence,
    prioritize_risk_flags,
    prioritize_rule_findings,
)
from app.clinical.orchestrator import ClinicalOrchestrator
from app.core.exceptions import ResourceNotFoundError
from app.llm.schemas import (
    AssistantAuditRequest,
    AssistantEvaluationRequest,
    RiskFlagItem,
    RuleBasedFindingItem,
)
from app.reasoning.models import PatientProfile


@pytest.fixture
def sample_ckd_patient() -> PatientProfile:
    return PatientProfile(
        patient_id="patient_ckd_metformin_alert",
        name="John Doe",
        age=68,
        gender="male",
        vitals={"systolic_bp": 142.0, "diastolic_bp": 88.0, "heart_rate": 74.0, "bmi": 31.5},
        lab_results={"hba1c": 8.4, "egfr": 24.0, "serum_potassium": 4.8, "serum_creatinine": 2.6},
        conditions=["Type 2 Diabetes Mellitus", "Chronic Kidney Disease"],
        symptoms=["Fatigue", "Increased Thirst"],
        current_medications=["Metformin", "Amlodipine"],
    )


class TestClinicalFusion:
    def test_alert_prioritization(self) -> None:
        flags = [
            RiskFlagItem(flag="Mild Risk", severity="low", clinical_implication="Low concern"),
            RiskFlagItem(flag="Critical Alert", severity="critical", clinical_implication="Emergency"),
            RiskFlagItem(flag="Moderate Warning", severity="moderate", clinical_implication="Moderate"),
            RiskFlagItem(flag="High Concern", severity="high", clinical_implication="High"),
        ]
        prioritized = prioritize_risk_flags(flags)
        assert prioritized[0].severity == "critical"
        assert prioritized[1].severity == "high"
        assert prioritized[2].severity == "moderate"
        assert prioritized[3].severity == "low"

    def test_rule_findings_prioritization(self) -> None:
        findings = [
            RuleBasedFindingItem(rule_id="R1", rule_name="R1", category="diag", severity="low", conclusion="c", actionable_guidance="g", guideline_source="s"),
            RuleBasedFindingItem(rule_id="R2", rule_name="R2", category="contra", severity="critical", conclusion="c", actionable_guidance="g", guideline_source="s"),
        ]
        prioritized = prioritize_rule_findings(findings)
        assert prioritized[0].severity == "critical"
        assert prioritized[1].severity == "low"

    def test_evidence_fusion_sources(self) -> None:
        patient_facts = {"conditions": ["CKD"], "current_medications": ["Metformin"], "lab_results": {"egfr": 24}}
        rag_evidence = [{"chunk_id": "c1", "content": "Guideline text", "filename": "doc.pdf", "score": 0.88}]
        kg_facts = [{"source": "Metformin", "relationship": "CONTRAINDICATED_WITH", "target": "CKD"}]
        symbolic_findings = [{"rule_id": "RULE_1", "description": "Metformin contraindicated", "severity": "critical"}]
        critical_alerts = [{"fact_id": "ALERT_1", "name": "Critical Renal Alert", "explanation": "Metformin contraindicated"}]

        fused = fuse_evidence(patient_facts, rag_evidence, kg_facts, symbolic_findings, critical_alerts)
        source_types = {e.source_type for e in fused}
        assert "PATIENT_DATA" in source_types
        assert "RAG" in source_types
        assert "KNOWLEDGE_GRAPH" in source_types
        assert "SYMBOLIC_RULE" in source_types


class TestClinicalOrchestrator:
    def test_resolve_patient_by_id(self) -> None:
        orchestrator = ClinicalOrchestrator()
        patient = orchestrator.resolve_patient(patient_id="patient_ckd_metformin_alert")
        assert patient.patient_id == "patient_ckd_metformin_alert"
        assert "Metformin" in patient.current_medications

    def test_resolve_patient_missing_raises(self) -> None:
        orchestrator = ClinicalOrchestrator()
        with pytest.raises(ResourceNotFoundError):
            orchestrator.resolve_patient(patient_id=None, patient_profile=None)

    def test_orchestrator_status(self) -> None:
        orchestrator = ClinicalOrchestrator()
        status = orchestrator.get_status()
        assert "status" in status
        assert "components" in status
        assert "patient_service" in status["components"]
        assert "reasoning" in status["components"]
        assert "safety" in status["components"]

    def test_orchestrator_evaluate_offline_fallback(self, sample_ckd_patient: PatientProfile) -> None:
        orchestrator = ClinicalOrchestrator()
        req = AssistantEvaluationRequest(
            patient=sample_ckd_patient,
            question="Evaluate renal safety.",
            include_audit=True,
        )
        response = orchestrator.evaluate(req)
        assert response.patient_id == "patient_ckd_metformin_alert"
        assert len(response.rule_based_findings) >= 1
        assert len(response.risk_flags) >= 1
        assert response.metadata.rules_evaluated_count >= 10
        assert response.metadata.audit_data is not None

    def test_orchestrator_audit(self, sample_ckd_patient: PatientProfile) -> None:
        orchestrator = ClinicalOrchestrator()
        audit_req = AssistantAuditRequest(patient=sample_ckd_patient)
        audit_res = orchestrator.audit(audit_req)
        assert audit_res.patient_id == "patient_ckd_metformin_alert"
        assert audit_res.rules_evaluated >= 10
        assert len(audit_res.critical_alerts) >= 1
        assert audit_res.validation_status == "passed"
