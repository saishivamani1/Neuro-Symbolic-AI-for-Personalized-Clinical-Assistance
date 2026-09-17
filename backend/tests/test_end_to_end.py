"""
tests/test_end_to_end.py

End-to-End Clinical Test Scenarios for Stage 6 Neuro-Symbolic Platform:
1. Scenario 1: patient_ckd_metformin_alert (CKD Stage 4 + Metformin Contraindication)
2. Scenario 2: patient_hyperkalemia_ddi (Hyperkalemia + Lisinopril/Spironolactone DDI)
3. Scenario 3: patient_healthy_normal (Normal Patient - No False Positive Critical Alerts)
4. Scenario 4: patient_final_test (Complex Multimorbidity Evaluation)
5. API Endpoint Tests: /assistant/evaluate, /assistant/audit, /assistant/status
6. Optional Live LLM Integration Tests (RUN_LLM_INTEGRATION_TESTS)
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.api.routes.patients import SAMPLE_PATIENTS
from app.clinical.orchestrator import ClinicalOrchestrator, get_clinical_orchestrator
from app.llm.schemas import (
    AssistantAuditRequest,
    AssistantEvaluationRequest,
    ClinicalAssistantResponse,
    ClinicalFindingItem,
    EvidenceItem,
    EvidenceSourceCitation,
    NextStepItem,
    RiskFlagItem,
    RuleBasedFindingItem,
)
from app.main import app
from app.reasoning.models import PatientProfile


class MockStructuredModel:
    """Mock LangChain chat model for unit test runs."""

    def __init__(self, output: ClinicalAssistantResponse) -> None:
        self.output = output

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = self.output
        return mock_chain


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# --------------------------------------------------------------------------- #
# End-to-End Scenarios
# --------------------------------------------------------------------------- #


class TestEndToEndClinicalScenarios:
    def test_scenario_1_ckd_metformin_alert(self) -> None:
        """Scenario 1: patient_ckd_metformin_alert must preserve critical renal contraindication."""
        orchestrator = get_clinical_orchestrator()
        req = AssistantEvaluationRequest(
            patient_id="patient_ckd_metformin_alert",
            question="What are the important clinical considerations for this patient?",
            include_audit=True,
        )
        response = orchestrator.evaluate(req)

        assert response.patient_id == "patient_ckd_metformin_alert"
        assert len(response.risk_flags) >= 1
        assert response.risk_flags[0].severity == "critical"
        rule_ids = [r.rule_id for r in response.rule_based_findings]
        assert "RULE_CKD_METFORMIN_CONTRAINDICATION_001" in rule_ids
        assert response.metadata.rules_evaluated_count >= 10
        assert response.metadata.critical_alerts_count >= 1

    def test_scenario_2_hyperkalemia_ddi(self) -> None:
        """Scenario 2: patient_hyperkalemia_ddi must flag hyperkalemia and ACEi/MRA interaction."""
        orchestrator = get_clinical_orchestrator()
        req = AssistantEvaluationRequest(
            patient_id="patient_hyperkalemia_ddi",
            question="Summarize the major medication and laboratory safety concerns for this patient.",
            include_audit=True,
        )
        response = orchestrator.evaluate(req)

        assert response.patient_id == "patient_hyperkalemia_ddi"
        assert len(response.risk_flags) >= 1
        rule_ids = [r.rule_id for r in response.rule_based_findings]
        assert any("HYPERKALEMIA" in rid or "DDI" in rid or "K_" in rid for rid in rule_ids)
        assert response.metadata.critical_alerts_count >= 1

    def test_scenario_3_healthy_normal_patient(self) -> None:
        """Scenario 3: Normal patient must not generate false positive critical alerts."""
        orchestrator = get_clinical_orchestrator()
        req = AssistantEvaluationRequest(
            patient_id="patient_healthy_normal",
            question="Assess overall clinical risk.",
        )
        response = orchestrator.evaluate(req)

        assert response.patient_id == "patient_healthy_normal"
        # No critical contraindications or emergency alerts should trigger for a healthy patient
        critical_flags = [f for f in response.risk_flags if f.severity == "critical"]
        assert len(critical_flags) == 0

    def test_scenario_4_complex_multimorbidity(self) -> None:
        """Scenario 4: patient_final_test with multiple conditions, medications, and abnormal labs."""
        orchestrator = get_clinical_orchestrator()
        req = AssistantEvaluationRequest(
            patient_id="patient_final_test",
            question="Summarize the important findings and explain which findings require the most attention.",
            include_audit=True,
        )
        response = orchestrator.evaluate(req)

        assert response.patient_id == "patient_final_test"
        assert len(response.risk_flags) >= 1
        assert response.metadata.rules_triggered_count >= 2
        assert len(response.recommended_next_steps) >= 1


# --------------------------------------------------------------------------- #
# REST API Endpoints Tests
# --------------------------------------------------------------------------- #


class TestAssistantAPIEndpoints:
    def test_api_evaluate_endpoint(self, client: TestClient) -> None:
        payload = {
            "patient_id": "patient_ckd_metformin_alert",
            "question": "What are the important clinical considerations for this patient?",
            "include_audit": True,
        }
        res = client.post("/api/v1/assistant/evaluate", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["data"]["patient_id"] == "patient_ckd_metformin_alert"
        assert "risk_flags" in data["data"]
        assert "symbolic_findings" in data["data"]
        assert "safety_notice" in data["data"]
        assert "metadata" in data["data"]

    def test_api_audit_endpoint(self, client: TestClient) -> None:
        payload = {
            "patient_id": "patient_ckd_metformin_alert",
            "question": "Assess renal safety.",
        }
        res = client.post("/api/v1/assistant/audit", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert data["data"]["patient_id"] == "patient_ckd_metformin_alert"
        assert "patient_facts" in data["data"]
        assert "rag_evidence" in data["data"]
        assert "knowledge_graph_facts" in data["data"]
        assert data["data"]["rules_evaluated"] >= 10
        assert len(data["data"]["critical_alerts"]) >= 1

    def test_api_status_endpoint(self, client: TestClient) -> None:
        res = client.get("/api/v1/assistant/status")
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert "components" in data["data"]
        comps = data["data"]["components"]
        assert "patient_service" in comps
        assert "rag" in comps
        assert "knowledge_graph" in comps
        assert "reasoning" in comps
        assert "llm" in comps
        assert "safety" in comps


# --------------------------------------------------------------------------- #
# Optional Live LLM Integration Test
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    os.getenv("RUN_LLM_INTEGRATION_TESTS", "false").lower() != "true",
    reason="Live LLM tests disabled by default. Set RUN_LLM_INTEGRATION_TESTS=true to run.",
)
def test_live_llm_integration_evaluation() -> None:
    orchestrator = get_clinical_orchestrator()
    req = AssistantEvaluationRequest(
        patient_id="patient_ckd_metformin_alert",
        question="Provide a structured safety assessment for this patient.",
    )
    res = orchestrator.evaluate(req)
    assert res.patient_id == "patient_ckd_metformin_alert"
    assert len(res.summary) > 20
    assert len(res.risk_flags) >= 1
