"""
tests/test_assistant.py

End-to-end and API integration tests for Stage 5: Clinical Assistant Layer.

Covers:
1. Synthetic clinical test cases (patient_ckd_metformin_alert & patient_hyperkalemia_ddi)
2. Inline patient profile evaluation
3. Audit mode with deterministic reasoning trace exposure
4. Graceful degradation when LLM is unconfigured or unavailable
5. REST API endpoints (/api/v1/assistant/evaluate & /api/v1/assistant/status)
6. Optional live integration test (disabled by default via RUN_LLM_INTEGRATION_TESTS)
"""

from __future__ import annotations

import os
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api.routes.patients import SAMPLE_PATIENTS
from app.llm.models import LLMFactory
from app.llm.schemas import (
    AssistantEvaluationRequest,
    ClinicalAssistantResponse,
    ClinicalFindingItem,
    EvidenceItem,
    EvidenceSourceCitation,
    NextStepItem,
    RiskFlagItem,
    RuleBasedFindingItem,
    UncertaintyItem,
)
from app.llm.service import ClinicalAssistantService
from app.main import app
from app.reasoning.models import PatientProfile


# --------------------------------------------------------------------------- #
# Mock Helpers & Fixtures
# --------------------------------------------------------------------------- #


class MockStructuredChatModel:
    """Mock LangChain chat model returning valid ClinicalAssistantResponse."""

    def __init__(self, response_data: ClinicalAssistantResponse) -> None:
        self.response_data = response_data

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        mock_runnable = MagicMock()
        mock_runnable.invoke.return_value = self.response_data
        return mock_runnable


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture
def mock_ckd_response() -> ClinicalAssistantResponse:
    return ClinicalAssistantResponse(
        patient_id="patient_ckd_metformin_alert",
        summary="Patient has Type 2 Diabetes with severe renal impairment (eGFR 24 mL/min). Metformin is strictly contraindicated due to lactic acidosis risk.",
        clinical_findings=[
            ClinicalFindingItem(
                finding="Severe renal impairment (eGFR 24 mL/min/1.73m²)",
                category="lab_abnormality",
                severity="critical",
                evidence=[
                    EvidenceSourceCitation(
                        type="patient_fact",
                        document="Patient Lab Record",
                    )
                ],
            )
        ],
        risk_flags=[
            RiskFlagItem(
                flag="Metformin Severe Renal Contraindication",
                severity="critical",
                clinical_implication="High risk of fatal lactic acidosis.",
                evidence=[
                    EvidenceSourceCitation(
                        type="rule",
                        rule_id="RULE_CKD_METFORMIN_CONTRAINDICATION_001",
                        document="ADA Standards of Care in Diabetes (2024)",
                        page=2,
                    )
                ],
            )
        ],
        rule_based_findings=[
            RuleBasedFindingItem(
                rule_id="RULE_CKD_METFORMIN_CONTRAINDICATION_001",
                rule_name="Metformin Contraindication in Severe Renal Impairment",
                category="contraindication",
                severity="critical",
                conclusion="Metformin contraindicated in eGFR < 30 mL/min.",
                actionable_guidance="Discontinue Metformin immediately.",
                guideline_source="ADA Standards of Care in Diabetes (2024)",
                page_number=2,
            )
        ],
        knowledge_graph_findings=[],
        evidence=[
            EvidenceItem(
                claim="Metformin is strictly contraindicated in patients with eGFR < 30 mL/min.",
                sources=[
                    EvidenceSourceCitation(
                        type="rag",
                        document="clinical_guidelines_sample.pdf",
                        page=2,
                    ),
                    EvidenceSourceCitation(
                        type="rule",
                        rule_id="RULE_CKD_METFORMIN_CONTRAINDICATION_001",
                    ),
                ],
            )
        ],
        uncertainties=[
            UncertaintyItem(
                gap_type="missing_laboratory_value",
                description="Urine albumin-to-creatinine ratio (uACR) is not recorded.",
                impact_on_decision="Needed to quantify nephropathy staging.",
            )
        ],
        recommended_next_steps=[
            NextStepItem(
                action="Discontinue Metformin and initiate alternative glycemic control.",
                priority="immediate",
                rationale="eGFR < 30 creates elevated lactic acidosis risk.",
                supporting_guideline="ADA Standards of Care in Diabetes (2024)",
            )
        ],
    )


# --------------------------------------------------------------------------- #
# 1. Clinical Assistant Service Tests
# --------------------------------------------------------------------------- #


class TestClinicalAssistantService:
    def test_evaluate_patient_ckd_metformin_with_mock_llm(
        self, mock_ckd_response: ClinicalAssistantResponse
    ) -> None:
        mock_model = MockStructuredChatModel(mock_ckd_response)
        mock_factory = MagicMock(spec=LLMFactory)
        mock_factory.is_configured.return_value = True
        mock_factory.get_model.return_value = mock_model

        service = ClinicalAssistantService(llm_factory=mock_factory)
        request = AssistantEvaluationRequest(
            patient_id="patient_ckd_metformin_alert",
            question="What are the important clinical considerations for this patient?",
        )

        response = service.evaluate_patient(request)

        assert response.patient_id == "patient_ckd_metformin_alert"
        assert "Metformin" in response.summary
        assert len(response.rule_based_findings) >= 1
        assert response.rule_based_findings[0].rule_id == "RULE_CKD_METFORMIN_CONTRAINDICATION_001"
        assert len(response.risk_flags) >= 1
        assert response.metadata.rules_triggered_count >= 1
        assert response.metadata.critical_alerts_count >= 1

    def test_evaluate_inline_patient_object(
        self, mock_ckd_response: ClinicalAssistantResponse
    ) -> None:
        mock_model = MockStructuredChatModel(mock_ckd_response)
        mock_factory = MagicMock(spec=LLMFactory)
        mock_factory.is_configured.return_value = True
        mock_factory.get_model.return_value = mock_model

        service = ClinicalAssistantService(llm_factory=mock_factory)
        inline_patient = PatientProfile(
            patient_id="patient_custom_eval",
            name="Custom Patient",
            vitals={"systolic_bp": 144.0, "diastolic_bp": 90.0},
            lab_results={"egfr": 25.0, "hba1c": 8.6},
            conditions=["Type 2 Diabetes Mellitus"],
            current_medications=["Metformin"],
        )
        request = AssistantEvaluationRequest(
            patient=inline_patient,
            question="Summarize the important safety findings.",
        )

        response = service.evaluate_patient(request)
        assert response.patient_id == "patient_custom_eval"

    def test_audit_mode_includes_deterministic_trace(
        self, mock_ckd_response: ClinicalAssistantResponse
    ) -> None:
        mock_model = MockStructuredChatModel(mock_ckd_response)
        mock_factory = MagicMock(spec=LLMFactory)
        mock_factory.is_configured.return_value = True
        mock_factory.get_model.return_value = mock_model

        service = ClinicalAssistantService(llm_factory=mock_factory)
        request = AssistantEvaluationRequest(
            patient_id="patient_ckd_metformin_alert",
            include_audit=True,
        )

        response = service.evaluate_patient(request)
        assert response.metadata.audit_data is not None
        audit = response.metadata.audit_data
        assert "triggered_rules" in audit
        assert "reasoning_trace" in audit
        assert "derived_facts" in audit
        assert len(audit["reasoning_trace"]) > 0

    def test_unconfigured_llm_graceful_fallback(self) -> None:
        """When OpenAI is unconfigured, service should return deterministic symbolic fallback."""
        mock_factory = MagicMock(spec=LLMFactory)
        mock_factory.is_configured.return_value = False

        service = ClinicalAssistantService(llm_factory=mock_factory)
        request = AssistantEvaluationRequest(
            patient_id="patient_ckd_metformin_alert",
        )

        response = service.evaluate_patient(request)
        assert response.patient_id == "patient_ckd_metformin_alert"
        assert len(response.rule_based_findings) >= 1
        assert "Deterministic clinical evaluation" in response.summary
        assert response.metadata.critical_alerts_count >= 1


# --------------------------------------------------------------------------- #
# 2. REST API Endpoints Tests
# --------------------------------------------------------------------------- #


class TestAssistantAPI:
    def test_assistant_status_endpoint(self, client: TestClient) -> None:
        response = client.get("/api/v1/assistant/status")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "status" in data["data"]
        assert "model" in data["data"]

    def test_assistant_evaluate_patient_id_endpoint(self, client: TestClient) -> None:
        payload = {
            "patient_id": "patient_ckd_metformin_alert",
            "question": "What are the important clinical considerations for this patient?",
        }
        response = client.post("/api/v1/assistant/evaluate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["patient_id"] == "patient_ckd_metformin_alert"
        assert len(data["data"]["rule_based_findings"]) >= 1

    def test_assistant_evaluate_with_audit_endpoint(self, client: TestClient) -> None:
        payload = {
            "patient_id": "patient_hyperkalemia_ddi",
            "include_audit": True,
        }
        response = client.post("/api/v1/assistant/evaluate?include_audit=true", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["metadata"]["audit_data"] is not None

    def test_assistant_evaluate_inline_patient_endpoint(self, client: TestClient) -> None:
        payload = {
            "patient": {
                "patient_id": "patient_inline_api_test",
                "age": 60,
                "vitals": {"systolic_bp": 148.0, "diastolic_bp": 92.0},
                "lab_results": {"egfr": 22.0, "serum_potassium": 4.6},
                "conditions": ["Type 2 Diabetes Mellitus"],
                "current_medications": ["Metformin"],
            },
            "question": "Summarize safety concerns.",
        }
        response = client.post("/api/v1/assistant/evaluate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["patient_id"] == "patient_inline_api_test"


# --------------------------------------------------------------------------- #
# 3. Optional Live OpenAI Integration Test
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    os.getenv("RUN_LLM_INTEGRATION_TESTS") != "true",
    reason="Live OpenAI integration test disabled by default. Set RUN_LLM_INTEGRATION_TESTS=true to run.",
)
class TestLiveOpenAIIntegration:
    def test_live_assistant_evaluation(self, client: TestClient) -> None:
        payload = {
            "patient_id": "patient_ckd_metformin_alert",
            "question": "What are the important clinical considerations and safety warnings for this patient?",
        }
        response = client.post("/api/v1/assistant/evaluate", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        res_data = data["data"]
        assert len(res_data["summary"]) > 20
        assert len(res_data["rule_based_findings"]) >= 1
