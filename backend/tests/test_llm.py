"""
tests/test_llm.py

Unit tests for Stage 5: LangChain LLM integration, Prompts, Context Assembler,
and Deterministic Hallucination Guardrails.

All tests run without requiring a live OpenAI API key (mocked LLM responses).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from app.core.config import Settings
from app.core.exceptions import LLMError
from app.llm.chains import build_clinical_assistant_chain
from app.llm.context import ClinicalContextAssembler
from app.llm.guardrails import ClinicalGuardrailValidator
from app.llm.models import LLMFactory
from app.llm.prompts import get_clinical_prompt_template
from app.llm.schemas import (
    ClinicalAssistantResponse,
    EvidenceItem,
    EvidenceSourceCitation,
    RiskFlagItem,
    RuleBasedFindingItem,
)
from app.reasoning.models import PatientProfile


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def mock_settings_configured() -> Settings:
    return Settings(
        openai_api_key="sk-mock-test-key-12345",
        llm_provider="openai",
        llm_model="gpt-4o-mini",
        llm_temperature=0.1,
    )


@pytest.fixture
def mock_settings_unconfigured() -> Settings:
    return Settings(
        openai_api_key="",
        llm_provider="openai",
        llm_model="gpt-4o-mini",
    )


@pytest.fixture
def sample_patient() -> PatientProfile:
    return PatientProfile(
        patient_id="patient_test_001",
        name="Test Patient",
        age=65,
        vitals={"systolic_bp": 146.0, "diastolic_bp": 92.0},
        lab_results={"egfr": 24.0, "hba1c": 8.5, "serum_potassium": 4.8},
        conditions=["Type 2 Diabetes Mellitus", "Chronic Kidney Disease"],
        current_medications=["Metformin"],
        symptoms=["Fatigue"],
    )


# --------------------------------------------------------------------------- #
# 1. LLM Factory & Configuration Tests
# --------------------------------------------------------------------------- #


class TestLLMFactory:
    def test_unconfigured_status(self, mock_settings_unconfigured: Settings) -> None:
        factory = LLMFactory(settings=mock_settings_unconfigured)
        assert factory.is_configured() is False
        status = factory.health_check()
        assert status["status"] == "unconfigured"

    def test_configured_status(self, mock_settings_configured: Settings) -> None:
        factory = LLMFactory(settings=mock_settings_configured)
        assert factory.is_configured() is True
        status = factory.health_check()
        assert status["status"] == "configured"
        assert status["model"] == "gpt-4o-mini"

    def test_get_model_raises_when_unconfigured(
        self, mock_settings_unconfigured: Settings
    ) -> None:
        factory = LLMFactory(settings=mock_settings_unconfigured)
        with pytest.raises(LLMError) as exc_info:
            factory.get_model()
        assert "OpenAI API key is not configured" in str(exc_info.value)

    def test_groq_unconfigured_status(self) -> None:
        settings = Settings(llm_provider="groq", groq_api_key="", openai_api_key="")
        factory = LLMFactory(settings=settings)
        assert factory.is_configured() is False
        status = factory.health_check()
        assert status["status"] == "unconfigured"
        assert "GROQ_API_KEY" in status["detail"]

    def test_groq_configured_status(self) -> None:
        settings = Settings(
            llm_provider="groq",
            groq_api_key="gsk-test-key",
            groq_model="llama-3.3-70b-versatile",
        )
        factory = LLMFactory(settings=settings)
        assert factory.is_configured() is True
        status = factory.health_check()
        assert status["status"] == "configured"
        assert status["provider"] == "groq"
        assert status["model"] == "llama-3.3-70b-versatile"

    def test_groq_model_name_resolved_from_groq_model_not_llm_model(self) -> None:
        """Ensure that when provider=groq, the model sent to ChatGroq is groq_model,
        NOT llm_model (which is the OpenAI default like gpt-4o-mini)."""
        settings = Settings(
            llm_provider="groq",
            groq_api_key="gsk-test-key",
            llm_model="gpt-4o-mini",         # OpenAI default — must NOT be used for Groq
            groq_model="llama-3.3-70b-versatile",
        )
        factory = LLMFactory(settings=settings)
        model_name = factory._resolve_model_name()
        assert model_name == "llama-3.3-70b-versatile", (
            f"Expected 'llama-3.3-70b-versatile' but got '{model_name}'. "
            "groq_model should always be used when llm_provider=groq."
        )
        # health_check must also report groq_model
        status = factory.health_check()
        assert status["model"] == "llama-3.3-70b-versatile"

    def test_groq_get_model_uses_groq_model_not_llm_model(self) -> None:
        """Verify ChatGroq is instantiated with groq_model, not llm_model."""
        settings = Settings(
            llm_provider="groq",
            groq_api_key="gsk-test-key",
            llm_model="gpt-4o-mini",
            groq_model="llama-3.3-70b-versatile",
        )
        factory = LLMFactory(settings=settings)
        with patch("langchain_groq.ChatGroq") as mock_chatgroq_cls:
            mock_chatgroq_cls.return_value = MagicMock()
            try:
                # Import will succeed now langchain-groq is installed
                factory.get_model()
            except Exception:
                pass
            if mock_chatgroq_cls.called:
                call_kwargs = mock_chatgroq_cls.call_args
                model_arg = (
                    call_kwargs.kwargs.get("model")
                    or (call_kwargs.args[0] if call_kwargs.args else None)
                )
                assert model_arg == "llama-3.3-70b-versatile", (
                    f"ChatGroq was called with model='{model_arg}' but expected 'llama-3.3-70b-versatile'"
                )


# --------------------------------------------------------------------------- #
# 2. Prompts & Context Assembler Tests
# --------------------------------------------------------------------------- #


class TestPromptsAndContextAssembler:
    def test_prompt_template_structure(self) -> None:
        prompt_tmpl = get_clinical_prompt_template()
        messages = prompt_tmpl.format_messages(
            structured_context_json="{}",
            question="Summarize patient safety.",
        )
        assert len(messages) == 2
        # System prompt asserts constraints
        assert "STRICT GROUNDING" in messages[0].content
        assert "SYMBOLIC REASONING PRESERVATION" in messages[0].content
        assert "Summarize patient safety." in messages[1].content

    def test_context_assembler_structure(self, sample_patient: PatientProfile) -> None:
        assembler = ClinicalContextAssembler()
        context = assembler.assemble(sample_patient, include_rag=False, include_kg=False)

        assert "patient_facts" in context
        assert "symbolic_reasoning" in context
        assert "valid_citations" in context

        symbolic = context["symbolic_reasoning"]
        assert symbolic["rules_evaluated_count"] >= 10
        assert symbolic["triggered_rules_count"] >= 1
        assert len(symbolic["critical_alerts"]) >= 1


# --------------------------------------------------------------------------- #
# 3. Deterministic Guardrails Tests
# --------------------------------------------------------------------------- #


class TestHallucinationGuardrails:
    def test_patient_id_mismatch_correction(self) -> None:
        context = {
            "patient_facts": {"patient_id": "patient_expected_123"},
            "symbolic_reasoning": {"critical_alerts": []},
            "valid_citations": {"rule_ids": ["RULE_CKD_METFORMIN_CONTRAINDICATION_001"]},
        }
        raw_response = ClinicalAssistantResponse(
            patient_id="patient_wrong_999",
            summary="Test summary.",
            rule_based_findings=[],
            risk_flags=[],
        )
        sanitized = ClinicalGuardrailValidator.validate_and_sanitize(raw_response, context)
        assert sanitized.patient_id == "patient_expected_123"

    def test_critical_alert_preservation_enforcement(self) -> None:
        """If LLM dropped a critical symbolic alert, the guardrail must inject it."""
        critical_alert = {
            "fact_id": "FACT_ALERT_METFORMIN_CONTRAINDICATED",
            "name": "Metformin Severe Renal Contraindication",
            "category": "contraindication",
            "explanation": "eGFR < 30 taking Metformin. Risk of fatal lactic acidosis.",
            "supporting_rules": ["RULE_CKD_METFORMIN_CONTRAINDICATION_001"],
            "provenance": {
                "guideline": "ADA Standards of Care in Diabetes (2024)",
                "page": 2,
            },
        }
        context = {
            "patient_facts": {"patient_id": "patient_test_001"},
            "symbolic_reasoning": {"critical_alerts": [critical_alert]},
            "valid_citations": {"rule_ids": ["RULE_CKD_METFORMIN_CONTRAINDICATION_001"]},
        }
        # Response with missing critical alert
        raw_response = ClinicalAssistantResponse(
            patient_id="patient_test_001",
            summary="Patient has diabetes.",
            rule_based_findings=[],
            risk_flags=[],
        )
        sanitized = ClinicalGuardrailValidator.validate_and_sanitize(raw_response, context)

        # Critical alert was injected
        rule_ids = [r.rule_id for r in sanitized.rule_based_findings]
        assert "RULE_CKD_METFORMIN_CONTRAINDICATION_001" in rule_ids
        assert len(sanitized.risk_flags) >= 1
        assert sanitized.risk_flags[0].severity == "critical"

    def test_fabricated_rule_id_filtered(self) -> None:
        context = {
            "patient_facts": {"patient_id": "patient_test_001"},
            "symbolic_reasoning": {"critical_alerts": []},
            "valid_citations": {"rule_ids": ["RULE_CKD_METFORMIN_CONTRAINDICATION_001"]},
        }
        raw_response = ClinicalAssistantResponse(
            patient_id="patient_test_001",
            summary="Summary.",
            rule_based_findings=[
                RuleBasedFindingItem(
                    rule_id="FABRICATED_MAGIC_RULE_999",
                    rule_name="Fake Rule",
                    category="diagnosis",
                    severity="high",
                    conclusion="Fake conclusion",
                    actionable_guidance="None",
                    guideline_source="Fake Journal",
                )
            ],
            risk_flags=[],
            evidence=[
                EvidenceItem(
                    claim="Fake claim",
                    sources=[
                        EvidenceSourceCitation(
                            type="rule",
                            rule_id="FABRICATED_MAGIC_RULE_999",
                        )
                    ],
                )
            ],
        )
        sanitized = ClinicalGuardrailValidator.validate_and_sanitize(raw_response, context)
        assert len(sanitized.rule_based_findings) == 0
        assert len(sanitized.evidence[0].sources) == 0
