"""
tests/test_clinical_safety.py

Unit tests for ClinicalSafetyService, hallucination defense, citation verification,
and contradiction detection.
"""

from __future__ import annotations

import pytest

from app.clinical.safety import ClinicalSafetyService
from app.llm.schemas import (
    ClinicalAssistantResponse,
    EvidenceItem,
    EvidenceSourceCitation,
    RiskFlagItem,
    RuleBasedFindingItem,
)


class TestClinicalSafetyService:
    def test_patient_id_consistency_enforcement(self) -> None:
        context = {"patient_facts": {"patient_id": "P_CORRECT_100"}}
        raw_response = ClinicalAssistantResponse(
            patient_id="P_WRONG_999",
            summary="Patient evaluation.",
        )
        validated = ClinicalSafetyService.validate_and_sanitize(raw_response, context)
        assert validated.patient_id == "P_CORRECT_100"

    def test_fabricated_rule_id_removal(self) -> None:
        context = {
            "patient_facts": {"patient_id": "P_001"},
            "valid_citations": {"rule_ids": ["RULE_CKD_METFORMIN_CONTRAINDICATION_001"]},
        }
        raw_response = ClinicalAssistantResponse(
            patient_id="P_001",
            summary="Summary.",
            rule_based_findings=[
                RuleBasedFindingItem(
                    rule_id="RULE_FABRICATED_MAGIC_999",
                    rule_name="Fake Rule",
                    category="diagnosis",
                    severity="high",
                    conclusion="Fake",
                    actionable_guidance="None",
                    guideline_source="Fake Source",
                )
            ],
        )
        validated = ClinicalSafetyService.validate_and_sanitize(raw_response, context)
        assert len(validated.rule_based_findings) == 0

    def test_critical_alert_preservation(self) -> None:
        alert = {
            "name": "Metformin Severe Renal Contraindication",
            "fact_id": "FACT_ALERT_1",
            "explanation": "eGFR < 30 taking Metformin. Risk of lactic acidosis.",
            "supporting_rules": ["RULE_CKD_METFORMIN_CONTRAINDICATION_001"],
            "provenance": {"guideline": "ADA Guidelines 2024", "page": 2},
        }
        context = {
            "patient_facts": {"patient_id": "P_001"},
            "symbolic_reasoning": {"critical_alerts": [alert]},
            "valid_citations": {"rule_ids": ["RULE_CKD_METFORMIN_CONTRAINDICATION_001"]},
        }
        raw_response = ClinicalAssistantResponse(
            patient_id="P_001",
            summary="General diabetes management.",
            risk_flags=[],
            rule_based_findings=[],
        )
        validated = ClinicalSafetyService.validate_and_sanitize(raw_response, context)
        assert len(validated.risk_flags) >= 1
        assert validated.risk_flags[0].severity == "critical"
        assert len(validated.rule_based_findings) >= 1
        assert validated.rule_based_findings[0].rule_id == "RULE_CKD_METFORMIN_CONTRAINDICATION_001"

    def test_contradiction_detection_metformin(self) -> None:
        alert = {
            "name": "Metformin Contraindicated",
            "explanation": "Metformin is contraindicated in severe CKD",
            "supporting_rules": ["RULE_CKD_METFORMIN_CONTRAINDICATION_001"],
        }
        context = {
            "patient_facts": {"patient_id": "P_001"},
            "symbolic_reasoning": {"critical_alerts": [alert]},
        }
        raw_response = ClinicalAssistantResponse(
            patient_id="P_001",
            summary="It is safe to use Metformin and continue normal dosing.",
        )
        validated = ClinicalSafetyService.validate_and_sanitize(raw_response, context)
        assert "SAFETY WARNING" in validated.summary
        assert "Metformin is strictly contraindicated" in validated.summary

    def test_safety_notice_inclusion(self) -> None:
        context = {"patient_facts": {"patient_id": "P_001"}}
        raw_response = ClinicalAssistantResponse(
            patient_id="P_001",
            summary="Summary.",
            safety_notice="",
        )
        validated = ClinicalSafetyService.validate_and_sanitize(raw_response, context)
        assert "clinical decision support" in validated.safety_notice.lower()
