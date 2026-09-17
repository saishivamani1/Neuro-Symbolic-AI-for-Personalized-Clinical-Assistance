"""
tests/test_reasoning.py

Comprehensive test suite for Stage 4: Symbolic Clinical Reasoning Engine.

Tests:
1. Rule Operators & Condition Evaluator (app.reasoning.engine)
2. Forward Chaining Inference & Conflict Resolution
3. Clinical Decision Rules (Contraindications, DDIs, Diagnostics, Renoprotection)
4. Explainable Reasoning Trace Generation
5. Evidence Assembler (Patient + Knowledge Graph + RAG)
6. REST API Endpoints (/api/v1/patients/* and /api/v1/reasoning/*)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.reasoning.engine import SymbolicRuleEngine
from app.reasoning.evidence import EvidenceAssembler
from app.reasoning.models import (
    ClinicalRule,
    DerivedFact,
    PatientProfile,
    RuleCategory,
    RuleCondition,
    RuleOperator,
    RuleSeverity,
)
from app.reasoning.rules import DEFAULT_CLINICAL_RULES
from app.reasoning.service import ReasoningService


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def engine() -> SymbolicRuleEngine:
    """Fixture providing a fresh rule engine with default clinical rules."""
    return SymbolicRuleEngine(rules=DEFAULT_CLINICAL_RULES)


@pytest.fixture
def client() -> TestClient:
    """FastAPI TestClient."""
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture
def ckd_metformin_patient() -> PatientProfile:
    """Patient with severe renal impairment (eGFR 22) taking Metformin."""
    return PatientProfile(
        patient_id="patient_test_ckd_001",
        name="Test Patient CKD",
        age=65,
        vitals={"systolic_bp": 138.0, "diastolic_bp": 82.0},
        lab_results={"egfr": 22.0, "hba1c": 8.5, "serum_potassium": 4.5},
        conditions=["Type 2 Diabetes Mellitus", "Chronic Kidney Disease"],
        current_medications=["Metformin", "Amlodipine"],
        symptoms=["Fatigue"],
    )


@pytest.fixture
def ddi_hyperkalemia_patient() -> PatientProfile:
    """Patient with concurrent Lisinopril + Spironolactone and severe hyperkalemia."""
    return PatientProfile(
        patient_id="patient_test_ddi_002",
        name="Test Patient DDI",
        age=70,
        vitals={"systolic_bp": 150.0, "diastolic_bp": 94.0},
        lab_results={"serum_potassium": 5.9, "egfr": 50.0},
        conditions=["Essential Hypertension", "Heart Failure"],
        current_medications=["Lisinopril", "Spironolactone"],
        symptoms=["Headache", "Shortness of Breath"],
    )


# --------------------------------------------------------------------------- #
# 1. Rule Operators & Condition Evaluation Tests
# --------------------------------------------------------------------------- #


class TestRuleConditionOperators:
    def test_numeric_comparisons(self, engine: SymbolicRuleEngine) -> None:
        memory = {"lab_results": {"egfr": 25.0, "potassium": 5.8}}

        # < operator
        cond_lt = RuleCondition(fact_path="lab_results.egfr", operator=RuleOperator.LESS_THAN, value=30.0)
        matched, _ = engine._evaluate_condition(cond_lt, memory)
        assert matched is True

        # >= operator
        cond_gte = RuleCondition(fact_path="lab_results.potassium", operator=RuleOperator.GREATER_EQUAL, value=5.5)
        matched, _ = engine._evaluate_condition(cond_gte, memory)
        assert matched is True

        # > operator false case
        cond_gt = RuleCondition(fact_path="lab_results.egfr", operator=RuleOperator.GREATER_THAN, value=60.0)
        matched, _ = engine._evaluate_condition(cond_gt, memory)
        assert matched is False

    def test_list_contains_operators(self, engine: SymbolicRuleEngine) -> None:
        memory = {"current_medications": ["Metformin", "Lisinopril"], "conditions": ["Type 2 Diabetes Mellitus"]}

        # contains case-insensitive
        cond_contains = RuleCondition(fact_path="current_medications", operator=RuleOperator.CONTAINS, value="metformin")
        matched, _ = engine._evaluate_condition(cond_contains, memory)
        assert matched is True

        # not_contains
        cond_not_contains = RuleCondition(fact_path="current_medications", operator=RuleOperator.NOT_CONTAINS, value="Empagliflozin")
        matched, _ = engine._evaluate_condition(cond_not_contains, memory)
        assert matched is True

    def test_is_empty_and_between(self, engine: SymbolicRuleEngine) -> None:
        memory = {"lab_results": {"egfr": 38.0}, "missing_test": None}

        # is_empty
        cond_empty = RuleCondition(fact_path="missing_test", operator=RuleOperator.IS_EMPTY)
        matched, _ = engine._evaluate_condition(cond_empty, memory)
        assert matched is True

        # between
        cond_between = RuleCondition(fact_path="lab_results.egfr", operator=RuleOperator.BETWEEN, value=[30.0, 45.0])
        matched, _ = engine._evaluate_condition(cond_between, memory)
        assert matched is True


# --------------------------------------------------------------------------- #
# 2. Forward Chaining & Clinical Rules Tests
# --------------------------------------------------------------------------- #


class TestSymbolicInference:
    def test_metformin_severe_ckd_contraindication_fires(
        self, engine: SymbolicRuleEngine, ckd_metformin_patient: PatientProfile
    ) -> None:
        result = engine.evaluate(ckd_metformin_patient)

        assert result.triggered_rules_count > 0
        rule_ids = [r.rule_id for r in result.triggered_rules]
        assert "R001" in rule_ids

        # Critical alert present
        assert len(result.critical_alerts) >= 1
        critical_names = [a.name for a in result.critical_alerts]
        assert "metformin_contraindicated" in critical_names

        # Verify explainable trace
        assert len(result.reasoning_trace) > 0
        step1 = next(s for s in result.reasoning_trace if s.rule_id == "R001")
        assert step1.severity == "critical"
        assert "ADA" in step1.guideline_citation

    def test_ddi_and_critical_hyperkalemia_fire(
        self, engine: SymbolicRuleEngine, ddi_hyperkalemia_patient: PatientProfile
    ) -> None:
        result = engine.evaluate(ddi_hyperkalemia_patient)

        rule_ids = [r.rule_id for r in result.triggered_rules]
        assert any(r in rule_ids for r in ("R015", "R017"))
        assert "R022" in rule_ids

        # Verify safety alert
        critical_alerts = result.critical_alerts
        assert len(critical_alerts) >= 1
        alert_names = [a.name for a in critical_alerts]
        assert any("hyperkalemia" in a for a in alert_names)

    def test_forward_chaining_cascading_inference(self) -> None:
        """Test rule 1 deducing a diagnosis which triggers rule 2."""
        rule_step_1 = ClinicalRule(
            rule_id="RULE_TEST_1",
            name="Detect High Fasting Glucose",
            category=RuleCategory.DIAGNOSIS,
            severity=RuleSeverity.HIGH,
            description="Fasting glucose >= 126 indicates diabetes.",
            guideline_source="ADA Guidelines",
            conditions=[
                RuleCondition(fact_path="lab_results.fasting_glucose", operator=RuleOperator.GREATER_EQUAL, value=126.0)
            ],
            conclusions=[
                DerivedFact(
                    fact_id="FACT_DIABETES_CONFIRMED",
                    category="diagnosis",
                    name="Type 2 Diabetes Mellitus",
                    explanation="Fasting blood glucose elevated.",
                )
            ],
        )

        rule_step_2 = ClinicalRule(
            rule_id="RULE_TEST_2",
            name="Recommend Metformin for Diabetes",
            category=RuleCategory.TREATMENT_RECOMMENDATION,
            severity=RuleSeverity.HIGH,
            description="First-line therapy for newly diagnosed diabetes is Metformin.",
            guideline_source="ADA Guidelines",
            conditions=[
                RuleCondition(fact_path="conditions", operator=RuleOperator.CONTAINS, value="Type 2 Diabetes Mellitus")
            ],
            conclusions=[
                DerivedFact(
                    fact_id="FACT_START_METFORMIN",
                    category="treatment_recommendation",
                    name="Initiate Metformin Pharmacotherapy",
                    explanation="First-line antidiabetic drug indicated.",
                )
            ],
        )

        custom_engine = SymbolicRuleEngine(rules=[rule_step_1, rule_step_2])
        patient = {
            "patient_id": "patient_cascade",
            "lab_results": {"fasting_glucose": 140.0},
            "conditions": [],
        }

        result = custom_engine.evaluate(patient)
        assert result.triggered_rules_count == 2
        fact_ids = [f.fact_id for f in result.derived_facts]
        assert "FACT_DIABETES_CONFIRMED" in fact_ids
        assert "FACT_START_METFORMIN" in fact_ids

    def test_empty_patient_evaluation(self, engine: SymbolicRuleEngine) -> None:
        empty_patient = PatientProfile(patient_id="patient_empty")
        result = engine.evaluate(empty_patient)
        assert result.patient_id == "patient_empty"
        assert result.triggered_rules_count == 0
        assert len(result.derived_facts) == 0


# --------------------------------------------------------------------------- #
# 3. Evidence Assembler & Reasoning Service Tests
# --------------------------------------------------------------------------- #


class TestEvidenceAssemblerAndService:
    def test_evidence_assembler_working_memory(self, ckd_metformin_patient: PatientProfile) -> None:
        assembler = EvidenceAssembler()
        # Assemble memory without crashing even if KG or RAG are empty
        memory = assembler.assemble_patient_working_memory(
            patient=ckd_metformin_patient,
            include_kg=False,
            include_rag=False,
        )
        assert memory["patient_id"] == "patient_test_ckd_001"
        assert memory["lab_results"]["egfr"] == 22.0

    def test_reasoning_service_evaluation(self, ckd_metformin_patient: PatientProfile) -> None:
        service = ReasoningService()
        result = service.evaluate_patient(
            patient=ckd_metformin_patient,
            include_kg=False,
            include_rag=False,
        )
        assert result.triggered_rules_count > 0
        assert len(service.list_rules()) >= 10
        assert service.get_rule("RULE_CKD_METFORMIN_CONTRAINDICATION_001") is not None


# --------------------------------------------------------------------------- #
# 4. FastAPI REST Endpoints Tests
# --------------------------------------------------------------------------- #


class TestPatientAndReasoningAPI:
    def test_evaluate_patient_endpoint(self, client: TestClient, ckd_metformin_patient: PatientProfile) -> None:
        response = client.post(
            "/api/v1/patients/evaluate?include_kg=false&include_rag=false",
            json=ckd_metformin_patient.model_dump(),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        result = data["data"]
        assert result["triggered_rules_count"] > 0
        assert len(result["critical_alerts"]) >= 1
        assert len(result["reasoning_trace"]) > 0

    def test_get_sample_patient_evaluation(self, client: TestClient) -> None:
        response = client.get("/api/v1/patients/patient_ckd_metformin_alert?include_kg=false&include_rag=false")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["patient_id"] == "patient_ckd_metformin_alert"
        assert data["data"]["triggered_rules_count"] > 0

    def test_list_sample_profiles_endpoint(self, client: TestClient) -> None:
        response = client.get("/api/v1/patients/sample/profiles")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) >= 3

    def test_list_clinical_rules_endpoint(self, client: TestClient) -> None:
        response = client.get("/api/v1/reasoning/rules")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) >= 10

    def test_get_clinical_rule_by_id(self, client: TestClient) -> None:
        response = client.get("/api/v1/reasoning/rules/R001")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["rule_id"] == "R001"
        assert data["data"]["priority"] == "critical"

        # Also verify legacy alias resolves to R001
        legacy_res = client.get("/api/v1/reasoning/rules/RULE_CKD_METFORMIN_CONTRAINDICATION_001")
        assert legacy_res.status_code == 200
        assert legacy_res.json()["data"]["rule_id"] == "R001"

    def test_get_clinical_rule_not_found(self, client: TestClient) -> None:
        response = client.get("/api/v1/reasoning/rules/NON_EXISTENT_RULE")
        assert response.status_code == 404
