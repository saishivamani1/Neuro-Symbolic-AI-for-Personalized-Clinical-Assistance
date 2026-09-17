"""
backend/tests/test_symbolic_engine_benchmark.py

Comprehensive unit test suite for the Deterministic Neuro-Symbolic Clinical Reasoning Engine.

Tests:
1. All comparison operators: ==, !=, <, <=, >, >=, contains, not_contains
2. Logical combinators: AND, OR
3. Missing facts handling and null safety
4. Numeric and string type-safe comparisons
5. Multiple medications containment
6. Exact clinical boundary threshold testing:
   - eGFR: 29, 30, 31 (metformin contraindication < 30)
   - Potassium: 5.4, 5.5, 5.6 (hyperkalemia > 5.5)
   - UACR: 29, 30, 31 (moderate albuminuria >= 30)
   - LDL-C: 69, 70, 71 (statin indication >= 70)
   - Systolic BP: 129, 130, 131 (CKD BP goal >= 130)
   - Systolic BP: 139, 140, 141 (Stage 2 HTN >= 140)
   - Diastolic BP: 79, 80, 81 (CKD BP goal >= 80)
   - Diastolic BP: 89, 90, 91 (Stage 2 HTN >= 90)
7. Forward chaining deduction and iteration control
8. Duplicate derived facts suppression
9. Rule priority ordering (critical > high > medium > low)
10. Critical alert identification and preservation
"""

import pytest
from typing import Any, Dict, List

from app.reasoning.engine import SymbolicRuleEngine
from app.reasoning.loader import get_clinical_rules
from app.reasoning.models import (
    ClinicalRule,
    DerivedFact,
    PatientProfile,
    RuleCondition,
    RuleConclusion,
    RuleOperator,
    RuleSeverity,
)


@pytest.fixture
def engine() -> SymbolicRuleEngine:
    """Fresh instance of the SymbolicRuleEngine loaded with 50 rules."""
    return SymbolicRuleEngine(rules=get_clinical_rules())


# --------------------------------------------------------------------------- #
# 1. Operators & Logic Unit Tests
# --------------------------------------------------------------------------- #

class TestOperatorsAndLogic:
    def test_equals_and_not_equals(self) -> None:
        rule_eq = ClinicalRule(
            rule_id="TEST_EQ",
            rule_name="Test Equals",
            priority="medium",
            conditions=[RuleCondition(fact="type2_diabetes", operator="==", value=True)],
            logic="AND",
            conclusion=RuleConclusion(type="test", fact="has_t2d", value=True),
        )
        rule_neq = ClinicalRule(
            rule_id="TEST_NEQ",
            rule_name="Test Not Equals",
            priority="medium",
            conditions=[RuleCondition(fact="type2_diabetes", operator="!=", value=True)],
            logic="AND",
            conclusion=RuleConclusion(type="test", fact="no_t2d", value=True),
        )
        custom_engine = SymbolicRuleEngine(rules=[rule_eq, rule_neq])

        res_pos = custom_engine.evaluate({"type2_diabetes": True})
        assert [r.rule_id for r in res_pos.triggered_rules] == ["TEST_EQ"]

        res_neg = custom_engine.evaluate({"type2_diabetes": False})
        assert [r.rule_id for r in res_neg.triggered_rules] == ["TEST_NEQ"]

    def test_numeric_comparisons(self) -> None:
        rules = [
            ClinicalRule(
                rule_id="TEST_LT",
                rule_name="Test Less Than",
                conditions=[RuleCondition(fact="val", operator="<", value=50)],
                conclusion=RuleConclusion(fact="is_lt", value=True),
            ),
            ClinicalRule(
                rule_id="TEST_LTE",
                rule_name="Test Less Than Equal",
                conditions=[RuleCondition(fact="val", operator="<=", value=50)],
                conclusion=RuleConclusion(fact="is_lte", value=True),
            ),
            ClinicalRule(
                rule_id="TEST_GT",
                rule_name="Test Greater Than",
                conditions=[RuleCondition(fact="val", operator=">", value=50)],
                conclusion=RuleConclusion(fact="is_gt", value=True),
            ),
            ClinicalRule(
                rule_id="TEST_GTE",
                rule_name="Test Greater Than Equal",
                conditions=[RuleCondition(fact="val", operator=">=", value=50)],
                conclusion=RuleConclusion(fact="is_gte", value=True),
            ),
        ]
        eng = SymbolicRuleEngine(rules=rules)

        # 49
        r49 = {r.rule_id for r in eng.evaluate({"val": 49}).triggered_rules}
        assert r49 == {"TEST_LT", "TEST_LTE"}

        # 50
        r50 = {r.rule_id for r in eng.evaluate({"val": 50}).triggered_rules}
        assert r50 == {"TEST_LTE", "TEST_GTE"}

        # 51
        r51 = {r.rule_id for r in eng.evaluate({"val": 51}).triggered_rules}
        assert r51 == {"TEST_GT", "TEST_GTE"}

    def test_contains_and_not_contains(self) -> None:
        rules = [
            ClinicalRule(
                rule_id="HAS_MET",
                rule_name="Has Metformin",
                conditions=[RuleCondition(fact="medications", operator="contains", value="metformin")],
                conclusion=RuleConclusion(fact="has_metformin", value=True),
            ),
            ClinicalRule(
                rule_id="NO_MET",
                rule_name="No Metformin",
                conditions=[RuleCondition(fact="medications", operator="not_contains", value="metformin")],
                conclusion=RuleConclusion(fact="no_metformin", value=True),
            ),
        ]
        eng = SymbolicRuleEngine(rules=rules)

        # Patient with metformin in list
        res_with = eng.evaluate({"medications": ["aspirin", "Metformin", "atorvastatin"]})
        assert [r.rule_id for r in res_with.triggered_rules] == ["HAS_MET"]

        # Patient without metformin
        res_without = eng.evaluate({"medications": ["lisinopril", "atorvastatin"]})
        assert [r.rule_id for r in res_without.triggered_rules] == ["NO_MET"]

        # Patient with empty medications list
        res_empty = eng.evaluate({"medications": []})
        assert [r.rule_id for r in res_empty.triggered_rules] == ["NO_MET"]

    def test_and_logic_vs_or_logic(self) -> None:
        rule_and = ClinicalRule(
            rule_id="RULE_AND",
            rule_name="AND Logic",
            logic="AND",
            conditions=[
                RuleCondition(fact="a", operator="==", value=True),
                RuleCondition(fact="b", operator="==", value=True),
            ],
            conclusion=RuleConclusion(fact="out_and", value=True),
        )
        rule_or = ClinicalRule(
            rule_id="RULE_OR",
            rule_name="OR Logic",
            logic="OR",
            conditions=[
                RuleCondition(fact="a", operator="==", value=True),
                RuleCondition(fact="b", operator="==", value=True),
            ],
            conclusion=RuleConclusion(fact="out_or", value=True),
        )
        eng = SymbolicRuleEngine(rules=[rule_and, rule_or])

        # a=True, b=False -> Only OR fires
        res1 = {r.rule_id for r in eng.evaluate({"a": True, "b": False}).triggered_rules}
        assert res1 == {"RULE_OR"}

        # a=True, b=True -> Both fire
        res2 = {r.rule_id for r in eng.evaluate({"a": True, "b": True}).triggered_rules}
        assert res2 == {"RULE_AND", "RULE_OR"}

        # a=False, b=False -> Neither fires
        res3 = {r.rule_id for r in eng.evaluate({"a": False, "b": False}).triggered_rules}
        assert res3 == set()

    def test_missing_facts_null_safety(self) -> None:
        rule = ClinicalRule(
            rule_id="TEST_NULL",
            rule_name="Null Safe",
            conditions=[RuleCondition(fact="egfr", operator="<", value=30)],
            conclusion=RuleConclusion(fact="low_egfr", value=True),
        )
        eng = SymbolicRuleEngine(rules=[rule])
        # Missing or None value should not crash and should not trigger < 30
        res1 = eng.evaluate({})
        assert len(res1.triggered_rules) == 0

        res2 = eng.evaluate({"egfr": None})
        assert len(res2.triggered_rules) == 0


# --------------------------------------------------------------------------- #
# 2. Mandatory Boundary Threshold Tests
# --------------------------------------------------------------------------- #

class TestMandatoryBoundaries:
    def test_egfr_boundary_29_30_31(self, engine: SymbolicRuleEngine) -> None:
        """R001 is eGFR < 30 with metformin."""
        base_patient = {"medications": ["metformin"]}

        # 29 -> R001 fires
        r29 = [r.rule_id for r in engine.evaluate({**base_patient, "egfr": 29}).triggered_rules]
        assert "R001" in r29

        # 30 -> R001 does NOT fire (it's strict < 30)
        r30 = [r.rule_id for r in engine.evaluate({**base_patient, "egfr": 30}).triggered_rules]
        assert "R001" not in r30

        # 31 -> R001 does NOT fire
        r31 = [r.rule_id for r in engine.evaluate({**base_patient, "egfr": 31}).triggered_rules]
        assert "R001" not in r31

    def test_potassium_boundary_54_55_56(self, engine: SymbolicRuleEngine) -> None:
        """R015 is potassium > 5.5 with ace_inhibitor."""
        base_patient = {"medications": ["ace_inhibitor"]}

        # 5.4 -> R015 does not fire
        r54 = [r.rule_id for r in engine.evaluate({**base_patient, "potassium": 5.4}).triggered_rules]
        assert "R015" not in r54

        # 5.5 -> R015 does not fire (it is strict > 5.5)
        r55 = [r.rule_id for r in engine.evaluate({**base_patient, "potassium": 5.5}).triggered_rules]
        assert "R015" not in r55

        # 5.6 -> R015 fires
        r56 = [r.rule_id for r in engine.evaluate({**base_patient, "potassium": 5.6}).triggered_rules]
        assert "R015" in r56

    def test_uacr_boundary_29_30_31(self, engine: SymbolicRuleEngine) -> None:
        """R010 is moderate albuminuria: UACR >= 30 and < 300."""
        # 29 -> R010 does not fire
        r29 = [r.rule_id for r in engine.evaluate({"uacr": 29}).triggered_rules]
        assert "R010" not in r29

        # 30 -> R010 fires
        r30 = [r.rule_id for r in engine.evaluate({"uacr": 30}).triggered_rules]
        assert "R010" in r30

        # 31 -> R010 fires
        r31 = [r.rule_id for r in engine.evaluate({"uacr": 31}).triggered_rules]
        assert "R010" in r31

    def test_ldl_boundary_69_70_71(self, engine: SymbolicRuleEngine) -> None:
        """R026 is type2_diabetes + age 40-75 + LDL-C >= 70."""
        patient = {"type2_diabetes": True, "age": 55}

        # 69 -> R026 does not fire
        r69 = [r.rule_id for r in engine.evaluate({**patient, "ldl_c": 69}).triggered_rules]
        assert "R026" not in r69

        # 70 -> R026 fires
        r70 = [r.rule_id for r in engine.evaluate({**patient, "ldl_c": 70}).triggered_rules]
        assert "R026" in r70

        # 71 -> R026 fires
        r71 = [r.rule_id for r in engine.evaluate({**patient, "ldl_c": 71}).triggered_rules]
        assert "R026" in r71

    def test_sbp_boundary_129_130_131(self, engine: SymbolicRuleEngine) -> None:
        """R018 is CKD + systolic_bp >= 130."""
        patient = {"ckd": True, "diastolic_bp": 70}

        r129 = [r.rule_id for r in engine.evaluate({**patient, "systolic_bp": 129}).triggered_rules]
        assert "R018" not in r129

        r130 = [r.rule_id for r in engine.evaluate({**patient, "systolic_bp": 130}).triggered_rules]
        assert "R018" in r130

        r131 = [r.rule_id for r in engine.evaluate({**patient, "systolic_bp": 131}).triggered_rules]
        assert "R018" in r131

    def test_sbp_boundary_139_140_141(self, engine: SymbolicRuleEngine) -> None:
        """R022 is Stage 2 hypertension: systolic_bp >= 140."""
        r139 = [r.rule_id for r in engine.evaluate({"systolic_bp": 139, "diastolic_bp": 75}).triggered_rules]
        assert "R022" not in r139

        r140 = [r.rule_id for r in engine.evaluate({"systolic_bp": 140, "diastolic_bp": 75}).triggered_rules]
        assert "R022" in r140

        r141 = [r.rule_id for r in engine.evaluate({"systolic_bp": 141, "diastolic_bp": 75}).triggered_rules]
        assert "R022" in r141

    def test_dbp_boundary_79_80_81(self, engine: SymbolicRuleEngine) -> None:
        """R019 is CKD + diastolic_bp >= 80."""
        patient = {"ckd": True, "systolic_bp": 115}

        r79 = [r.rule_id for r in engine.evaluate({**patient, "diastolic_bp": 79}).triggered_rules]
        assert "R019" not in r79

        r80 = [r.rule_id for r in engine.evaluate({**patient, "diastolic_bp": 80}).triggered_rules]
        assert "R019" in r80

        r81 = [r.rule_id for r in engine.evaluate({**patient, "diastolic_bp": 81}).triggered_rules]
        assert "R019" in r81

    def test_dbp_boundary_89_90_91(self, engine: SymbolicRuleEngine) -> None:
        """R023 is Stage 2 hypertension: diastolic_bp >= 90."""
        r89 = [r.rule_id for r in engine.evaluate({"diastolic_bp": 89, "systolic_bp": 115}).triggered_rules]
        assert "R023" not in r89

        r90 = [r.rule_id for r in engine.evaluate({"diastolic_bp": 90, "systolic_bp": 115}).triggered_rules]
        assert "R023" in r90

        r91 = [r.rule_id for r in engine.evaluate({"diastolic_bp": 91, "systolic_bp": 115}).triggered_rules]
        assert "R023" in r91


# --------------------------------------------------------------------------- #
# 3. Forward Chaining & Reasoning Trace Tests
# --------------------------------------------------------------------------- #

class TestForwardChainingAndSafety:
    def test_forward_chaining_multi_step(self) -> None:
        """Rule 1 infers fact A; Rule 2 triggers only once fact A is asserted."""
        rule1 = ClinicalRule(
            rule_id="STEP_1",
            rule_name="Infer Severe CKD Stage",
            priority="high",
            conditions=[RuleCondition(fact="egfr", operator="<", value=30)],
            conclusion=RuleConclusion(type="risk", fact="severe_ckd_stage", value=True),
        )
        rule2 = ClinicalRule(
            rule_id="STEP_2",
            rule_name="Dependent Specialist Referral",
            priority="high",
            conditions=[
                RuleCondition(fact="severe_ckd_stage", operator="==", value=True),
                RuleCondition(fact="medications", operator="contains", value="metformin"),
            ],
            conclusion=RuleConclusion(type="referral", fact="urgent_nephrology_consult", value=True),
        )
        eng = SymbolicRuleEngine(rules=[rule1, rule2])

        res = eng.evaluate({"egfr": 22, "medications": ["metformin"]})
        assert [r.rule_id for r in res.triggered_rules] == ["STEP_1", "STEP_2"]
        derived_names = [f.name for f in res.derived_facts]
        assert "severe_ckd_stage" in derived_names
        assert "urgent_nephrology_consult" in derived_names
        assert len(res.reasoning_trace) == 2
        assert res.reasoning_trace[0].rule_id == "STEP_1"
        assert res.reasoning_trace[1].rule_id == "STEP_2"

    def test_duplicate_derived_facts_suppression(self) -> None:
        """Each rule must fire at most once, and facts are recorded without infinite looping."""
        rule = ClinicalRule(
            rule_id="R_ONCE",
            rule_name="Fire Once",
            priority="critical",
            conditions=[RuleCondition(fact="x", operator="==", value=1)],
            conclusion=RuleConclusion(fact="x_is_1", value=True),
        )
        eng = SymbolicRuleEngine(rules=[rule])
        res = eng.evaluate({"x": 1}, max_iterations=5)
        assert len(res.triggered_rules) == 1
        assert len(res.derived_facts) == 1

    def test_priority_ordering(self) -> None:
        """Critical rules execute/sort before high, high before medium."""
        r_med = ClinicalRule(
            rule_id="R_MED", rule_name="Med", priority="medium",
            conditions=[RuleCondition(fact="x", operator="==", value=1)],
            conclusion=RuleConclusion(fact="med_fact", value=True),
        )
        r_crit = ClinicalRule(
            rule_id="R_CRIT", rule_name="Crit", priority="critical",
            conditions=[RuleCondition(fact="x", operator="==", value=1)],
            conclusion=RuleConclusion(fact="crit_fact", value=True),
        )
        r_high = ClinicalRule(
            rule_id="R_HIGH", rule_name="High", priority="high",
            conditions=[RuleCondition(fact="x", operator="==", value=1)],
            conclusion=RuleConclusion(fact="high_fact", value=True),
        )
        eng = SymbolicRuleEngine(rules=[r_med, r_crit, r_high])
        ordered_ids = [r.rule_id for r in eng.rules]
        assert ordered_ids == ["R_CRIT", "R_HIGH", "R_MED"]

    def test_critical_alerts_extraction(self, engine: SymbolicRuleEngine) -> None:
        """Patient triggering R001 must produce a critical alert."""
        patient = {
            "medications": ["metformin"],
            "egfr": 22,
        }
        res = engine.evaluate(patient)
        assert any(r.rule_id == "R001" for r in res.triggered_rules)
        alert_names = [a.name for a in res.critical_alerts]
        assert "metformin_contraindicated" in alert_names
