"""
app/reasoning/engine.py

Deterministic Symbolic Clinical Reasoning Engine implementing Forward Chaining Inference.

Evaluates evidence-grounded rules from external knowledge bases (rules.csv) against
structured patient facts, Knowledge Graph context, and clinical guidelines.
Generates explainable, step-by-step reasoning traces and derived clinical conclusions.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from app.core.exceptions import RuleEngineError
from app.core.logging import get_logger
from app.reasoning.loader import get_clinical_rules
from app.reasoning.models import (
    ClinicalRule,
    ConditionMatchDetail,
    DerivedFact,
    PatientProfile,
    ReasoningResult,
    ReasoningTraceStep,
    RuleCondition,
    RuleOperator,
    RuleSeverity,
)

logger = get_logger(__name__)

PRIORITY_WEIGHTS: Dict[str, int] = {
    "critical": 400,
    "high": 300,
    "medium": 200,
    "moderate": 200,
    "low": 100,
    "info": 50,
}


def get_rule_priority_weight(rule: ClinicalRule) -> int:
    """Map rule priority string to numeric weight for deterministic sorting."""
    p = str(rule.priority).lower().strip()
    return PRIORITY_WEIGHTS.get(p, 100)


class SymbolicRuleEngine:
    """Deterministic forward-chaining clinical inference engine."""

    def __init__(self, rules: Optional[List[ClinicalRule]] = None) -> None:
        self._rules: List[ClinicalRule] = []
        if rules is not None:
            self.register_rules(rules)
        else:
            loaded_rules = get_clinical_rules()
            self.register_rules(loaded_rules)

    @property
    def rules(self) -> List[ClinicalRule]:
        """Return registered rules sorted by priority (highest priority first), then rule_id."""
        return sorted(self._rules, key=lambda r: (get_rule_priority_weight(r), r.rule_id), reverse=True)

    def list_rules(self) -> List[ClinicalRule]:
        """Return registered rules (alias for rules property)."""
        return self.rules

    def register_rule(self, rule: ClinicalRule) -> None:
        """Add or replace a rule in the engine's rulebase."""
        self._rules = [r for r in self._rules if r.rule_id != rule.rule_id]
        self._rules.append(rule)
        logger.debug("Registered clinical rule: %s (%s)", rule.rule_id, rule.rule_name)

    def register_rules(self, rules: List[ClinicalRule]) -> None:
        """Batch register multiple rules using dict index."""
        rules_map = {r.rule_id: r for r in self._rules}
        for rule in rules:
            rules_map[rule.rule_id] = rule
        self._rules = list(rules_map.values())
        logger.info("Loaded %d clinical rules into symbolic reasoning engine", len(self._rules))

    def _resolve_fact(self, working_memory: Dict[str, Any], fact_name: str) -> Any:
        """Resolve a fact key from working memory, checking root, labs, and vitals."""
        # 1. Direct key match
        if fact_name in working_memory:
            return working_memory[fact_name]

        # 2. Check nested dot path if present
        if "." in fact_name:
            curr: Any = working_memory
            for part in fact_name.strip().split("."):
                if curr is None:
                    return None
                if isinstance(curr, dict):
                    curr = curr.get(part)
                elif hasattr(curr, part):
                    curr = getattr(curr, part)
                else:
                    return None
            return curr

        # 3. Check inside 'labs' or 'lab_results'
        for sub_key in ("labs", "lab_results"):
            sub_dict = working_memory.get(sub_key)
            if isinstance(sub_dict, dict) and fact_name in sub_dict:
                return sub_dict[fact_name]

        # 4. Check inside 'vitals'
        vitals = working_memory.get("vitals")
        if isinstance(vitals, dict) and fact_name in vitals:
            return vitals[fact_name]

        # 5. Laboratory aliases
        if fact_name == "potassium":
            for lab_k in ("labs", "lab_results"):
                d = working_memory.get(lab_k)
                if isinstance(d, dict) and "serum_potassium" in d:
                    return d["serum_potassium"]
            if "serum_potassium" in working_memory:
                return working_memory["serum_potassium"]

        if fact_name == "creatinine":
            for lab_k in ("labs", "lab_results"):
                d = working_memory.get(lab_k)
                if isinstance(d, dict) and "serum_creatinine" in d:
                    return d["serum_creatinine"]
            if "serum_creatinine" in working_memory:
                return working_memory["serum_creatinine"]

        # 6. Condition flags inferred from conditions list
        cond_list = working_memory.get("conditions", [])
        if isinstance(cond_list, list):
            norm_conds = " ".join(str(c).lower() for c in cond_list)
            if fact_name == "type2_diabetes" and ("diabetes" in norm_conds or "t2d" in norm_conds):
                return True
            if fact_name == "ckd" and ("chronic kidney disease" in norm_conds or "ckd" in norm_conds):
                return True
            if fact_name == "heart_failure" and "heart failure" in norm_conds:
                return True

        # 7. Medication list aliases and class expansion
        med_list = working_memory.get("medications") or working_memory.get("current_medications") or []
        if isinstance(med_list, list):
            norm_meds = [str(m).strip().lower() for m in med_list]
            # If checking fact medications, expand drug class equivalents
            if fact_name in ("medications", "current_medications"):
                expanded_meds = list(norm_meds)
                for m in norm_meds:
                    if any(x in m for x in ("lisinopril", "ramipril", "enalapril", "benazepril")):
                        expanded_meds.append("ace_inhibitor")
                    if any(x in m for x in ("losartan", "valsartan", "irbesartan", "candesartan")):
                        expanded_meds.append("arb")
                    if any(x in m for x in ("furosemide", "torsemide", "bumetanide")):
                        expanded_meds.append("loop_diuretic")
                    if any(x in m for x in ("empagliflozin", "dapagliflozin", "canagliflozin")):
                        expanded_meds.append("sglt2_inhibitor")
                    if any(x in m for x in ("ibuprofen", "naproxen", "meloxicam", "celecoxib")):
                        expanded_meds.append("nsaid")
                return expanded_meds

        return None

    def _evaluate_condition(
        self, condition: RuleCondition, working_memory: Dict[str, Any]
    ) -> Tuple[bool, Any]:
        """Evaluate a single rule condition deterministically against working memory.

        Returns (matched: bool, actual_value: Any).
        """
        fact_key = condition.fact or condition.fact_path or ""
        actual = self._resolve_fact(working_memory, fact_key)
        op = condition.operator.value if hasattr(condition.operator, "value") else str(condition.operator)
        expected = condition.value

        # Handle emptiness operators
        if op == RuleOperator.IS_EMPTY.value:
            matched = actual is None or actual == "" or actual == [] or actual == {}
            return matched, actual

        if op == RuleOperator.IS_NOT_EMPTY.value:
            matched = actual is not None and actual != "" and actual != [] and actual != {}
            return matched, actual

        # Equality checks
        if op in ("==", RuleOperator.EQUALS.value):
            if isinstance(actual, bool) and isinstance(expected, bool):
                return actual is expected, actual
            if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
                return float(actual) == float(expected), actual
            if actual is None or expected is None:
                return actual == expected, actual
            return str(actual).strip().lower() == str(expected).strip().lower(), actual

        if op in ("!=", RuleOperator.NOT_EQUALS.value):
            if isinstance(actual, bool) and isinstance(expected, bool):
                return actual is not expected, actual
            if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
                return float(actual) != float(expected), actual
            if actual is None or expected is None:
                return actual != expected, actual
            return str(actual).strip().lower() != str(expected).strip().lower(), actual

        # Numeric comparisons (<, <=, >, >=)
        if op in ("<", RuleOperator.LESS_THAN.value):
            if actual is None or expected is None:
                return False, actual
            try:
                return float(actual) < float(expected), actual
            except (ValueError, TypeError):
                return False, actual

        if op in ("<=", RuleOperator.LESS_EQUAL.value):
            if actual is None or expected is None:
                return False, actual
            try:
                return float(actual) <= float(expected), actual
            except (ValueError, TypeError):
                return False, actual

        if op in (">", RuleOperator.GREATER_THAN.value):
            if actual is None or expected is None:
                return False, actual
            try:
                return float(actual) > float(expected), actual
            except (ValueError, TypeError):
                return False, actual

        if op in (">=", RuleOperator.GREATER_EQUAL.value):
            if actual is None or expected is None:
                return False, actual
            try:
                return float(actual) >= float(expected), actual
            except (ValueError, TypeError):
                return False, actual

        # List / String containment
        if op in ("contains", RuleOperator.CONTAINS.value):
            if actual is None:
                return False, actual
            expected_str = str(expected).strip().lower()
            if isinstance(actual, list):
                return any(
                    expected_str == str(item).strip().lower()
                    or expected_str in str(item).strip().lower()
                    for item in actual
                ), actual
            if isinstance(actual, str):
                return expected_str in actual.strip().lower(), actual
            return False, actual

        if op in ("not_contains", RuleOperator.NOT_CONTAINS.value):
            if actual is None or actual == [] or actual == "":
                return True, actual
            expected_str = str(expected).strip().lower()
            if isinstance(actual, list):
                return not any(
                    expected_str == str(item).strip().lower()
                    or expected_str in str(item).strip().lower()
                    for item in actual
                ), actual
            if isinstance(actual, str):
                return expected_str not in actual.strip().lower(), actual
            return True, actual

        # Set membership operators
        if op in ("in", RuleOperator.IN.value):
            if isinstance(expected, list):
                norm_expected = [str(x).strip().lower() for x in expected]
                return str(actual).strip().lower() in norm_expected, actual
            return False, actual

        if op in ("not_in", RuleOperator.NOT_IN.value):
            if isinstance(expected, list):
                norm_expected = [str(x).strip().lower() for x in expected]
                return str(actual).strip().lower() not in norm_expected, actual
            return True, actual

        if op in ("between", RuleOperator.BETWEEN.value):
            if isinstance(expected, (list, tuple)) and len(expected) == 2:
                try:
                    val = float(actual)
                    low = float(expected[0])
                    high = float(expected[1])
                    return low <= val <= high, actual
                except (ValueError, TypeError):
                    return False, actual
            return False, actual

        return False, actual

    def evaluate(
        self,
        patient: Union[PatientProfile, Dict[str, Any]],
        max_iterations: int = 10,
    ) -> ReasoningResult:
        """Execute deterministic forward-chaining inference over patient facts.

        Parameters
        ----------
        patient : Union[PatientProfile, Dict[str, Any]]
            Patient clinical data dictionary or Pydantic model.
        max_iterations : int
            Iteration ceiling to guarantee termination.

        Returns
        -------
        ReasoningResult
            Inference outcome with triggered rules, derived facts,
            critical alerts, and explainability reasoning trace.
        """
        t0 = time.monotonic()

        # Build working memory dict
        if isinstance(patient, PatientProfile):
            working_memory: Dict[str, Any] = patient.model_dump()
        elif isinstance(patient, dict):
            working_memory = dict(patient)
        else:
            raise RuleEngineError(
                detail="Invalid patient object provided to rule engine.",
                context={"type": str(type(patient))},
            )

        # Flatten nested labs/vitals for fast lookup while preserving originals
        for sub_key in ("labs", "lab_results", "vitals"):
            sub_dict = working_memory.get(sub_key)
            if isinstance(sub_dict, dict):
                for k, v in sub_dict.items():
                    if k not in working_memory:
                        working_memory[k] = v

        if "medications" not in working_memory and "current_medications" in working_memory:
            working_memory["medications"] = working_memory["current_medications"]
        elif "current_medications" not in working_memory and "medications" in working_memory:
            working_memory["current_medications"] = working_memory["medications"]

        patient_id = str(working_memory.get("patient_id", "anonymous_patient"))

        # Tracking state
        triggered_rules: List[ClinicalRule] = []
        triggered_rule_ids: Set[str] = set()
        derived_facts: List[DerivedFact] = []
        derived_fact_keys: Set[str] = set()
        critical_alerts: List[DerivedFact] = []
        reasoning_trace: List[ReasoningTraceStep] = []
        evidence_sources: List[Dict[str, Any]] = []

        step_counter = 1

        # Forward chaining loop
        for iteration in range(max_iterations):
            rules_fired_in_iteration = 0
            sorted_rules = self.rules

            for rule in sorted_rules:
                if rule.rule_id in triggered_rule_ids:
                    continue  # Each rule fires at most once per patient inference

                # Evaluate rule conditions
                condition_details: List[ConditionMatchDetail] = []
                condition_matches: List[bool] = []

                for cond in rule.conditions:
                    matched, actual_val = self._evaluate_condition(cond, working_memory)
                    condition_matches.append(matched)
                    condition_details.append(
                        ConditionMatchDetail(
                            fact_path=cond.fact or cond.fact_path or "",
                            operator=cond.operator.value if hasattr(cond.operator, "value") else str(cond.operator),
                            expected_value=cond.value,
                            actual_value=actual_val,
                            matched=matched,
                        )
                    )

                is_triggered = (
                    all(condition_matches)
                    if rule.logic.upper() == "AND"
                    else any(condition_matches)
                )

                if is_triggered:
                    rules_fired_in_iteration += 1
                    triggered_rules.append(rule)
                    triggered_rule_ids.add(rule.rule_id)

                    # Extract primary conclusion
                    conc = rule.conclusion
                    conc_fact = conc.fact if conc else f"{rule.rule_id}_fired"
                    conc_val = conc.value if conc else True
                    conc_type = conc.type if conc else "inference"

                    # Assert derived fact into working memory for forward chaining
                    working_memory[conc_fact] = conc_val
                    if conc_type == "diagnosis":
                        if "conditions" not in working_memory or not isinstance(working_memory["conditions"], list):
                            working_memory["conditions"] = []
                        working_memory["conditions"].append(conc_fact)

                    if rule.conclusions and len(rule.conclusions) > 0:
                        first_c = rule.conclusions[0]
                        derived_fact_obj = DerivedFact(
                            fact_id=first_c.fact_id or f"FACT_{rule.rule_id}_{conc_fact}",
                            category=first_c.category or conc_type,
                            name=first_c.name or conc_fact,
                            value=first_c.value if first_c.value is not None else conc_val,
                            severity=first_c.severity or rule.priority,
                            explanation=first_c.explanation or rule.explanation,
                            confidence=first_c.confidence,
                            supporting_rules=[rule.rule_id],
                            provenance={
                                "evidence_source": rule.evidence_source,
                                "evidence_reference": rule.evidence_reference,
                                "domain": rule.domain,
                                "validation_status": rule.validation_status,
                            },
                        )
                    else:
                        derived_fact_obj = DerivedFact(
                            fact_id=f"FACT_{rule.rule_id}_{conc_fact}",
                            category=conc_type,
                            name=conc_fact,
                            value=conc_val,
                            severity=rule.priority,
                            explanation=rule.explanation,
                            confidence=1.0,
                            supporting_rules=[rule.rule_id],
                            provenance={
                                "evidence_source": rule.evidence_source,
                                "evidence_reference": rule.evidence_reference,
                                "domain": rule.domain,
                                "validation_status": rule.validation_status,
                            },
                        )

                    derived_facts.append(derived_fact_obj)
                    derived_fact_keys.add(conc_fact)

                    # Check for critical alerts
                    # An alert is recognized if rule priority is critical or conclusion type is safety_alert / risk_alert
                    if rule.priority == "critical" or conc_type in ("safety_alert", "risk_alert"):
                        critical_alerts.append(derived_fact_obj)

                    # Record trace step
                    citation = f"{rule.evidence_source} ({rule.evidence_reference})" if rule.evidence_reference else rule.evidence_source
                    reasoning_trace.append(
                        ReasoningTraceStep(
                            step_number=step_counter,
                            rule_id=rule.rule_id,
                            rule_name=rule.rule_name,
                            category=conc_type,
                            severity=rule.priority,
                            priority=rule.priority,
                            conditions_satisfied=True,
                            derived_fact=conc_fact,
                            evidence_reference=rule.evidence_reference,
                            matched_conditions=condition_details,
                            derived_facts=[derived_fact_obj],
                            explanation=rule.explanation,
                            guideline_citation=citation,
                        )
                    )
                    step_counter += 1

                    # Record unique evidence source
                    source_entry = {
                        "rule_id": rule.rule_id,
                        "source": rule.evidence_source,
                        "reference": rule.evidence_reference,
                        "domain": rule.domain,
                    }
                    if source_entry not in evidence_sources:
                        evidence_sources.append(source_entry)

            # Fixpoint reached: no new rules fired in this pass
            if rules_fired_in_iteration == 0:
                break

        elapsed_ms = round((time.monotonic() - t0) * 1000, 2)

        return ReasoningResult(
            patient_id=patient_id,
            total_rules_evaluated=len(self._rules),
            triggered_rules_count=len(triggered_rules),
            triggered_rules=triggered_rules,
            derived_facts=derived_facts,
            critical_alerts=critical_alerts,
            reasoning_trace=reasoning_trace,
            evidence_sources=evidence_sources,
            execution_time_ms=elapsed_ms,
        )


_cached_rule_engine: Optional[SymbolicRuleEngine] = None


def get_rule_engine() -> SymbolicRuleEngine:
    """Singleton getter for SymbolicRuleEngine."""
    global _cached_rule_engine
    if _cached_rule_engine is None:
        _cached_rule_engine = SymbolicRuleEngine()
    return _cached_rule_engine
