"""
app/reasoning/models.py

Pydantic models for the Symbolic Clinical Reasoning Engine.

Defines schemas for Patient Profiles, Clinical Rules from external knowledge bases (rules.csv),
Evaluation Conditions, Derived Clinical Facts, and Explainable Reasoning Traces.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, model_validator


class RuleSeverity(str, Enum):
    """Clinical severity level for rule findings and derived facts."""

    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class RuleCategory(str, Enum):
    """Clinical category of the rule."""

    CONTRAINDICATION = "contraindication"
    DRUG_INTERACTION = "drug_interaction"
    DIAGNOSIS = "diagnosis"
    TREATMENT_RECOMMENDATION = "treatment_recommendation"
    THERAPY_INDICATION = "therapy_indication"
    RISK_STRATIFICATION = "risk_stratification"
    RISK_CLASSIFICATION = "risk_classification"
    RISK_ALERT = "risk_alert"
    SAFETY_ALERT = "safety_alert"
    MONITORING = "monitoring"
    SAFETY_MONITORING = "safety_monitoring"
    SAFETY_CONSIDERATION = "safety_consideration"
    SCREENING = "screening"
    CLASSIFICATION = "classification"
    DECISION_SUPPORT = "decision_support"
    REFERRAL = "referral"
    OTHER = "other"


class RuleOperator(str, Enum):
    """Supported comparison operators for rule conditions."""

    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    GREATER_EQUAL = ">="
    LESS_THAN = "<"
    LESS_EQUAL = "<="
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IN = "in"
    NOT_IN = "not_in"
    IS_EMPTY = "is_empty"
    IS_NOT_EMPTY = "is_not_empty"
    BETWEEN = "between"


class PatientProfile(BaseModel):
    """Structured clinical representation of a patient."""

    patient_id: str = Field(..., description="Unique patient identifier.")
    name: Optional[str] = Field(default=None, description="Patient name.")
    age: Optional[int] = Field(default=None, ge=0, le=130, description="Patient age in years.")
    gender: Optional[Literal["male", "female", "other"]] = Field(default=None)

    # Clinical measurements
    vitals: Dict[str, float] = Field(
        default_factory=dict,
        description="Vital signs (e.g., systolic_bp, diastolic_bp, heart_rate, bmi).",
    )
    lab_results: Dict[str, float] = Field(
        default_factory=dict,
        description="Laboratory values (e.g., hba1c, egfr, potassium, serum_creatinine, uacr, ldl_c).",
    )

    # Clinical history & states
    conditions: List[str] = Field(
        default_factory=list,
        description="Diagnosed conditions (e.g., 'Type 2 Diabetes Mellitus', 'Essential Hypertension').",
    )
    symptoms: List[str] = Field(
        default_factory=list,
        description="Reported symptoms (e.g., 'Fatigue', 'Increased Thirst', 'Headache').",
    )
    current_medications: List[str] = Field(
        default_factory=list,
        description="Active medications (e.g., 'Metformin', 'Lisinopril', 'Spironolactone').",
    )
    allergies: List[str] = Field(default_factory=list)
    extra_facts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary additional clinical facts.",
    )


class RuleCondition(BaseModel):
    """Atomic condition evaluated against working memory."""

    fact: str = Field(
        ...,
        description="Fact identifier in working memory (e.g., 'egfr', 'medications', 'type2_diabetes').",
    )
    operator: Union[RuleOperator, str] = Field(..., description="Comparison operator.")
    value: Any = Field(default=None, description="Target comparison value.")
    fact_path: Optional[str] = Field(
        default=None,
        description="Dot-notated path in working memory (backward compatibility alias for fact).",
    )

    @model_validator(mode="before")
    @classmethod
    def sync_fact_and_path(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "fact" in data and not data.get("fact_path"):
                data["fact_path"] = data["fact"]
            elif "fact_path" in data and not data.get("fact"):
                data["fact"] = data["fact_path"]
        return data


class RuleConclusion(BaseModel):
    """Deterministic conclusion asserted when rule conditions are satisfied."""

    type: str = Field(default="inference", description="Conclusion type (e.g. contraindication, safety_alert, therapy_indication).")
    fact: str = Field(..., description="Fact asserted into derived facts set.")
    value: Any = Field(default=True, description="Value of the derived fact.")


class DerivedFact(BaseModel):
    """A verified clinical fact or alert deduced by the rule engine."""

    fact_id: str = Field(..., description="Unique identifier for the derived fact.")
    category: str = Field(default="inference", description="Category (e.g., contraindication, safety_alert).")
    name: str = Field(..., description="Concise clinical finding title or fact key.")
    value: Any = Field(default=True, description="Value or status of the derived fact.")
    severity: Union[RuleSeverity, str] = Field(default=RuleSeverity.MODERATE)
    explanation: str = Field(default="", description="Clinical justification and actionable guidance.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    supporting_rules: List[str] = Field(default_factory=list, description="IDs of rules that inferred this fact.")
    provenance: Dict[str, Any] = Field(
        default_factory=dict,
        description="Guideline citation metadata (guideline_name, section, reference).",
    )


class ClinicalRule(BaseModel):
    """Deterministic clinical logic rule grounded in external knowledge base (rules.csv)."""

    rule_id: str = Field(..., min_length=1, description="Unique rule code (e.g., R001, R002).")
    rule_name: str = Field(..., min_length=1, description="Descriptive rule identifier.")
    domain: str = Field(default="clinical", description="Clinical domain (e.g. diabetes_kidney, hypertension).")
    priority: str = Field(default="medium", description="Priority level: critical, high, medium, low.")
    conditions: List[RuleCondition] = Field(..., min_length=1)
    logic: Literal["AND", "OR"] = Field(default="AND", description="Boolean aggregation logic for conditions.")
    conclusion: Optional[RuleConclusion] = Field(default=None, description="Primary conclusion asserted by this rule.")
    explanation: str = Field(default="", description="Clinical explanation and guideline rationale.")
    evidence_source: str = Field(default="", description="Authoritative clinical guideline or regulatory source.")
    evidence_reference: str = Field(default="", description="Guideline section or reference tag (e.g. ADA2026_S11).")
    validation_status: str = Field(default="validated", description="Clinical validation status (validated/needs_validation).")

    # Backward compatibility aliases for existing orchestrator/guardrail consumers
    name: Optional[str] = None
    description: Optional[str] = None
    guideline_source: Optional[str] = None
    section: Optional[str] = None
    page_number: Optional[int] = None
    category: Optional[str] = None
    severity: Optional[str] = None
    condition_logic: Optional[Literal["ALL", "ANY"]] = None
    conclusions: Optional[List[DerivedFact]] = None
    confidence: float = 1.0

    @model_validator(mode="before")
    @classmethod
    def populate_compatibility_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Rule name
            r_name = data.get("rule_name") or data.get("name") or "clinical_rule"
            data["rule_name"] = r_name
            if not data.get("name"):
                data["name"] = r_name

            # Explanation / description
            expl = data.get("explanation") or data.get("description") or ""
            data["explanation"] = expl
            if not data.get("description"):
                data["description"] = expl

            # Evidence / guideline source
            ev_src = data.get("evidence_source") or data.get("guideline_source") or ""
            data["evidence_source"] = ev_src
            if not data.get("guideline_source"):
                data["guideline_source"] = ev_src

            # Evidence reference / section
            ev_ref = data.get("evidence_reference") or data.get("section") or ""
            data["evidence_reference"] = ev_ref
            if not data.get("section"):
                data["section"] = ev_ref

            # Condition logic vs logic
            logic = data.get("logic")
            cond_logic = data.get("condition_logic")
            if logic:
                data["logic"] = logic.upper()
                data["condition_logic"] = "ALL" if logic.upper() == "AND" else "ANY"
            elif cond_logic:
                data["condition_logic"] = cond_logic.upper()
                data["logic"] = "AND" if cond_logic.upper() == "ALL" else "OR"
            else:
                data["logic"] = "AND"
                data["condition_logic"] = "ALL"

            # Priority / severity
            prio = str(data.get("priority", "medium")).lower()
            data["priority"] = prio
            if not data.get("severity"):
                data["severity"] = prio

            # Conclusion & conclusions
            conc = data.get("conclusion")
            conclusions_list = data.get("conclusions")
            if conc and not conclusions_list:
                conc_fact = conc.get("fact") if isinstance(conc, dict) else getattr(conc, "fact", "")
                conc_type = conc.get("type", "inference") if isinstance(conc, dict) else getattr(conc, "type", "inference")
                conc_val = conc.get("value", True) if isinstance(conc, dict) else getattr(conc, "value", True)
                derived = DerivedFact(
                    fact_id=f"FACT_{data.get('rule_id', '')}_{conc_fact}",
                    category=conc_type,
                    name=conc_fact,
                    value=conc_val,
                    severity=prio,
                    explanation=expl,
                    supporting_rules=[str(data.get("rule_id", ""))],
                    provenance={
                        "source": ev_src,
                        "reference": ev_ref,
                    },
                )
                data["conclusions"] = [derived]
                if not data.get("category"):
                    data["category"] = conc_type
            elif conclusions_list and not conc:
                first = conclusions_list[0]
                if isinstance(first, dict):
                    data["conclusion"] = RuleConclusion(
                        type=first.get("category", "inference"),
                        fact=first.get("name", first.get("fact_id", "")),
                        value=first.get("value", True),
                    )
                else:
                    data["conclusion"] = RuleConclusion(
                        type=getattr(first, "category", "inference"),
                        fact=getattr(first, "name", getattr(first, "fact_id", "")),
                        value=getattr(first, "value", True),
                    )
                if not data.get("category"):
                    data["category"] = data["conclusion"].type

        return data


class ConditionMatchDetail(BaseModel):
    """Record of an evaluated condition for explainability trace."""

    fact_path: str
    operator: str
    expected_value: Any
    actual_value: Any
    matched: bool


class ReasoningTraceStep(BaseModel):
    """A single execution step in the symbolic inference trace."""

    step_number: int
    rule_id: str
    rule_name: str
    category: str
    severity: str
    priority: str = "medium"
    conditions_satisfied: bool = True
    derived_fact: str = ""
    evidence_reference: str = ""
    matched_conditions: List[ConditionMatchDetail]
    derived_facts: List[DerivedFact]
    explanation: str
    guideline_citation: str


class ReasoningResult(BaseModel):
    """Complete output produced by the Symbolic Clinical Reasoning Engine."""

    patient_id: str
    total_rules_evaluated: int
    triggered_rules_count: int
    triggered_rules: List[ClinicalRule]
    derived_facts: List[DerivedFact]
    critical_alerts: List[DerivedFact] = Field(
        default_factory=list,
        description="Subset of derived facts with severity == 'critical' or from safety alert rules.",
    )
    reasoning_trace: List[ReasoningTraceStep]
    evidence_sources: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Referenced clinical guidelines with page provenance.",
    )
    execution_time_ms: float = Field(default=0.0)
