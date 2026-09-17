"""
app/api/routes/patients.py

Patient evaluation and symbolic reasoning endpoints.

Endpoints:
- POST /api/v1/patients/evaluate: Execute symbolic reasoning on a patient profile
- GET  /api/v1/patients/{patient_id}: Retrieve patient profile and clinical reasoning summary
- GET  /api/v1/patients/sample/profiles: List available synthetic test patient profiles
- GET  /api/v1/reasoning/rules: List all registered symbolic clinical rules
- GET  /api/v1/reasoning/rules/{rule_id}: Get details of a specific clinical rule
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Path, Query, status

from app.core.logging import get_logger
from app.reasoning.models import ClinicalRule, PatientProfile, ReasoningResult
from app.reasoning.service import get_reasoning_service
from app.schemas.response import APIResponse

router = APIRouter()
logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Synthetic Patient Profiles for Testing
# --------------------------------------------------------------------------- #

SAMPLE_PATIENTS: Dict[str, PatientProfile] = {
    "patient_ckd_metformin_alert": PatientProfile(
        patient_id="patient_ckd_metformin_alert",
        name="John Doe",
        age=68,
        gender="male",
        vitals={"systolic_bp": 142.0, "diastolic_bp": 88.0, "heart_rate": 74.0, "bmi": 31.5},
        lab_results={"hba1c": 8.4, "egfr": 24.0, "serum_potassium": 4.8, "serum_creatinine": 2.6},
        conditions=["Type 2 Diabetes Mellitus", "Chronic Kidney Disease"],
        symptoms=["Fatigue", "Increased Thirst"],
        current_medications=["Metformin", "Amlodipine"],
    ),
    "patient_hyperkalemia_ddi": PatientProfile(
        patient_id="patient_hyperkalemia_ddi",
        name="Eleanor Vance",
        age=72,
        gender="female",
        vitals={"systolic_bp": 148.0, "diastolic_bp": 92.0, "heart_rate": 68.0, "bmi": 27.2},
        lab_results={"hba1c": 6.8, "egfr": 52.0, "serum_potassium": 5.8, "serum_creatinine": 1.4},
        conditions=["Essential Hypertension", "Heart Failure"],
        symptoms=["Headache", "Shortness of Breath"],
        current_medications=["Lisinopril", "Spironolactone"],
    ),
    "patient_undiagnosed_hyperglycemia": PatientProfile(
        patient_id="patient_undiagnosed_hyperglycemia",
        name="Marcus Smith",
        age=45,
        gender="male",
        vitals={"systolic_bp": 130.0, "diastolic_bp": 82.0, "heart_rate": 78.0, "bmi": 32.0},
        lab_results={"fasting_glucose": 165.0},
        conditions=["Obesity"],
        symptoms=["Increased Thirst", "Frequent Urination", "Fatigue"],
        current_medications=[],
    ),
    "patient_healthy_normal": PatientProfile(
        patient_id="patient_healthy_normal",
        name="Sarah Jenkins",
        age=32,
        gender="female",
        vitals={"systolic_bp": 118.0, "diastolic_bp": 76.0, "heart_rate": 68.0, "bmi": 22.4},
        lab_results={"hba1c": 5.2, "egfr": 105.0, "serum_potassium": 4.1, "serum_creatinine": 0.8},
        conditions=[],
        symptoms=[],
        current_medications=[],
    ),
    "patient_final_test": PatientProfile(
        patient_id="patient_final_test",
        name="Robert Taylor",
        age=66,
        gender="male",
        vitals={"systolic_bp": 154.0, "diastolic_bp": 94.0, "heart_rate": 76.0, "bmi": 29.8},
        lab_results={"hba1c": 8.8, "egfr": 27.0, "serum_potassium": 5.4, "serum_creatinine": 2.3},
        conditions=["Type 2 Diabetes Mellitus", "Chronic Kidney Disease", "Essential Hypertension"],
        symptoms=["Fatigue", "Bilateral Lower Extremity Edema", "Mild Dyspnea"],
        current_medications=["Metformin", "Lisinopril", "Spironolactone"],
    ),
}


# --------------------------------------------------------------------------- #
# Patient Reasoning Endpoints
# --------------------------------------------------------------------------- #


@router.post(
    "/patients/evaluate",
    response_model=APIResponse[ReasoningResult],
    status_code=status.HTTP_200_OK,
    tags=["Reasoning"],
    summary="Evaluate patient profile using the Symbolic Clinical Reasoning Engine",
)
async def evaluate_patient_profile(
    profile: PatientProfile,
    include_kg: bool = Query(default=True, description="Enrich with Knowledge Graph facts"),
    include_rag: bool = Query(default=True, description="Enrich with RAG guideline evidence"),
) -> APIResponse[ReasoningResult]:
    """Execute deterministic symbolic reasoning on an arbitrary patient profile."""
    service = get_reasoning_service()
    result = service.evaluate_patient(
        patient=profile,
        include_kg=include_kg,
        include_rag=include_rag,
    )
    return APIResponse(success=True, data=result)


def find_sample_patient(identifier: str) -> Optional[PatientProfile]:
    """Locate a synthetic patient profile by exact ID, normalized ID, or patient name."""
    if not identifier:
        return None
    if identifier in SAMPLE_PATIENTS:
        return SAMPLE_PATIENTS[identifier]
    cleaned = identifier.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    for pid, profile in SAMPLE_PATIENTS.items():
        norm_pid = pid.lower().replace("-", "").replace("_", "").replace(" ", "")
        norm_name = (profile.name or "").lower().replace("-", "").replace("_", "").replace(" ", "")
        if cleaned == norm_pid or cleaned == norm_name or (cleaned and cleaned in norm_name):
            return profile
    return None


@router.get(
    "/patients/{patient_id}",
    response_model=APIResponse[ReasoningResult],
    tags=["Patients"],
    summary="Get patient profile and run symbolic reasoning",
)
async def get_patient_evaluation(
    patient_id: str = Path(..., description="Unique patient identifier"),
    include_kg: bool = Query(default=True),
    include_rag: bool = Query(default=True),
) -> APIResponse[ReasoningResult]:
    """Retrieve sample patient data or evaluate a specific patient."""
    service = get_reasoning_service()
    patient = find_sample_patient(patient_id) or SAMPLE_PATIENTS.get(
        patient_id,
        PatientProfile(
            patient_id=patient_id,
            name=f"Patient {patient_id}",
            vitals={"systolic_bp": 145.0, "diastolic_bp": 92.0},
            lab_results={"egfr": 28.0, "hba1c": 8.2},
            conditions=["Type 2 Diabetes Mellitus"],
            current_medications=["Metformin"],
            symptoms=["Fatigue"],
        ),
    )

    result = service.evaluate_patient(
        patient=patient,
        include_kg=include_kg,
        include_rag=include_rag,
    )
    return APIResponse(success=True, data=result)


@router.get(
    "/patients/{patient_id}/profile",
    response_model=APIResponse[PatientProfile],
    tags=["Patients"],
    summary="Get raw patient profile (demographics, vitals, labs, medications)",
)
async def get_patient_profile(
    patient_id: str = Path(..., description="Unique patient identifier or name"),
) -> APIResponse[PatientProfile]:
    """Retrieve patient demographic, lab, and medication details."""
    matched = find_sample_patient(patient_id)
    if matched:
        return APIResponse(success=True, data=matched)

    fallback_profile = PatientProfile(
        patient_id=patient_id,
        name=f"Patient {patient_id}",
        age=55,
        gender="Adult",
        vitals={"systolic_bp": 130.0, "diastolic_bp": 84.0},
        lab_results={"egfr": 75.0, "hba1c": 5.8},
        conditions=[],
        current_medications=[],
        symptoms=[],
    )
    return APIResponse(success=True, data=fallback_profile)


@router.get(
    "/patients/sample/profiles",
    response_model=APIResponse[List[PatientProfile]],
    tags=["Patients"],
    summary="List synthetic sample patient profiles for clinical reasoning testing",
)
async def list_sample_patients() -> APIResponse[List[PatientProfile]]:
    """List available synthetic patient profiles covering various clinical test scenarios."""
    return APIResponse(success=True, data=list(SAMPLE_PATIENTS.values()))


# --------------------------------------------------------------------------- #
# Clinical Rules Inspection Endpoints
# --------------------------------------------------------------------------- #


@router.get(
    "/reasoning/rules",
    response_model=APIResponse[List[ClinicalRule]],
    tags=["Reasoning"],
    summary="List all loaded clinical rules in the symbolic reasoning engine",
)
async def list_clinical_rules() -> APIResponse[List[ClinicalRule]]:
    """Retrieve all currently active deterministic clinical rules."""
    service = get_reasoning_service()
    rules = service.list_rules()
    return APIResponse(success=True, data=rules)


@router.get(
    "/reasoning/rules/{rule_id}",
    response_model=APIResponse[ClinicalRule],
    tags=["Reasoning"],
    summary="Get details of a specific clinical rule",
)
async def get_clinical_rule(
    rule_id: str = Path(..., description="Rule ID code"),
) -> APIResponse[ClinicalRule]:
    """Retrieve rule conditions, logic, severity, and evidence citations."""
    from app.reasoning.rules import OLD_TO_NEW_RULE_MAP

    service = get_reasoning_service()
    resolved_id = OLD_TO_NEW_RULE_MAP.get(rule_id, rule_id)
    rule = service.get_rule(resolved_id)
    if not rule:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Clinical rule '{rule_id}' not found.",
        )
    return APIResponse(success=True, data=rule)


@router.get(
    "/reasoning/evaluation",
    response_model=APIResponse[Dict[str, Any]],
    tags=["Reasoning"],
    summary="Get deterministic symbolic reasoning benchmark evaluation metrics",
)
async def get_reasoning_evaluation() -> APIResponse[Dict[str, Any]]:
    """Return benchmark metrics from evaluation_results.json across all 500 test cases."""
    import json
    from pathlib import Path

    eval_paths = [
        Path("evaluation_results.json"),
        Path("../evaluation_results.json"),
        Path(__file__).resolve().parent.parent.parent.parent / "evaluation_results.json",
    ]
    for p in eval_paths:
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                return APIResponse(success=True, data=data)
            except Exception as exc:
                logger.warning("Failed reading %s: %s", p, exc)

    return APIResponse(
        success=False,
        data={
            "error": "evaluation_results.json not found. Run 'python evaluate_symbolic_engine.py' to generate metrics."
        },
    )

