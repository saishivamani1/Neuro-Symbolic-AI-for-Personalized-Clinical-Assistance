"""
app/knowledge_graph/builder.py

Medical Knowledge Graph dataset seeder and schema builder.

Populates Neo4j with a controlled, evidence-grounded synthetic clinical ontology
covering Diseases, Symptoms, Drugs, Tests, Risk Factors, and Clinical Guidelines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.logging import get_logger
from app.knowledge_graph.models import MedicalEntity, MedicalRelationship
from app.knowledge_graph.service import KnowledgeGraphService, get_kg_service

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Synthetic Sample Clinical Dataset
# --------------------------------------------------------------------------- #

SAMPLE_ENTITIES: List[MedicalEntity] = [
    # Diseases
    MedicalEntity(
        id="disease_type_2_diabetes",
        name="Type 2 Diabetes Mellitus",
        type="Disease",
        description="Chronic metabolic disorder characterized by hyperglycemia, insulin resistance, and relative lack of insulin.",
        source_document="clinical_guidelines_sample.pdf",
        page_number=2,
        section="Type 2 Diabetes Mellitus Management",
        confidence=1.0,
    ),
    MedicalEntity(
        id="disease_essential_hypertension",
        name="Essential Hypertension",
        type="Disease",
        description="Persistent elevation of systemic blood pressure (SBP >= 140 mmHg or DBP >= 90 mmHg).",
        source_document="clinical_guidelines_sample.pdf",
        page_number=1,
        section="Management of Essential Hypertension",
        confidence=1.0,
    ),
    MedicalEntity(
        id="disease_chronic_kidney_disease",
        name="Chronic Kidney Disease",
        type="Disease",
        description="Gradual loss of kidney function characterized by persistent reduction in estimated GFR (< 60 mL/min/1.73m²).",
        source_document="clinical_guidelines_sample.pdf",
        page_number=2,
        section="Renal Impairment Protocols",
        confidence=0.95,
    ),
    MedicalEntity(
        id="disease_heart_failure",
        name="Heart Failure",
        type="Disease",
        description="Complex clinical syndrome resulting from structural or functional impairment of ventricular filling or ejection.",
        source_document="clinical_guidelines_sample.pdf",
        page_number=2,
        section="Cardiovascular Risk Reduction",
        confidence=0.95,
    ),

    # Symptoms
    MedicalEntity(
        id="symptom_fatigue",
        name="Fatigue",
        type="Symptom",
        description="Persistent feeling of tiredness or exhaustion not relieved by rest.",
        source_document="clinical_guidelines_sample.pdf",
        page_number=2,
        confidence=0.9,
    ),
    MedicalEntity(
        id="symptom_increased_thirst",
        name="Increased Thirst",
        type="Symptom",
        description="Polydipsia, excessive sensation of thirst commonly associated with osmotic diuresis.",
        source_document="clinical_guidelines_sample.pdf",
        page_number=2,
        confidence=0.95,
    ),
    MedicalEntity(
        id="symptom_frequent_urination",
        name="Frequent Urination",
        type="Symptom",
        description="Polyuria, abnormally large or frequent passage of urine.",
        source_document="clinical_guidelines_sample.pdf",
        page_number=2,
        confidence=0.95,
    ),
    MedicalEntity(
        id="symptom_headache",
        name="Headache",
        type="Symptom",
        description="Occipital or general cephalalgia associated with elevated blood pressure spikes.",
        source_document="clinical_guidelines_sample.pdf",
        page_number=1,
        confidence=0.85,
    ),
    MedicalEntity(
        id="symptom_blurred_vision",
        name="Blurred Vision",
        type="Symptom",
        description="Decreased visual acuity secondary to acute lens hydration changes or retinopathy.",
        source_document="clinical_guidelines_sample.pdf",
        page_number=2,
        confidence=0.9,
    ),

    # Drugs
    MedicalEntity(
        id="drug_metformin",
        name="Metformin",
        type="Drug",
        description="Biguanide oral antidiabetic agent that decreases hepatic glucose production and increases peripheral insulin sensitivity.",
        source_document="clinical_guidelines_sample.pdf",
        page_number=2,
        section="Glycemic Targets and First-Line Management",
        confidence=1.0,
    ),
    MedicalEntity(
        id="drug_lisinopril",
        name="Lisinopril",
        type="Drug",
        description="Angiotensin-converting enzyme (ACE) inhibitor used in hypertension, heart failure, and diabetic nephropathy.",
        source_document="clinical_guidelines_sample.pdf",
        page_number=1,
        section="Pharmacological First-Line Therapies",
        confidence=1.0,
    ),
    MedicalEntity(
        id="drug_losartan",
        name="Losartan",
        type="Drug",
        description="Angiotensin II receptor blocker (ARB) providing antihypertensive and nephroprotective effects.",
        source_document="clinical_guidelines_sample.pdf",
        page_number=1,
        section="Pharmacological First-Line Therapies",
        confidence=1.0,
    ),
    MedicalEntity(
        id="drug_amlodipine",
        name="Amlodipine",
        type="Drug",
        description="Dihydropyridine calcium channel blocker (CCB) causing peripheral arterial vasodilation.",
        source_document="clinical_guidelines_sample.pdf",
        page_number=1,
        section="Pharmacological First-Line Therapies",
        confidence=1.0,
    ),
    MedicalEntity(
        id="drug_spironolactone",
        name="Spironolactone",
        type="Drug",
        description="Potassium-sparing aldosterone receptor antagonist diuretic.",
        source_document="clinical_guidelines_sample.pdf",
        page_number=3,
        section="Critical Drug Interactions",
        confidence=1.0,
    ),
    MedicalEntity(
        id="drug_empagliflozin",
        name="Empagliflozin",
        type="Drug",
        description="Sodium-glucose cotransporter-2 (SGLT2) inhibitor with proven cardiovascular and renal benefits.",
        source_document="clinical_guidelines_sample.pdf",
        page_number=2,
        section="Cardiovascular Risk Reduction",
        confidence=0.95,
    ),

    # Medical Tests
    MedicalEntity(
        id="test_hba1c",
        name="HbA1c Test",
        type="MedicalTest",
        description="Glycated hemoglobin measurement reflecting average glycemia over the preceding 2 to 3 months (target < 7.0%).",
        source_document="clinical_guidelines_sample.pdf",
        page_number=2,
        section="Glycemic Targets",
        confidence=1.0,
    ),
    MedicalEntity(
        id="test_blood_pressure",
        name="Blood Pressure Measurement",
        type="MedicalTest",
        description="Standardized clinic or ambulatory sphygmomanometry.",
        source_document="clinical_guidelines_sample.pdf",
        page_number=1,
        section="Diagnostic Criteria",
        confidence=1.0,
    ),
    MedicalEntity(
        id="test_egfr",
        name="eGFR Test",
        type="MedicalTest",
        description="Estimated Glomerular Filtration Rate used to stage renal impairment.",
        source_document="clinical_guidelines_sample.pdf",
        page_number=2,
        section="Renal Impairment and Contraindications",
        confidence=1.0,
    ),
    MedicalEntity(
        id="test_serum_potassium",
        name="Serum Potassium Test",
        type="MedicalTest",
        description="Serum electrolyte monitoring required when initiating ACE inhibitors or ARBs.",
        source_document="clinical_guidelines_sample.pdf",
        page_number=3,
        section="Safety Monitoring",
        confidence=1.0,
    ),

    # Risk Factors & Clinical Conditions
    MedicalEntity(
        id="risk_obesity",
        name="Obesity",
        type="RiskFactor",
        description="BMI >= 30 kg/m², major predisposing factor for insulin resistance and arterial stiffness.",
        source_document="clinical_guidelines_sample.pdf",
        page_number=2,
        confidence=0.9,
    ),
    MedicalEntity(
        id="condition_severe_renal_impairment",
        name="Severe Renal Impairment",
        type="Condition",
        description="eGFR < 30 mL/min/1.73m², absolute contraindication for Metformin due to lactic acidosis risk.",
        source_document="clinical_guidelines_sample.pdf",
        page_number=2,
        section="Renal Impairment and Metformin Contraindications",
        confidence=1.0,
    ),

    # Guidelines
    MedicalEntity(
        id="guideline_diabetes_2024",
        name="ADA Standards of Care in Diabetes",
        type="Guideline",
        description="American Diabetes Association annual clinical management recommendations.",
        source_document="clinical_guidelines_sample.pdf",
        page_number=2,
        confidence=1.0,
    ),
    MedicalEntity(
        id="guideline_hypertension_2023",
        name="AHA/ACC Hypertension Clinical Practice Guidelines",
        type="Guideline",
        description="Comprehensive guideline for the prevention, detection, evaluation, and management of high blood pressure.",
        source_document="clinical_guidelines_sample.pdf",
        page_number=1,
        confidence=1.0,
    ),
]

SAMPLE_RELATIONSHIPS: List[MedicalRelationship] = [
    # Diabetes -> Symptoms
    MedicalRelationship(
        source_id="disease_type_2_diabetes",
        type="HAS_SYMPTOM",
        target_id="symptom_fatigue",
        confidence=0.9,
        source_document="clinical_guidelines_sample.pdf",
    ),
    MedicalRelationship(
        source_id="disease_type_2_diabetes",
        type="HAS_SYMPTOM",
        target_id="symptom_increased_thirst",
        confidence=0.95,
        source_document="clinical_guidelines_sample.pdf",
    ),
    MedicalRelationship(
        source_id="disease_type_2_diabetes",
        type="HAS_SYMPTOM",
        target_id="symptom_frequent_urination",
        confidence=0.95,
        source_document="clinical_guidelines_sample.pdf",
    ),

    # Diabetes -> Tests & Treatments
    MedicalRelationship(
        source_id="disease_type_2_diabetes",
        type="RECOMMENDED_TEST",
        target_id="test_hba1c",
        description="Evaluate glycemic control every 3 to 6 months.",
        source_document="clinical_guidelines_sample.pdf",
    ),
    MedicalRelationship(
        source_id="disease_type_2_diabetes",
        type="RECOMMENDED_TEST",
        target_id="test_egfr",
        description="Annual renal function monitoring.",
        source_document="clinical_guidelines_sample.pdf",
    ),
    MedicalRelationship(
        source_id="disease_type_2_diabetes",
        type="TREATED_BY",
        target_id="drug_metformin",
        description="First-line oral pharmacotherapy in absence of renal contraindication.",
        source_document="clinical_guidelines_sample.pdf",
    ),
    MedicalRelationship(
        source_id="disease_type_2_diabetes",
        type="TREATED_BY",
        target_id="drug_empagliflozin",
        description="Indicated for patients with established ASCVD or CKD.",
        source_document="clinical_guidelines_sample.pdf",
    ),
    MedicalRelationship(
        source_id="disease_type_2_diabetes",
        type="ASSOCIATED_WITH",
        target_id="disease_essential_hypertension",
        description="High comorbidity and shared cardiovascular risk profile.",
        source_document="clinical_guidelines_sample.pdf",
    ),
    MedicalRelationship(
        source_id="disease_type_2_diabetes",
        type="SUPPORTED_BY",
        target_id="guideline_diabetes_2024",
        source_document="clinical_guidelines_sample.pdf",
    ),

    # Hypertension -> Symptoms, Tests, Treatments
    MedicalRelationship(
        source_id="disease_essential_hypertension",
        type="HAS_SYMPTOM",
        target_id="symptom_headache",
        confidence=0.85,
        source_document="clinical_guidelines_sample.pdf",
    ),
    MedicalRelationship(
        source_id="disease_essential_hypertension",
        type="HAS_SYMPTOM",
        target_id="symptom_blurred_vision",
        confidence=0.85,
        source_document="clinical_guidelines_sample.pdf",
    ),
    MedicalRelationship(
        source_id="disease_essential_hypertension",
        type="RECOMMENDED_TEST",
        target_id="test_blood_pressure",
        description="Clinic and home blood pressure monitoring.",
        source_document="clinical_guidelines_sample.pdf",
    ),
    MedicalRelationship(
        source_id="disease_essential_hypertension",
        type="TREATED_BY",
        target_id="drug_lisinopril",
        description="First-line ACE inhibitor.",
        source_document="clinical_guidelines_sample.pdf",
    ),
    MedicalRelationship(
        source_id="disease_essential_hypertension",
        type="TREATED_BY",
        target_id="drug_losartan",
        description="First-line ARB alternative for ACE inhibitor cough.",
        source_document="clinical_guidelines_sample.pdf",
    ),
    MedicalRelationship(
        source_id="disease_essential_hypertension",
        type="TREATED_BY",
        target_id="drug_amlodipine",
        description="First-line Calcium Channel Blocker.",
        source_document="clinical_guidelines_sample.pdf",
    ),
    MedicalRelationship(
        source_id="disease_essential_hypertension",
        type="SUPPORTED_BY",
        target_id="guideline_hypertension_2023",
        source_document="clinical_guidelines_sample.pdf",
    ),

    # Contraindications & Drug Interactions
    MedicalRelationship(
        source_id="drug_metformin",
        type="CONTRAINDICATED_WITH",
        target_id="condition_severe_renal_impairment",
        description="Contraindicated when eGFR < 30 mL/min/1.73m² due to lactic acidosis risk.",
        source_document="clinical_guidelines_sample.pdf",
        page_number=2,
    ),
    MedicalRelationship(
        source_id="drug_lisinopril",
        type="CONTRAINDICATED_WITH",
        target_id="drug_spironolactone",
        description="Concurrent use significantly elevates risk of severe hyperkalemia.",
        source_document="clinical_guidelines_sample.pdf",
        page_number=3,
    ),
    MedicalRelationship(
        source_id="drug_lisinopril",
        type="RECOMMENDED_TEST",
        target_id="test_serum_potassium",
        description="Electrolyte monitoring within 1-2 weeks of ACE inhibitor initiation.",
        source_document="clinical_guidelines_sample.pdf",
        page_number=3,
    ),

    # Risk Factors
    MedicalRelationship(
        source_id="risk_obesity",
        type="RISK_FACTOR_FOR",
        target_id="disease_type_2_diabetes",
        confidence=0.95,
        source_document="clinical_guidelines_sample.pdf",
    ),
    MedicalRelationship(
        source_id="risk_obesity",
        type="RISK_FACTOR_FOR",
        target_id="disease_essential_hypertension",
        confidence=0.9,
        source_document="clinical_guidelines_sample.pdf",
    ),
]


def seed_sample_knowledge_graph(
    service: Optional[KnowledgeGraphService] = None,
) -> Dict[str, Any]:
    """Seed Neo4j with the synthetic medical ontology dataset.

    This function is idempotent and safe to run multiple times.
    """
    kg_service = service or get_kg_service()

    logger.info("Initializing schema constraints...")
    kg_service.initialize_constraints()

    logger.info("Seeding %d medical entities...", len(SAMPLE_ENTITIES))
    entity_counts_by_type: Dict[str, int] = {}
    for entity in SAMPLE_ENTITIES:
        kg_service.create_entity(entity)
        entity_counts_by_type[entity.type] = entity_counts_by_type.get(entity.type, 0) + 1

    logger.info("Seeding %d medical relationships...", len(SAMPLE_RELATIONSHIPS))
    rel_counts_by_type: Dict[str, int] = {}
    for rel in SAMPLE_RELATIONSHIPS:
        kg_service.create_relationship(rel)
        rel_counts_by_type[rel.type] = rel_counts_by_type.get(rel.type, 0) + 1

    stats = kg_service.get_stats()
    logger.info("Knowledge Graph seeding completed successfully.")

    return {
        "status": "success",
        "entities_created": len(SAMPLE_ENTITIES),
        "entities_by_type": entity_counts_by_type,
        "relationships_created": len(SAMPLE_RELATIONSHIPS),
        "relationships_by_type": rel_counts_by_type,
        "graph_stats": stats.model_dump(),
    }
