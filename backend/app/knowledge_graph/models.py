"""
app/knowledge_graph/models.py

Pydantic domain models and schemas for the Medical Knowledge Graph.

Defines schemas for clinical entities, medical relationships, graph query responses,
and graph context structures.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class EntityType(str, Enum):
    """Allowed medical entity categories."""

    PATIENT = "Patient"
    DISEASE = "Disease"
    SYMPTOM = "Symptom"
    DRUG = "Drug"
    TREATMENT = "Treatment"
    MEDICAL_TEST = "MedicalTest"
    RISK_FACTOR = "RiskFactor"
    CONDITION = "Condition"
    GUIDELINE = "Guideline"
    MEDICAL_CONCEPT = "MedicalConcept"


class RelationshipType(str, Enum):
    """Allowed medical graph relationship types."""

    HAS_SYMPTOM = "HAS_SYMPTOM"
    HAS_CONDITION = "HAS_CONDITION"
    TREATED_BY = "TREATED_BY"
    RECOMMENDED_TEST = "RECOMMENDED_TEST"
    RISK_FACTOR_FOR = "RISK_FACTOR_FOR"
    CONTRAINDICATED_WITH = "CONTRAINDICATED_WITH"
    HAS_SIDE_EFFECT = "HAS_SIDE_EFFECT"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    SUPPORTED_BY = "SUPPORTED_BY"


class MedicalEntity(BaseModel):
    """Representation of a node in the Medical Knowledge Graph."""

    id: str = Field(..., description="Unique identifier for the entity (e.g., disease_type_2_diabetes).")
    name: str = Field(..., min_length=1, description="Canonical name of the entity.")
    type: str = Field(..., description="Entity category label (e.g., Disease, Drug, Symptom).")
    description: Optional[str] = Field(default=None, description="Clinical summary or definition.")
    source_document: Optional[str] = Field(default=None, description="Originating document filename or source.")
    page_number: Optional[int] = Field(default=None, ge=1, description="Page number where the entity was extracted.")
    section: Optional[str] = Field(default=None, description="Guideline section heading.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction or fact confidence score.")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary additional clinical properties.")


class MedicalRelationship(BaseModel):
    """Representation of a directed edge between two entities in the graph."""

    source_id: str = Field(..., description="Identifier of the source entity node.")
    type: str = Field(..., description="Relationship type (e.g., HAS_SYMPTOM, TREATED_BY).")
    target_id: str = Field(..., description="Identifier of the target entity node.")
    source_name: Optional[str] = Field(default=None, description="Optional name of the source entity.")
    target_name: Optional[str] = Field(default=None, description="Optional name of the target entity.")
    description: Optional[str] = Field(default=None, description="Context or explanation of the relationship.")
    source_document: Optional[str] = Field(default=None, description="Originating document filename or source.")
    page_number: Optional[int] = Field(default=None, ge=1, description="Page number of the guideline source.")
    section: Optional[str] = Field(default=None, description="Section heading from the source document.")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Fact confidence score (0.0 to 1.0).")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary additional relationship attributes.")


# --------------------------------------------------------------------------- #
# Request and Response Schemas
# --------------------------------------------------------------------------- #


class EntityCreateRequest(BaseModel):
    """Payload for creating or updating a graph entity."""

    id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    type: str = Field(..., description="Entity category (e.g. Disease, Drug, Symptom)")
    description: Optional[str] = None
    source_document: Optional[str] = None
    page_number: Optional[int] = None
    section: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    properties: Dict[str, Any] = Field(default_factory=dict)


class RelationshipCreateRequest(BaseModel):
    """Payload for creating a relationship between two entities."""

    source_id: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1)
    target_id: str = Field(..., min_length=1)
    description: Optional[str] = None
    source_document: Optional[str] = None
    page_number: Optional[int] = None
    section: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    properties: Dict[str, Any] = Field(default_factory=dict)


class EntityDetailResponse(BaseModel):
    """Detailed response for an entity including its incoming and outgoing relationships."""

    entity: MedicalEntity
    outgoing_relationships: List[MedicalRelationship] = Field(default_factory=list)
    incoming_relationships: List[MedicalRelationship] = Field(default_factory=list)


class RelatedEntity(BaseModel):
    """A neighbor node reached via a relationship."""

    entity: MedicalEntity
    relationship_type: str
    direction: str = Field(description="'outgoing' (source -> target) or 'incoming' (target <- source)")
    confidence: float = 1.0


class RelatedEntitiesResponse(BaseModel):
    """Response containing all neighbor entities for a given node."""

    entity_id: str
    total_related: int
    related: List[RelatedEntity]


class KnowledgeGraphStats(BaseModel):
    """Summary statistics for the Neo4j Knowledge Graph."""

    total_entities: int
    total_relationships: int
    num_entities: Optional[int] = None
    num_relations: Optional[int] = None
    entity_counts_by_type: Dict[str, int] = Field(default_factory=dict)
    relationship_counts_by_type: Dict[str, int] = Field(default_factory=dict)
    entity_types: Dict[str, int] = Field(default_factory=dict)
    relation_types: Dict[str, int] = Field(default_factory=dict)
    status: str = "healthy"


class DiseaseSymptomMatch(BaseModel):
    """Matched disease result based on reported symptoms."""

    disease: MedicalEntity
    matched_symptoms: List[str]
    match_count: int
    total_disease_symptoms: int
    confidence: float


class PatientGraphContext(BaseModel):
    """Structured graph context for a patient to be consumed by Stage 4 Reasoning."""

    patient_id: str
    entities: List[MedicalEntity]
    relationships: List[MedicalRelationship]
    derived_associations: List[Dict[str, Any]] = Field(default_factory=list)
