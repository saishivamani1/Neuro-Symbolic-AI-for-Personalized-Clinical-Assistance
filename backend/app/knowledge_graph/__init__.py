"""
app/knowledge_graph/__init__.py

Medical Knowledge Graph package backed by Neo4j.

Exports:
- Neo4jConnectionManager, get_neo4j_manager
- KnowledgeGraphService, get_kg_service
- MedicalEntity, MedicalRelationship, EntityType, RelationshipType
- EntityCreateRequest, RelationshipCreateRequest, EntityDetailResponse, RelatedEntitiesResponse
- KnowledgeGraphStats, DiseaseSymptomMatch, PatientGraphContext
- seed_sample_knowledge_graph
"""

from app.knowledge_graph.builder import seed_sample_knowledge_graph
from app.knowledge_graph.connection import Neo4jConnectionManager, get_neo4j_manager
from app.knowledge_graph.models import (
    DiseaseSymptomMatch,
    EntityCreateRequest,
    EntityDetailResponse,
    EntityType,
    KnowledgeGraphStats,
    MedicalEntity,
    MedicalRelationship,
    PatientGraphContext,
    RelatedEntitiesResponse,
    RelatedEntity,
    RelationshipCreateRequest,
    RelationshipType,
)
from app.knowledge_graph.service import KnowledgeGraphService, get_kg_service

__all__ = [
    "Neo4jConnectionManager",
    "get_neo4j_manager",
    "KnowledgeGraphService",
    "get_kg_service",
    "MedicalEntity",
    "MedicalRelationship",
    "EntityType",
    "RelationshipType",
    "EntityCreateRequest",
    "RelationshipCreateRequest",
    "EntityDetailResponse",
    "RelatedEntitiesResponse",
    "RelatedEntity",
    "KnowledgeGraphStats",
    "DiseaseSymptomMatch",
    "PatientGraphContext",
    "seed_sample_knowledge_graph",
]
