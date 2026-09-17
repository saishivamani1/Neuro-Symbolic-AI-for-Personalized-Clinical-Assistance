"""
app/api/routes/knowledge_graph.py

REST API endpoints for querying and managing the Medical Knowledge Graph.

Endpoints:
- GET  /api/v1/knowledge-graph/entities: List entities with optional type filter
- GET  /api/v1/knowledge-graph/entities/{entity_id}: Entity details with relationships
- POST /api/v1/knowledge-graph/entities: Create or update entity node
- POST /api/v1/knowledge-graph/relationships: Create relationship between entities
- GET  /api/v1/knowledge-graph/related/{entity_id}: Get neighbor entities
- GET  /api/v1/knowledge-graph/stats: Graph statistics and node/edge counts
- GET  /api/v1/knowledge-graph/diseases: Disease diagnosis matching by symptoms
- GET  /api/v1/knowledge-graph/{disease}/treatments: Treatments indicated for a disease
- GET  /api/v1/knowledge-graph/{disease}/risk-factors: Known risk factors for a disease
- GET  /api/v1/knowledge-graph/{drug}/contraindications: Contraindications for a drug
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Path, Query, status

from app.core.logging import get_logger
from app.knowledge_graph.models import (
    DiseaseSymptomMatch,
    EntityCreateRequest,
    EntityDetailResponse,
    KnowledgeGraphStats,
    MedicalEntity,
    MedicalRelationship,
    RelatedEntitiesResponse,
    RelationshipCreateRequest,
)
from app.knowledge_graph.service import get_kg_service
from app.schemas.response import APIResponse

router = APIRouter()
logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# General Graph Inspection & Stats
# --------------------------------------------------------------------------- #


@router.get(
    "/knowledge-graph/stats",
    response_model=APIResponse[KnowledgeGraphStats],
    tags=["Knowledge Graph"],
    summary="Get summary statistics of the Medical Knowledge Graph",
)
async def get_graph_statistics() -> APIResponse[KnowledgeGraphStats]:
    """Return total node and edge counts and breakdown by entity label."""
    kg_service = get_kg_service()
    stats = kg_service.get_stats()
    return APIResponse(success=True, data=stats)


@router.get(
    "/knowledge-graph/entities",
    response_model=APIResponse[List[MedicalEntity]],
    tags=["Knowledge Graph"],
    summary="List medical entities in the knowledge graph",
)
async def list_entities(
    entity_type: Optional[str] = Query(
        default=None,
        description="Filter by entity type (e.g., Disease, Drug, Symptom, MedicalTest)",
    ),
    search: Optional[str] = Query(
        default=None,
        description="Search entities by name or identifier",
    ),
    limit: int = Query(default=100, ge=1, le=500, description="Maximum number of entities to return"),
) -> APIResponse[List[MedicalEntity]]:
    """Retrieve entities matching an optional category filter or search keyword."""
    kg_service = get_kg_service()
    entities = kg_service.list_entities(entity_type=entity_type, search=search, limit=limit)
    return APIResponse(success=True, data=entities)


@router.get(
    "/knowledge-graph/entities/{entity_id}",
    response_model=APIResponse[EntityDetailResponse],
    tags=["Knowledge Graph"],
    summary="Get entity details and its incoming/outgoing relationships",
)
async def get_entity_detail(
    entity_id: str = Path(..., description="Unique entity identifier"),
) -> APIResponse[EntityDetailResponse]:
    """Fetch complete entity profile along with its topological connections."""
    kg_service = get_kg_service()
    detail = kg_service.get_entity_detail(entity_id=entity_id)
    if not detail:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Medical entity with ID '{entity_id}' not found.",
        )
    return APIResponse(success=True, data=detail)


@router.post(
    "/knowledge-graph/entities",
    response_model=APIResponse[MedicalEntity],
    status_code=status.HTTP_201_CREATED,
    tags=["Knowledge Graph"],
    summary="Create or update a medical entity node",
)
async def create_entity(
    request: EntityCreateRequest,
) -> APIResponse[MedicalEntity]:
    """Create a new clinical entity node or update an existing one."""
    kg_service = get_kg_service()
    entity = kg_service.create_entity(request)
    return APIResponse(success=True, data=entity)


@router.post(
    "/knowledge-graph/relationships",
    response_model=APIResponse[MedicalRelationship],
    status_code=status.HTTP_201_CREATED,
    tags=["Knowledge Graph"],
    summary="Create a directed relationship between two entities",
)
async def create_relationship(
    request: RelationshipCreateRequest,
) -> APIResponse[MedicalRelationship]:
    """Create or update a directed relationship edge between two nodes."""
    kg_service = get_kg_service()
    relationship = kg_service.create_relationship(request)
    return APIResponse(success=True, data=relationship)


@router.get(
    "/knowledge-graph/related/{entity_id}",
    response_model=APIResponse[RelatedEntitiesResponse],
    tags=["Knowledge Graph"],
    summary="Get neighbor entities connected to a node",
)
async def get_related_entities(
    entity_id: str = Path(..., description="Source entity ID"),
    limit: int = Query(default=50, ge=1, le=200, description="Max neighbor nodes to return"),
) -> APIResponse[RelatedEntitiesResponse]:
    """Retrieve all neighboring nodes directly connected via any relationship."""
    kg_service = get_kg_service()
    related = kg_service.find_related_entities(entity_id=entity_id, limit=limit)
    return APIResponse(success=True, data=related)


# --------------------------------------------------------------------------- #
# Graph Visualization Endpoint
# --------------------------------------------------------------------------- #


@router.get(
    "/knowledge-graph/graph",
    response_model=APIResponse[Dict[str, Any]],
    tags=["Knowledge Graph"],
    summary="Fetch a ReactFlow-ready visual subgraph from the database",
)
async def get_visual_graph(
    disease_limit: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Max number of disease nodes to include as central hubs",
    ),
    total_limit: int = Query(
        default=300,
        ge=10,
        le=1000,
        description="Max total node+edge rows to return for canvas rendering",
    ),
) -> APIResponse[Dict[str, Any]]:
    """Return a structured subgraph of diseases and their connected entities (symptoms, drugs, risk factors)
    as flat node/edge lists suitable for ReactFlow canvas rendering."""
    kg_service = get_kg_service()
    graph = kg_service.get_sample_graph(
        disease_limit=disease_limit,
        total_limit=total_limit,
    )
    return APIResponse(success=True, data=graph)


# --------------------------------------------------------------------------- #
# Clinical Query Endpoints
# --------------------------------------------------------------------------- #


@router.get(
    "/knowledge-graph/diseases",
    response_model=APIResponse[List[DiseaseSymptomMatch]],
    tags=["Knowledge Graph Queries"],
    summary="Find diseases matching a list of clinical symptoms",
)
async def find_diseases_by_symptoms(
    symptoms: str = Query(
        ...,
        min_length=1,
        description="Comma-separated list of symptoms (e.g. 'fatigue,increased thirst')",
    ),
    limit: int = Query(default=10, ge=1, le=50, description="Max disease matches"),
) -> APIResponse[List[DiseaseSymptomMatch]]:
    """Match diseases based on reported patient symptoms and rank by overlap score."""
    symptom_list = [s.strip() for s in symptoms.split(",") if s.strip()]
    if not symptom_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please provide at least one non-empty symptom in the query parameter.",
        )

    kg_service = get_kg_service()
    matches = kg_service.find_diseases_from_symptoms(symptoms=symptom_list, limit=limit)
    return APIResponse(success=True, data=matches)


@router.get(
    "/knowledge-graph/{disease}/treatments",
    response_model=APIResponse[List[MedicalEntity]],
    tags=["Knowledge Graph Queries"],
    summary="Find treatments and medications indicated for a disease",
)
async def find_treatments_for_disease(
    disease: str = Path(..., description="Disease name or ID"),
) -> APIResponse[List[MedicalEntity]]:
    """Retrieve recommended drugs or therapies for the specified disease."""
    kg_service = get_kg_service()
    treatments = kg_service.find_treatments(disease_name_or_id=disease)
    return APIResponse(success=True, data=treatments)


@router.get(
    "/knowledge-graph/{disease}/risk-factors",
    response_model=APIResponse[List[MedicalEntity]],
    tags=["Knowledge Graph Queries"],
    summary="Find risk factors for a disease",
)
async def find_risk_factors_for_disease(
    disease: str = Path(..., description="Disease name or ID"),
) -> APIResponse[List[MedicalEntity]]:
    """Retrieve known risk factors associated with the specified disease."""
    kg_service = get_kg_service()
    risk_factors = kg_service.find_risk_factors(disease_name_or_id=disease)
    return APIResponse(success=True, data=risk_factors)


@router.get(
    "/knowledge-graph/{drug}/contraindications",
    response_model=APIResponse[List[Dict[str, Any]]],
    tags=["Knowledge Graph Queries"],
    summary="Find contraindications and adverse interactions for a drug",
)
async def find_contraindications_for_drug(
    drug: str = Path(..., description="Drug name or ID"),
) -> APIResponse[List[Dict[str, Any]]]:
    """Retrieve conditions, diseases, or concurrent medications contraindicated with the given drug."""
    kg_service = get_kg_service()
    contraindications = kg_service.find_contraindications(drug_name_or_id=drug)
    return APIResponse(success=True, data=contraindications)
