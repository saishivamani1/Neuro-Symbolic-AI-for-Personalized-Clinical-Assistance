"""
app/knowledge_graph/service.py

High-level domain service for interacting with the Neo4j Medical Knowledge Graph.

Provides clean CRUD operations, structured clinical traversals, and patient context
assembly for downstream symbolic reasoning.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

from app.core.exceptions import KnowledgeGraphError, MalformedGraphDataError
from app.core.logging import get_logger
from app.knowledge_graph.connection import Neo4jConnectionManager, get_neo4j_manager
from app.knowledge_graph.models import (
    DiseaseSymptomMatch,
    EntityCreateRequest,
    EntityDetailResponse,
    KnowledgeGraphStats,
    MedicalEntity,
    MedicalRelationship,
    PatientGraphContext,
    RelatedEntitiesResponse,
    RelatedEntity,
    RelationshipCreateRequest,
)
from app.knowledge_graph.queries import (
    BATCH_UPSERT_ENTITIES_QUERY,
    CREATE_ENTITY_NAME_INDEX,
    CREATE_ENTITY_TYPE_INDEX,
    CREATE_UNIQUE_ENTITY_ID_CONSTRAINT,
    FIND_CONTRAINDICATIONS_FOR_DRUG_QUERY,
    FIND_DISEASES_FROM_SYMPTOMS_QUERY,
    FIND_RELATED_ENTITIES_QUERY,
    FIND_RISK_FACTORS_FOR_DISEASE_QUERY,
    FIND_TREATMENTS_FOR_DISEASE_QUERY,
    GET_EDGES_FOR_DISEASES_QUERY,
    GET_ENTITY_BY_ID_QUERY,
    GET_GRAPH_STATISTICS_QUERY,
    GET_INCOMING_RELATIONSHIPS_QUERY,
    GET_OUTGOING_RELATIONSHIPS_QUERY,
    GET_PATIENT_GRAPH_CONTEXT_QUERY,
    GET_RELATIONSHIP_COUNTS_QUERY,
    GET_SAMPLE_GRAPH_QUERY,
    GET_TOP_DISEASES_QUERY,
    LIST_ENTITIES_QUERY,
    UPSERT_ENTITY_QUERY,
    build_batch_upsert_relationships_query,
    build_upsert_relationship_query,
)

logger = get_logger(__name__)


class KnowledgeGraphService:
    """Service layer encapsulating all Medical Knowledge Graph operations."""

    def __init__(
        self, connection_manager: Optional[Neo4jConnectionManager] = None
    ) -> None:
        self.manager = connection_manager or get_neo4j_manager()

    def initialize_constraints(self) -> None:
        """Create uniqueness constraints and indexes idempotently."""
        logger.info("Initializing Neo4j schema constraints and indexes...")
        try:
            self.manager.execute_query(CREATE_UNIQUE_ENTITY_ID_CONSTRAINT)
            self.manager.execute_query(CREATE_ENTITY_NAME_INDEX)
            self.manager.execute_query(CREATE_ENTITY_TYPE_INDEX)
            logger.info("Neo4j constraints and indexes successfully verified/created.")
        except Exception as exc:
            logger.error("Failed to initialize Neo4j constraints: %s", str(exc))
            raise KnowledgeGraphError(
                detail=f"Schema initialization error: {exc}",
                context={"error": str(exc)},
            ) from exc

    def _node_to_entity(self, node: Any) -> MedicalEntity:
        """Convert a raw Neo4j node dictionary to a MedicalEntity model."""
        if not node:
            return MedicalEntity(id="unknown", name="Unknown", type="MedicalConcept")

        if isinstance(node, dict):
            props = node.get("e") or node.get("other") or node.get("t") or node.get("d") or node.get("rf") or node
        else:
            props = getattr(node, "_properties", {}) or {}

        if not isinstance(props, dict):
            props = getattr(props, "_properties", {}) or {}

        properties_json = props.get("properties")
        properties_dict: Dict[str, Any] = {}
        if properties_json:
            try:
                properties_dict = (
                    json.loads(properties_json)
                    if isinstance(properties_json, str)
                    else properties_json
                )
            except Exception:
                properties_dict = {}

        return MedicalEntity(
            id=str(props.get("id", "")),
            name=str(props.get("name", "")),
            type=str(props.get("type", "MedicalConcept")),
            description=props.get("description"),
            source_document=props.get("source_document"),
            page_number=props.get("page_number"),
            section=props.get("section"),
            confidence=float(props.get("confidence", 1.0)),
            properties=properties_dict,
        )

    def create_entity(
        self, entity_data: Union[MedicalEntity, EntityCreateRequest]
    ) -> MedicalEntity:
        """Upsert a medical entity node into the graph."""
        params = {
            "id": entity_data.id.strip(),
            "name": entity_data.name.strip(),
            "type": entity_data.type.strip(),
            "description": entity_data.description,
            "source_document": entity_data.source_document,
            "page_number": entity_data.page_number,
            "section": entity_data.section,
            "confidence": entity_data.confidence,
            "properties_json": json.dumps(entity_data.properties or {}),
        }

        results = self.manager.execute_query(UPSERT_ENTITY_QUERY, params)
        if not results:
            raise KnowledgeGraphError(
                detail=f"Failed to upsert entity with ID: '{entity_data.id}'",
                context={"entity_id": entity_data.id},
            )

        logger.info("Upserted entity node: %s (%s)", entity_data.name, entity_data.type)
        return self._node_to_entity(results[0])

    def batch_create_entities(
        self, entities: List[Union[MedicalEntity, EntityCreateRequest]], batch_size: int = 500
    ) -> int:
        """Upsert a list of medical entity nodes in high-performance batches using UNWIND."""
        if not entities:
            return 0

        total_upserted = 0
        for i in range(0, len(entities), batch_size):
            batch = entities[i : i + batch_size]
            batch_params = [
                {
                    "id": e.id.strip(),
                    "name": e.name.strip(),
                    "type": e.type.strip(),
                    "description": e.description,
                    "source_document": e.source_document,
                    "page_number": e.page_number,
                    "section": e.section,
                    "confidence": e.confidence,
                    "properties_json": json.dumps(e.properties or {}),
                }
                for e in batch
            ]
            results = self.manager.execute_query(BATCH_UPSERT_ENTITIES_QUERY, {"batch": batch_params})
            if results and "upserted_count" in results[0]:
                total_upserted += int(results[0]["upserted_count"])
            else:
                total_upserted += len(batch)
            logger.info("Batch upserted %d entities (progress: %d/%d)", len(batch), min(i + batch_size, len(entities)), len(entities))

        return total_upserted

    def batch_create_relationships(
        self, relationships: List[Union[MedicalRelationship, RelationshipCreateRequest]], batch_size: int = 500
    ) -> int:
        """Upsert relationships grouped by rel_type in high-performance batches using UNWIND."""
        if not relationships:
            return 0

        from collections import defaultdict
        by_type = defaultdict(list)
        for r in relationships:
            rel_type = r.type.strip().upper()
            by_type[rel_type].append(r)

        total_created = 0
        for rel_type, rel_list in by_type.items():
            query = build_batch_upsert_relationships_query(rel_type)
            for i in range(0, len(rel_list), batch_size):
                batch = rel_list[i : i + batch_size]
                batch_params = [
                    {
                        "source_id": r.source_id.strip(),
                        "target_id": r.target_id.strip(),
                        "description": r.description,
                        "source_document": r.source_document,
                        "page_number": r.page_number,
                        "section": r.section,
                        "confidence": r.confidence,
                        "properties_json": json.dumps(r.properties or {}),
                    }
                    for r in batch
                ]
                results = self.manager.execute_query(query, {"batch": batch_params})
                if results and "upserted_count" in results[0]:
                    total_created += int(results[0]["upserted_count"])
                else:
                    total_created += len(batch)
                logger.info("Batch created %d %s relationships (progress: %d/%d)", len(batch), rel_type, min(i + batch_size, len(rel_list)), len(rel_list))

        return total_created

    def get_entity(self, entity_id: str) -> Optional[MedicalEntity]:
        """Fetch an entity node by its unique ID."""
        results = self.manager.execute_query(
            GET_ENTITY_BY_ID_QUERY, {"id": entity_id.strip()}
        )
        if not results:
            return None
        return self._node_to_entity(results[0])

    def get_entity_detail(self, entity_id: str) -> Optional[EntityDetailResponse]:
        """Fetch entity along with all its incoming and outgoing relationships."""
        entity = self.get_entity(entity_id)
        if not entity:
            return None

        outgoing_rows = self.manager.execute_query(
            GET_OUTGOING_RELATIONSHIPS_QUERY, {"id": entity_id.strip()}
        )
        incoming_rows = self.manager.execute_query(
            GET_INCOMING_RELATIONSHIPS_QUERY, {"id": entity_id.strip()}
        )

        outgoing = [self._row_to_relationship(row) for row in outgoing_rows]
        incoming = [self._row_to_relationship(row) for row in incoming_rows]

        return EntityDetailResponse(
            entity=entity,
            outgoing_relationships=outgoing,
            incoming_relationships=incoming,
        )

    def list_entities(
        self, entity_type: Optional[str] = None, search: Optional[str] = None, limit: int = 100
    ) -> List[MedicalEntity]:
        """List entities with optional type filter and text search."""
        results = self.manager.execute_query(
            LIST_ENTITIES_QUERY,
            {
                "entity_type": entity_type.strip() if entity_type else None,
                "search": search.strip() if search else None,
                "limit": limit,
            },
        )
        return [self._node_to_entity(row) for row in results]

    def _row_to_relationship(self, row: Dict[str, Any]) -> MedicalRelationship:
        """Convert a Cypher result row to a MedicalRelationship model."""
        props_json = row.get("properties_json")
        props_dict = {}
        if props_json:
            try:
                props_dict = (
                    json.loads(props_json)
                    if isinstance(props_json, str)
                    else props_json
                )
            except Exception:
                props_dict = {}

        return MedicalRelationship(
            source_id=row.get("source_id", ""),
            type=row.get("rel_type", ""),
            target_id=row.get("target_id", ""),
            source_name=row.get("source_name"),
            target_name=row.get("target_name"),
            description=row.get("description"),
            source_document=row.get("source_document"),
            page_number=row.get("page_number"),
            section=row.get("section"),
            confidence=float(row.get("confidence", 1.0)),
            properties=props_dict,
        )

    def create_relationship(
        self, rel_data: Union[MedicalRelationship, RelationshipCreateRequest]
    ) -> MedicalRelationship:
        """Create or update a directed relationship between two entities."""
        # Ensure source and target entities exist
        source = self.get_entity(rel_data.source_id)
        if not source:
            raise MalformedGraphDataError(
                detail=f"Source entity '{rel_data.source_id}' does not exist in graph.",
                context={"source_id": rel_data.source_id},
            )

        target = self.get_entity(rel_data.target_id)
        if not target:
            raise MalformedGraphDataError(
                detail=f"Target entity '{rel_data.target_id}' does not exist in graph.",
                context={"target_id": rel_data.target_id},
            )

        query = build_upsert_relationship_query(rel_data.type)
        params = {
            "source_id": rel_data.source_id.strip(),
            "target_id": rel_data.target_id.strip(),
            "description": rel_data.description,
            "source_document": rel_data.source_document,
            "page_number": rel_data.page_number,
            "section": rel_data.section,
            "confidence": rel_data.confidence,
            "properties_json": json.dumps(rel_data.properties or {}),
        }

        results = self.manager.execute_query(query, params)
        if not results:
            raise KnowledgeGraphError(
                detail=f"Failed to create relationship {rel_data.source_id} -[{rel_data.type}]-> {rel_data.target_id}",
                context={"relationship": rel_data.model_dump()},
            )

        logger.info(
            "Created relationship: %s -[%s]-> %s",
            source.name,
            rel_data.type,
            target.name,
        )

        return MedicalRelationship(
            source_id=source.id,
            source_name=source.name,
            type=rel_data.type,
            target_id=target.id,
            target_name=target.name,
            description=rel_data.description,
            source_document=rel_data.source_document,
            page_number=rel_data.page_number,
            section=rel_data.section,
            confidence=rel_data.confidence,
            properties=rel_data.properties or {},
        )

    def find_related_entities(
        self, entity_id: str, limit: int = 50
    ) -> RelatedEntitiesResponse:
        """Find immediate neighbor nodes connected via any relationship."""
        results = self.manager.execute_query(
            FIND_RELATED_ENTITIES_QUERY,
            {"id": entity_id.strip(), "limit": limit},
        )
        related_list: List[RelatedEntity] = []
        for row in results:
            node_data = row.get("t", {})
            entity = self._node_to_entity({"e": node_data})
            related_list.append(
                RelatedEntity(
                    entity=entity,
                    relationship_type=row.get("rel_type", ""),
                    direction=row.get("direction", "outgoing"),
                    confidence=float(row.get("confidence", 1.0)),
                )
            )

        return RelatedEntitiesResponse(
            entity_id=entity_id,
            total_related=len(related_list),
            related=related_list,
        )

    def find_diseases_from_symptoms(
        self, symptoms: List[str], limit: int = 10
    ) -> List[DiseaseSymptomMatch]:
        """Rank diseases according to matched symptoms."""
        normalized_symptoms = [s.strip().lower() for s in symptoms if s.strip()]
        if not normalized_symptoms:
            return []

        results = self.manager.execute_query(
            FIND_DISEASES_FROM_SYMPTOMS_QUERY,
            {"symptoms": normalized_symptoms, "limit": limit},
        )

        matches: List[DiseaseSymptomMatch] = []
        for row in results:
            disease_node = row.get("d", {})
            disease_entity = self._node_to_entity({"e": disease_node})
            matched = row.get("matched_symptoms", [])
            match_count = int(row.get("match_count", 0))
            total_symptoms = int(row.get("total_disease_symptoms", 1))
            confidence = float(row.get("confidence", 0.0))

            matches.append(
                DiseaseSymptomMatch(
                    disease=disease_entity,
                    matched_symptoms=matched,
                    match_count=match_count,
                    total_disease_symptoms=total_symptoms,
                    confidence=round(confidence, 4),
                )
            )

        return matches

    def find_treatments(self, disease_name_or_id: str) -> List[MedicalEntity]:
        """Find treatments or medications indicated for a disease."""
        results = self.manager.execute_query(
            FIND_TREATMENTS_FOR_DISEASE_QUERY,
            {"disease": disease_name_or_id.strip()},
        )
        return [self._node_to_entity({"e": row.get("t", {})}) for row in results]

    def find_risk_factors(self, disease_name_or_id: str) -> List[MedicalEntity]:
        """Find known risk factors for a disease."""
        results = self.manager.execute_query(
            FIND_RISK_FACTORS_FOR_DISEASE_QUERY,
            {"disease": disease_name_or_id.strip()},
        )
        return [self._node_to_entity({"e": row.get("rf", {})}) for row in results]

    def _extract_rel_properties(self, rel: Any) -> Dict[str, Any]:
        """Extract relationship properties from dicts, Neo4j 3-tuples, or Relationship objects."""
        if isinstance(rel, dict):
            return rel
        if isinstance(rel, (list, tuple)) and len(rel) >= 3 and isinstance(rel[2], dict):
            return rel[2]
        if hasattr(rel, "_properties") and isinstance(rel._properties, dict):
            return rel._properties
        return {}

    def find_contraindications(self, drug_name_or_id: str) -> List[Dict[str, Any]]:
        """Find conditions or drugs contraindicated with a specified drug."""
        results = self.manager.execute_query(
            FIND_CONTRAINDICATIONS_FOR_DRUG_QUERY,
            {"drug": drug_name_or_id.strip()},
        )
        contraindications: List[Dict[str, Any]] = []
        for row in results:
            other_entity = self._node_to_entity({"e": row.get("other", {})})
            rel_props = self._extract_rel_properties(row.get("r", {}))
            contraindications.append(
                {
                    "contraindicated_with": other_entity,
                    "description": rel_props.get("description"),
                    "source_document": rel_props.get("source_document"),
                    "page_number": rel_props.get("page_number"),
                    "confidence": float(rel_props.get("confidence", 1.0)),
                }
            )
        return contraindications

    def get_context_for_patient(self, patient_id: str) -> PatientGraphContext:
        """Retrieve 2-hop structured knowledge graph subgraph for a patient."""
        results = self.manager.execute_query(
            GET_PATIENT_GRAPH_CONTEXT_QUERY,
            {"patient_id": patient_id.strip()},
        )
        if not results or not results[0].get("p"):
            # If patient node does not exist yet, return minimal context
            return PatientGraphContext(
                patient_id=patient_id,
                entities=[],
                relationships=[],
                derived_associations=[],
            )

        row = results[0]
        entities_dict: Dict[str, MedicalEntity] = {}

        # Add patient
        p_entity = self._node_to_entity({"e": row["p"]})
        entities_dict[p_entity.id] = p_entity

        # Direct entities
        for n in row.get("direct_entities", []):
            if n:
                ent = self._node_to_entity({"e": n})
                entities_dict[ent.id] = ent

        # 2nd hop entities
        for n in row.get("second_hop_entities", []):
            if n:
                ent = self._node_to_entity({"e": n})
                entities_dict[ent.id] = ent

        # Construct relationships
        relationships: List[MedicalRelationship] = []
        for r in row.get("direct_relationships", []) + row.get("second_hop_relationships", []):
            if r and r.get("source_id") and r.get("target_id"):
                relationships.append(
                    MedicalRelationship(
                        source_id=r["source_id"],
                        source_name=entities_dict.get(r["source_id"], MedicalEntity(id=r["source_id"], name=r["source_id"], type="Unknown")).name,
                        type=r.get("type", "ASSOCIATED_WITH"),
                        target_id=r["target_id"],
                        target_name=entities_dict.get(r["target_id"], MedicalEntity(id=r["target_id"], name=r["target_id"], type="Unknown")).name,
                        confidence=1.0,
                    )
                )

        return PatientGraphContext(
            patient_id=patient_id,
            entities=list(entities_dict.values()),
            relationships=relationships,
        )

    def get_sample_graph(
        self,
        disease_limit: int = 6,
        total_limit: int = 300,
        max_leaves_per_hub: int = 8,
    ) -> Dict[str, Any]:
        """Return a balanced, performant sample subgraph of diseases with connected entities.

        Guarantees that each top disease hub gets up to max_leaves_per_hub connections,
        preventing single diseases with many relationships from crowding out other diseases,
        and keeping the canvas rendering fluid and lag-free.
        """
        disease_rows = self.manager.execute_query(
            GET_TOP_DISEASES_QUERY,
            {"disease_limit": disease_limit},
        )
        if not disease_rows:
            return {"nodes": [], "edges": [], "total_nodes": 0, "total_edges": 0}

        disease_ids = [row["id"] for row in disease_rows if row.get("id")]
        edge_rows = self.manager.execute_query(
            GET_EDGES_FOR_DISEASES_QUERY,
            {"disease_ids": disease_ids},
        )

        nodes_dict: Dict[str, Dict[str, Any]] = {}
        for row in disease_rows:
            d_id = row.get("id")
            if d_id:
                src_type = (row.get("type") or "disease").lower()
                nodes_dict[d_id] = {
                    "id": d_id,
                    "label": row.get("name") or d_id,
                    "type": src_type,
                    "description": row.get("description"),
                    "entity_type": src_type,
                }

        leaves_per_hub: Dict[str, int] = {d_id: 0 for d_id in nodes_dict}
        edges: List[Dict[str, Any]] = []

        for row in edge_rows:
            src_id = row.get("source_id")
            tgt_id = row.get("target_id")
            rel_type = row.get("rel_type")
            if not src_id or not tgt_id or not rel_type:
                continue

            if leaves_per_hub.get(src_id, 0) >= max_leaves_per_hub:
                continue

            tgt_type = (row.get("target_type") or "Unknown").lower()
            if tgt_id not in nodes_dict:
                nodes_dict[tgt_id] = {
                    "id": tgt_id,
                    "label": row.get("target_name") or tgt_id,
                    "type": tgt_type,
                    "description": None,
                    "entity_type": tgt_type,
                }

            edges.append({
                "id": f"{src_id}__{rel_type}__{tgt_id}",
                "source": src_id,
                "target": tgt_id,
                "rel_type": rel_type,
            })
            leaves_per_hub[src_id] = leaves_per_hub.get(src_id, 0) + 1

        return {
            "nodes": list(nodes_dict.values()),
            "edges": edges,
            "total_nodes": len(nodes_dict),
            "total_edges": len(edges),
        }

    def get_stats(self) -> KnowledgeGraphStats:
        """Retrieve total entity, relationship, and label distribution stats."""
        results = self.manager.execute_query(GET_GRAPH_STATISTICS_QUERY)
        if not results:
            return KnowledgeGraphStats(
                total_entities=0,
                total_relationships=0,
                num_entities=0,
                num_relations=0,
                status="healthy",
            )

        row = results[0]
        total_entities = int(row.get("total_entities", 0))
        total_rels = int(row.get("total_relationships", 0))
        types_list = [t for t in row.get("types", []) if t]
        entity_counts = dict(Counter(types_list))

        # Query relationship breakdown by type
        rel_type_counts: Dict[str, int] = {}
        try:
            rel_results = self.manager.execute_query(GET_RELATIONSHIP_COUNTS_QUERY)
            for r_row in rel_results:
                r_type = r_row.get("rel_type")
                count = int(r_row.get("count", 0))
                if r_type:
                    rel_type_counts[r_type] = count
        except Exception as exc:
            logger.warning("Could not fetch relationship breakdown: %s", exc)

        return KnowledgeGraphStats(
            total_entities=total_entities,
            total_relationships=total_rels,
            num_entities=total_entities,
            num_relations=total_rels,
            entity_counts_by_type=entity_counts,
            relationship_counts_by_type=rel_type_counts,
            entity_types=entity_counts,
            relation_types=rel_type_counts,
            status="healthy",
        )


_cached_kg_service: Optional[KnowledgeGraphService] = None


def get_kg_service() -> KnowledgeGraphService:
    """Singleton getter for KnowledgeGraphService."""
    global _cached_kg_service
    if _cached_kg_service is None:
        _cached_kg_service = KnowledgeGraphService()
    return _cached_kg_service
