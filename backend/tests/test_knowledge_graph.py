"""
tests/test_knowledge_graph.py

Comprehensive test suite for Stage 3: Medical Knowledge Graph.

Covers:
1. Neo4j Connection Manager & session handling (app.knowledge_graph.connection)
2. Medical Entity & Relationship Pydantic models (app.knowledge_graph.models)
3. Cypher query generation & injection defense (app.knowledge_graph.queries)
4. KnowledgeGraphService CRUD, traversals, and medical query logic (app.knowledge_graph.service)
5. Synthetic clinical dataset seeder (app.knowledge_graph.builder)
6. FastAPI REST API endpoints (/api/v1/knowledge-graph/*)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.exceptions import KnowledgeGraphError, MalformedGraphDataError
from app.knowledge_graph.builder import SAMPLE_ENTITIES, SAMPLE_RELATIONSHIPS, seed_sample_knowledge_graph
from app.knowledge_graph.connection import Neo4jConnectionManager
from app.knowledge_graph.models import (
    DiseaseSymptomMatch,
    EntityCreateRequest,
    EntityType,
    MedicalEntity,
    MedicalRelationship,
    PatientGraphContext,
    RelationshipCreateRequest,
    RelationshipType,
)
from app.knowledge_graph.queries import build_upsert_relationship_query
from app.knowledge_graph.service import KnowledgeGraphService
from app.main import app


# --------------------------------------------------------------------------- #
# Mock In-Memory Neo4j Manager for Unit Testing
# --------------------------------------------------------------------------- #


class MockNeo4jManager(Neo4jConnectionManager):
    """In-memory simulation of Neo4j graph storage for isolated unit tests."""

    def __init__(self) -> None:
        super().__init__(uri="bolt://mock:7687", username="mock", password="mock")
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.relationships: List[Dict[str, Any]] = []

    def verify_connectivity(self) -> bool:
        return True

    def execute_query(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        params = parameters or {}
        normalized = " ".join(query.split())

        # Schema constraints / indexes
        if "CREATE CONSTRAINT" in query or "CREATE INDEX" in query:
            return []

        # Upsert Entity
        if "MERGE (e:MedicalEntity {id: $id})" in query:
            entity_id = params["id"]
            props = {
                "id": entity_id,
                "name": params["name"],
                "type": params["type"],
                "description": params.get("description"),
                "source_document": params.get("source_document"),
                "page_number": params.get("page_number"),
                "section": params.get("section"),
                "confidence": params.get("confidence", 1.0),
                "properties": params.get("properties_json", "{}"),
            }
            self.entities[entity_id] = props
            return [{"e": props}]

        # Get Entity by ID
        if "MATCH (e:MedicalEntity {id: $id})" in query and "RETURN e" in query:
            entity_id = params["id"]
            if entity_id in self.entities:
                return [{"e": self.entities[entity_id]}]
            return []

        # List Entities
        if "MATCH (e:MedicalEntity)" in query and "ORDER BY e.name ASC" in query:
            target_type = params.get("entity_type")
            results = []
            for ent in self.entities.values():
                if not target_type or ent["type"].lower() == target_type.lower():
                    results.append({"e": ent})
            return sorted(results, key=lambda x: x["e"]["name"])[: params.get("limit", 100)]

        # Upsert Relationship
        if "MERGE (s)-[r:" in query:
            source_id = params["source_id"]
            target_id = params["target_id"]
            # Extract relationship type from query
            import re
            rel_match = re.search(r"MERGE \(s\)-\[r:([A-Z0-9_]+)\]->\(t\)", query)
            rel_type = rel_match.group(1) if rel_match else "ASSOCIATED_WITH"

            rel = {
                "source_id": source_id,
                "source_name": self.entities.get(source_id, {}).get("name", source_id),
                "rel_type": rel_type,
                "target_id": target_id,
                "target_name": self.entities.get(target_id, {}).get("name", target_id),
                "description": params.get("description"),
                "source_document": params.get("source_document"),
                "page_number": params.get("page_number"),
                "section": params.get("section"),
                "confidence": params.get("confidence", 1.0),
                "properties_json": params.get("properties_json", "{}"),
            }
            # Remove duplicate if exists
            self.relationships = [
                r for r in self.relationships
                if not (r["source_id"] == source_id and r["target_id"] == target_id and r["rel_type"] == rel_type)
            ]
            self.relationships.append(rel)
            return [{"s": self.entities.get(source_id), "r": rel, "t": self.entities.get(target_id), "rel_type": rel_type}]

        # Outgoing relationships
        if "MATCH (s:MedicalEntity {id: $id})-[r]->(t:MedicalEntity)" in query:
            entity_id = params["id"]
            return [r for r in self.relationships if r["source_id"] == entity_id]

        # Incoming relationships
        if "MATCH (s:MedicalEntity)-[r]->(t:MedicalEntity {id: $id})" in query:
            entity_id = params["id"]
            return [r for r in self.relationships if r["target_id"] == entity_id]

        # Related entities
        if "MATCH (s:MedicalEntity {id: $id})-[r]-(t:MedicalEntity)" in query:
            entity_id = params["id"]
            results = []
            for r in self.relationships:
                if r["source_id"] == entity_id:
                    results.append({
                        "t": self.entities.get(r["target_id"], {}),
                        "rel_type": r["rel_type"],
                        "direction": "outgoing",
                        "confidence": r["confidence"],
                    })
                elif r["target_id"] == entity_id:
                    results.append({
                        "t": self.entities.get(r["source_id"], {}),
                        "rel_type": r["rel_type"],
                        "direction": "incoming",
                        "confidence": r["confidence"],
                    })
            return results[: params.get("limit", 50)]

        # Disease from symptoms
        if "MATCH (d:MedicalEntity)-[:HAS_SYMPTOM]->(s:MedicalEntity)" in query:
            symptoms = [s.lower() for s in params.get("symptoms", [])]
            results = []
            # Find all diseases
            diseases = [e for e in self.entities.values() if e["type"].lower() in ["disease", "condition"]]
            for d in diseases:
                # Find symptoms for this disease
                disease_symptom_rels = [
                    r for r in self.relationships
                    if r["source_id"] == d["id"] and r["rel_type"] == "HAS_SYMPTOM"
                ]
                total_symptoms = max(1, len(disease_symptom_rels))
                matched = []
                for r in disease_symptom_rels:
                    s_node = self.entities.get(r["target_id"], {})
                    if s_node.get("name", "").lower() in symptoms or s_node.get("id", "").lower() in symptoms:
                        matched.append(s_node.get("name", ""))

                if matched:
                    results.append({
                        "d": d,
                        "matched_symptoms": matched,
                        "match_count": len(matched),
                        "total_disease_symptoms": total_symptoms,
                        "confidence": len(matched) / float(total_symptoms),
                    })
            return sorted(results, key=lambda x: (x["match_count"], x["confidence"]), reverse=True)

        # Treatments
        if "MATCH (d:MedicalEntity)-[r:TREATED_BY]->(t:MedicalEntity)" in query:
            target = params["disease"].lower()
            results = []
            for r in self.relationships:
                if r["rel_type"] == "TREATED_BY":
                    d = self.entities.get(r["source_id"], {})
                    if d.get("name", "").lower() == target or d.get("id", "").lower() == target:
                        results.append({"t": self.entities.get(r["target_id"], {}), "r": r, "d": d})
            return results

        # Risk Factors
        if "MATCH (rf:MedicalEntity)-[r:RISK_FACTOR_FOR]->(d:MedicalEntity)" in query:
            target = params["disease"].lower()
            results = []
            for r in self.relationships:
                if r["rel_type"] == "RISK_FACTOR_FOR":
                    d = self.entities.get(r["target_id"], {})
                    if d.get("name", "").lower() == target or d.get("id", "").lower() == target:
                        results.append({"rf": self.entities.get(r["source_id"], {}), "r": r, "d": d})
            return results

        # Contraindications
        if "MATCH (drug:MedicalEntity)-[r:CONTRAINDICATED_WITH]-(other:MedicalEntity)" in query:
            target = params["drug"].lower()
            results = []
            for r in self.relationships:
                if r["rel_type"] == "CONTRAINDICATED_WITH":
                    s = self.entities.get(r["source_id"], {})
                    t = self.entities.get(r["target_id"], {})
                    if s.get("name", "").lower() == target or s.get("id", "").lower() == target:
                        results.append({"other": t, "r": r, "drug": s})
                    elif t.get("name", "").lower() == target or t.get("id", "").lower() == target:
                        results.append({"other": s, "r": r, "drug": t})
            return results

        # Patient Graph Context
        if "MATCH (p:MedicalEntity {id: $patient_id})" in query:
            pid = params["patient_id"]
            p_node = self.entities.get(pid)
            if not p_node:
                return []
            direct_rels = [r for r in self.relationships if r["source_id"] == pid]
            direct_entities = [self.entities.get(r["target_id"]) for r in direct_rels if r["target_id"] in self.entities]
            return [{
                "p": p_node,
                "direct_entities": direct_entities,
                "direct_relationships": [
                    {"source_id": r["source_id"], "target_id": r["target_id"], "type": r["rel_type"], "properties": "{}"}
                    for r in direct_rels
                ],
                "second_hop_entities": [],
                "second_hop_relationships": [],
            }]

        # Stats
        if "MATCH (e:MedicalEntity)" in query and "count(e) AS total_entities" in query:
            types = [e["type"] for e in self.entities.values()]
            return [{
                "total_entities": len(self.entities),
                "total_relationships": len(self.relationships),
                "types": types,
            }]

        return []


@pytest.fixture
def mock_kg_service() -> KnowledgeGraphService:
    """Fixture providing an isolated in-memory KnowledgeGraphService."""
    manager = MockNeo4jManager()
    service = KnowledgeGraphService(connection_manager=manager)
    seed_sample_knowledge_graph(service=service)
    return service


@pytest.fixture
def client(mock_kg_service: KnowledgeGraphService) -> TestClient:
    """TestClient with injected mock KnowledgeGraphService."""
    with patch("app.api.routes.knowledge_graph.get_kg_service", return_value=mock_kg_service), \
         patch("app.knowledge_graph.service.get_kg_service", return_value=mock_kg_service), \
         patch("app.knowledge_graph.connection.get_neo4j_manager", return_value=mock_kg_service.manager):
        return TestClient(app, raise_server_exceptions=True)


# --------------------------------------------------------------------------- #
# 1. Models & Query Tests
# --------------------------------------------------------------------------- #


class TestKnowledgeGraphModels:
    def test_entity_model_validation(self) -> None:
        entity = MedicalEntity(
            id="disease_test",
            name="Test Disease",
            type=EntityType.DISEASE,
            description="Clinical description",
            confidence=0.95,
        )
        assert entity.id == "disease_test"
        assert entity.type == "Disease"
        assert entity.confidence == 0.95

    def test_relationship_model_validation(self) -> None:
        rel = MedicalRelationship(
            source_id="disease_test",
            type=RelationshipType.HAS_SYMPTOM,
            target_id="symptom_test",
            confidence=0.9,
        )
        assert rel.source_id == "disease_test"
        assert rel.type == "HAS_SYMPTOM"
        assert rel.target_id == "symptom_test"

    def test_build_upsert_relationship_query_safety(self) -> None:
        valid_query = build_upsert_relationship_query("TREATED_BY")
        assert "MERGE (s)-[r:TREATED_BY]->(t)" in valid_query

        # Prevent Cypher Injection attempts
        with pytest.raises(ValueError):
            build_upsert_relationship_query("INVALID_TYPE; MATCH (n) DETACH DELETE n;")


# --------------------------------------------------------------------------- #
# 2. Knowledge Graph Service CRUD & Traversals
# --------------------------------------------------------------------------- #


class TestKnowledgeGraphService:
    def test_create_and_get_entity(self, mock_kg_service: KnowledgeGraphService) -> None:
        new_entity = MedicalEntity(
            id="drug_aspirin",
            name="Aspirin",
            type="Drug",
            description="Antiplatelet medication",
            confidence=1.0,
        )
        created = mock_kg_service.create_entity(new_entity)
        assert created.id == "drug_aspirin"

        fetched = mock_kg_service.get_entity("drug_aspirin")
        assert fetched is not None
        assert fetched.name == "Aspirin"

    def test_list_entities_with_filter(self, mock_kg_service: KnowledgeGraphService) -> None:
        diseases = mock_kg_service.list_entities(entity_type="Disease")
        assert len(diseases) >= 2
        for d in diseases:
            assert d.type.lower() == "disease"

    def test_create_relationship_between_entities(self, mock_kg_service: KnowledgeGraphService) -> None:
        rel = RelationshipCreateRequest(
            source_id="drug_metformin",
            type="ASSOCIATED_WITH",
            target_id="disease_essential_hypertension",
            description="Comorbid metabolic treatment",
        )
        created_rel = mock_kg_service.create_relationship(rel)
        assert created_rel.source_id == "drug_metformin"
        assert created_rel.type == "ASSOCIATED_WITH"

    def test_create_relationship_invalid_entity_raises(self, mock_kg_service: KnowledgeGraphService) -> None:
        rel = RelationshipCreateRequest(
            source_id="non_existent_source",
            type="TREATED_BY",
            target_id="drug_metformin",
        )
        with pytest.raises(MalformedGraphDataError):
            mock_kg_service.create_relationship(rel)

    def test_find_diseases_from_symptoms(self, mock_kg_service: KnowledgeGraphService) -> None:
        matches = mock_kg_service.find_diseases_from_symptoms(["Fatigue", "Increased Thirst"])
        assert len(matches) > 0
        top_match = matches[0]
        assert "diabetes" in top_match.disease.id.lower()
        assert len(top_match.matched_symptoms) >= 2

    def test_find_treatments(self, mock_kg_service: KnowledgeGraphService) -> None:
        treatments = mock_kg_service.find_treatments("Type 2 Diabetes Mellitus")
        assert len(treatments) >= 1
        names = [t.name for t in treatments]
        assert "Metformin" in names

    def test_find_risk_factors(self, mock_kg_service: KnowledgeGraphService) -> None:
        risk_factors = mock_kg_service.find_risk_factors("Type 2 Diabetes Mellitus")
        assert len(risk_factors) >= 1
        names = [rf.name for rf in risk_factors]
        assert "Obesity" in names

    def test_find_contraindications(self, mock_kg_service: KnowledgeGraphService) -> None:
        contraindications = mock_kg_service.find_contraindications("Metformin")
        assert len(contraindications) >= 1
        contra_names = [c["contraindicated_with"].name for c in contraindications]
        assert "Severe Renal Impairment" in contra_names

    def test_get_patient_graph_context(self, mock_kg_service: KnowledgeGraphService) -> None:
        # Create a patient node and connect symptoms
        mock_kg_service.create_entity(
            MedicalEntity(id="patient_001", name="Patient John Doe", type="Patient")
        )
        mock_kg_service.create_relationship(
            RelationshipCreateRequest(
                source_id="patient_001",
                type="HAS_SYMPTOM",
                target_id="symptom_fatigue",
            )
        )

        context = mock_kg_service.get_context_for_patient("patient_001")
        assert context.patient_id == "patient_001"
        assert len(context.entities) >= 2
        assert len(context.relationships) >= 1


# --------------------------------------------------------------------------- #
# 3. FastAPI REST Endpoints Tests
# --------------------------------------------------------------------------- #


class TestKnowledgeGraphAPI:
    def test_get_stats_endpoint(self, client: TestClient) -> None:
        response = client.get("/api/v1/knowledge-graph/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["total_entities"] > 0
        assert data["data"]["total_relationships"] > 0

    def test_list_entities_endpoint(self, client: TestClient) -> None:
        response = client.get("/api/v1/knowledge-graph/entities?entity_type=Disease")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert isinstance(data["data"], list)
        assert len(data["data"]) >= 1

    def test_get_entity_detail_endpoint(self, client: TestClient) -> None:
        response = client.get("/api/v1/knowledge-graph/entities/disease_type_2_diabetes")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["entity"]["id"] == "disease_type_2_diabetes"
        assert len(data["data"]["outgoing_relationships"]) > 0

    def test_get_entity_not_found(self, client: TestClient) -> None:
        response = client.get("/api/v1/knowledge-graph/entities/non_existent_entity")
        assert response.status_code == 404

    def test_create_entity_endpoint(self, client: TestClient) -> None:
        payload = {
            "id": "drug_atorvastatin",
            "name": "Atorvastatin",
            "type": "Drug",
            "description": "HMG-CoA reductase inhibitor (statin)",
            "confidence": 1.0,
        }
        response = client.post("/api/v1/knowledge-graph/entities", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["data"]["id"] == "drug_atorvastatin"

    def test_create_relationship_endpoint(self, client: TestClient) -> None:
        payload = {
            "source_id": "disease_type_2_diabetes",
            "type": "RECOMMENDED_TEST",
            "target_id": "test_blood_pressure",
            "description": "Routine comorbidity evaluation",
        }
        response = client.post("/api/v1/knowledge-graph/relationships", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert data["data"]["type"] == "RECOMMENDED_TEST"

    def test_disease_diagnosis_by_symptoms_endpoint(self, client: TestClient) -> None:
        response = client.get("/api/v1/knowledge-graph/diseases?symptoms=fatigue,increased thirst")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) >= 1
        assert "diabetes" in data["data"][0]["disease"]["id"].lower()

    def test_disease_treatments_endpoint(self, client: TestClient) -> None:
        response = client.get("/api/v1/knowledge-graph/disease_type_2_diabetes/treatments")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) >= 1

    def test_disease_risk_factors_endpoint(self, client: TestClient) -> None:
        response = client.get("/api/v1/knowledge-graph/disease_type_2_diabetes/risk-factors")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) >= 1

    def test_drug_contraindications_endpoint(self, client: TestClient) -> None:
        response = client.get("/api/v1/knowledge-graph/drug_metformin/contraindications")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) >= 1
