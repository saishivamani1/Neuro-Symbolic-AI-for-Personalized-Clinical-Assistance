"""
app/knowledge_graph/queries.py

Parameterized Cypher queries for Neo4j Medical Knowledge Graph operations.

Guarantees safety against Cypher injection by using strict parameter binding and
whitelisted relationship types.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Schema & Constraint Queries
# --------------------------------------------------------------------------- #

CREATE_UNIQUE_ENTITY_ID_CONSTRAINT = """
CREATE CONSTRAINT medical_entity_id_unique IF NOT EXISTS
FOR (e:MedicalEntity)
REQUIRE e.id IS UNIQUE
"""

CREATE_ENTITY_NAME_INDEX = """
CREATE INDEX medical_entity_name_idx IF NOT EXISTS
FOR (e:MedicalEntity)
ON (e.name)
"""

CREATE_ENTITY_TYPE_INDEX = """
CREATE INDEX medical_entity_type_idx IF NOT EXISTS
FOR (e:MedicalEntity)
ON (e.type)
"""

# --------------------------------------------------------------------------- #
# Entity Queries
# --------------------------------------------------------------------------- #

UPSERT_ENTITY_QUERY = """
MERGE (e:MedicalEntity {id: $id})
SET e.name = $name,
    e.type = $type,
    e.description = $description,
    e.source_document = $source_document,
    e.page_number = $page_number,
    e.section = $section,
    e.confidence = $confidence,
    e.properties = $properties_json,
    e.updated_at = timestamp()
RETURN e
"""

BATCH_UPSERT_ENTITIES_QUERY = """
UNWIND $batch AS item
MERGE (e:MedicalEntity {id: item.id})
SET e.name = item.name,
    e.type = item.type,
    e.description = item.description,
    e.source_document = item.source_document,
    e.page_number = item.page_number,
    e.section = item.section,
    e.confidence = item.confidence,
    e.properties = item.properties_json,
    e.updated_at = timestamp()
RETURN count(e) AS upserted_count
"""

GET_ENTITY_BY_ID_QUERY = """
MATCH (e:MedicalEntity {id: $id})
RETURN e
"""

LIST_ENTITIES_QUERY = """
MATCH (e:MedicalEntity)
WHERE ($entity_type IS NULL OR toLower(e.type) = toLower($entity_type))
  AND ($search IS NULL OR toLower(e.name) CONTAINS toLower($search) OR toLower(e.id) CONTAINS toLower($search))
RETURN e
ORDER BY e.name ASC
LIMIT $limit
"""

DELETE_ENTITY_QUERY = """
MATCH (e:MedicalEntity {id: $id})
DETACH DELETE e
RETURN count(e) AS deleted_count
"""

# --------------------------------------------------------------------------- #
# Relationship Queries (Dynamically typed with injection-safe validation)
# --------------------------------------------------------------------------- #

VALID_REL_TYPE_REGEX = re.compile(r"^[A-Z0-9_]{2,64}$")


def build_upsert_relationship_query(rel_type: str) -> str:
    """Safely build a Cypher query for creating a relationship of specific type."""
    normalized_type = rel_type.strip().upper()
    if not VALID_REL_TYPE_REGEX.match(normalized_type):
        raise ValueError(f"Invalid relationship type format: '{rel_type}'")

    return f"""
    MATCH (s:MedicalEntity {{id: $source_id}})
    MATCH (t:MedicalEntity {{id: $target_id}})
    MERGE (s)-[r:{normalized_type}]->(t)
    SET r.description = $description,
        r.source_document = $source_document,
        r.page_number = $page_number,
        r.section = $section,
        r.confidence = $confidence,
        r.properties = $properties_json,
        r.updated_at = timestamp()
    RETURN s, r, t, type(r) AS rel_type
    """


def build_batch_upsert_relationships_query(rel_type: str) -> str:
    """Safely build a Cypher query for batch creating relationships of specific type."""
    normalized_type = rel_type.strip().upper()
    if not VALID_REL_TYPE_REGEX.match(normalized_type):
        raise ValueError(f"Invalid relationship type format: '{rel_type}'")

    return f"""
    UNWIND $batch AS item
    MATCH (s:MedicalEntity {{id: item.source_id}})
    MATCH (t:MedicalEntity {{id: item.target_id}})
    MERGE (s)-[r:{normalized_type}]->(t)
    SET r.description = item.description,
        r.source_document = item.source_document,
        r.page_number = item.page_number,
        r.section = item.section,
        r.confidence = item.confidence,
        r.properties = item.properties_json,
        r.updated_at = timestamp()
    RETURN count(r) AS upserted_count
    """


GET_OUTGOING_RELATIONSHIPS_QUERY = """
MATCH (s:MedicalEntity {id: $id})-[r]->(t:MedicalEntity)
RETURN s.id AS source_id, s.name AS source_name,
       type(r) AS rel_type,
       t.id AS target_id, t.name AS target_name,
       r.description AS description,
       r.source_document AS source_document,
       r.page_number AS page_number,
       r.section AS section,
       r.confidence AS confidence,
       r.properties AS properties_json
"""

GET_INCOMING_RELATIONSHIPS_QUERY = """
MATCH (s:MedicalEntity)-[r]->(t:MedicalEntity {id: $id})
RETURN s.id AS source_id, s.name AS source_name,
       type(r) AS rel_type,
       t.id AS target_id, t.name AS target_name,
       r.description AS description,
       r.source_document AS source_document,
       r.page_number AS page_number,
       r.section AS section,
       r.confidence AS confidence,
       r.properties AS properties_json
"""

FIND_RELATED_ENTITIES_QUERY = """
MATCH (s:MedicalEntity {id: $id})-[r]-(t:MedicalEntity)
RETURN t,
       type(r) AS rel_type,
       (CASE WHEN startNode(r) = s THEN 'outgoing' ELSE 'incoming' END) AS direction,
       coalesce(r.confidence, 1.0) AS confidence
LIMIT $limit
"""

# --------------------------------------------------------------------------- #
# Clinical Domain Graph Queries
# --------------------------------------------------------------------------- #

FIND_DISEASES_FROM_SYMPTOMS_QUERY = """
MATCH (d:MedicalEntity)-[:HAS_SYMPTOM]->(s:MedicalEntity)
WHERE (toLower(d.type) IN ['disease', 'condition'])
  AND (toLower(s.name) IN $symptoms OR toLower(s.id) IN $symptoms)
WITH d, collect(DISTINCT s.name) AS matched_symptoms, count(DISTINCT s) AS match_count
MATCH (d)-[:HAS_SYMPTOM]->(all_s:MedicalEntity)
WITH d, matched_symptoms, match_count, count(DISTINCT all_s) AS total_disease_symptoms
RETURN d, matched_symptoms, match_count, total_disease_symptoms,
       (toFloat(match_count) / toFloat(total_disease_symptoms)) AS confidence
ORDER BY match_count DESC, confidence DESC
LIMIT $limit
"""

FIND_TREATMENTS_FOR_DISEASE_QUERY = """
MATCH (d:MedicalEntity)-[r:TREATED_BY]->(t:MedicalEntity)
WHERE (toLower(d.name) = toLower($disease)
   OR (size(d.name) >= 4 AND toLower($disease) CONTAINS toLower(d.name))
   OR (size($disease) >= 4 AND toLower(d.name) CONTAINS toLower($disease))
   OR d.id = $disease)
RETURN t, r, d
ORDER BY coalesce(r.confidence, 1.0) DESC
"""

FIND_RISK_FACTORS_FOR_DISEASE_QUERY = """
MATCH (rf:MedicalEntity)-[r:RISK_FACTOR_FOR]->(d:MedicalEntity)
WHERE (toLower(d.name) = toLower($disease)
   OR (size(d.name) >= 4 AND toLower($disease) CONTAINS toLower(d.name))
   OR (size($disease) >= 4 AND toLower(d.name) CONTAINS toLower($disease))
   OR d.id = $disease)
RETURN rf, r, d
ORDER BY coalesce(r.confidence, 1.0) DESC
"""

FIND_CONTRAINDICATIONS_FOR_DRUG_QUERY = """
MATCH (drug:MedicalEntity)-[r:CONTRAINDICATED_WITH]-(other:MedicalEntity)
WHERE (toLower(drug.name) = toLower($drug)
   OR (size(drug.name) >= 4 AND toLower($drug) CONTAINS toLower(drug.name))
   OR (size($drug) >= 4 AND toLower(drug.name) CONTAINS toLower($drug))
   OR drug.id = $drug)
RETURN other, r, drug
ORDER BY coalesce(r.confidence, 1.0) DESC
"""

GET_PATIENT_GRAPH_CONTEXT_QUERY = """
MATCH (p:MedicalEntity {id: $patient_id})
OPTIONAL MATCH (p)-[r1]->(direct:MedicalEntity)
OPTIONAL MATCH (direct)-[r2]->(second_hop:MedicalEntity)
WHERE NOT second_hop.id = p.id
RETURN p,
       collect(DISTINCT direct) AS direct_entities,
       collect(DISTINCT {
           source_id: startNode(r1).id,
           target_id: endNode(r1).id,
           type: type(r1),
           properties: r1.properties
       }) AS direct_relationships,
       collect(DISTINCT second_hop) AS second_hop_entities,
       collect(DISTINCT {
           source_id: startNode(r2).id,
           target_id: endNode(r2).id,
           type: type(r2),
           properties: r2.properties
       }) AS second_hop_relationships
"""

GET_GRAPH_STATISTICS_QUERY = """
MATCH (e:MedicalEntity)
WITH count(e) AS total_entities, collect(e.type) AS types
OPTIONAL MATCH ()-[r]->()
RETURN total_entities,
       count(r) AS total_relationships,
       types
"""

GET_RELATIONSHIP_COUNTS_QUERY = """
MATCH ()-[r]->()
RETURN type(r) AS rel_type, count(r) AS count
ORDER BY count DESC
"""

# Fetch top disease hubs by degree
GET_TOP_DISEASES_QUERY = """
MATCH (d:MedicalEntity)
WHERE toLower(d.type) = 'disease'
OPTIONAL MATCH (d)-[r]->()
WITH d, count(r) AS degree
ORDER BY degree DESC
RETURN d.id AS id, d.name AS name, d.type AS type, d.description AS description
LIMIT $disease_limit
"""

# Fetch outgoing relationships from specified disease hubs
GET_EDGES_FOR_DISEASES_QUERY = """
MATCH (d:MedicalEntity)-[r]->(t:MedicalEntity)
WHERE d.id IN $disease_ids
RETURN d.id AS source_id, d.name AS source_name, d.type AS source_type, d.description AS source_desc,
       t.id AS target_id, t.name AS target_name, t.type AS target_type,
       type(r) AS rel_type
"""

# Fetch a representative subgraph for visual canvas rendering (legacy fallback)
GET_SAMPLE_GRAPH_QUERY = """
MATCH (d:MedicalEntity)
WHERE toLower(d.type) = 'disease'
OPTIONAL MATCH (d)-[r]->(t:MedicalEntity)
RETURN d.id AS source_id, d.name AS source_name, d.type AS source_type, d.description AS source_desc,
       t.id AS target_id, t.name AS target_name, t.type AS target_type,
       type(r) AS rel_type
LIMIT $total_limit
"""

