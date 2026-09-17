"""
app/llm/context.py

Context assembler synthesizing Patient Facts, ChromaDB RAG Evidence,
Neo4j Knowledge Graph relations, and Symbolic Reasoning results into a unified payload.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from app.core.logging import get_logger
from app.knowledge_graph.service import KnowledgeGraphService, get_kg_service
from app.rag.retriever import RAGRetriever, get_retriever
from app.reasoning.engine import SymbolicRuleEngine, get_rule_engine
from app.reasoning.models import PatientProfile, ReasoningResult

logger = get_logger(__name__)


class ClinicalContextAssembler:
    """Assembles all neuro-symbolic layers into a structured context for the LLM."""

    def __init__(
        self,
        retriever: Optional[RAGRetriever] = None,
        kg_service: Optional[KnowledgeGraphService] = None,
        rule_engine: Optional[SymbolicRuleEngine] = None,
    ) -> None:
        self.retriever = retriever
        self.kg_service = kg_service
        self.rule_engine = rule_engine

    def _get_retriever(self) -> RAGRetriever:
        if self.retriever is None:
            self.retriever = get_retriever()
        return self.retriever

    def _get_kg_service(self) -> KnowledgeGraphService:
        if self.kg_service is None:
            self.kg_service = get_kg_service()
        return self.kg_service

    def _get_rule_engine(self) -> SymbolicRuleEngine:
        if self.rule_engine is None:
            self.rule_engine = get_rule_engine()
        return self.rule_engine

    def retrieve_rag_evidence(
        self, patient: PatientProfile, top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant guideline chunks from ChromaDB for the patient profile."""
        terms: List[str] = []
        if patient.conditions:
            terms.extend(patient.conditions)
        if patient.current_medications:
            terms.extend(patient.current_medications)
        if patient.symptoms:
            terms.extend(patient.symptoms)

        # Add clinical abnormalities to query
        if patient.lab_results.get("egfr", 100.0) < 30.0:
            terms.append("severe renal impairment egfr lactic acidosis contraindication")
        elif patient.lab_results.get("egfr", 100.0) < 60.0:
            terms.append("chronic kidney disease renoprotection ace inhibitor")

        if patient.lab_results.get("serum_potassium", 4.0) >= 5.5:
            terms.append("hyperkalemia spironolactone lisinopril interaction")

        query_text = " ".join(terms).strip()
        if not query_text:
            query_text = "clinical treatment guidelines and safety protocols"

        try:
            retriever = self._get_retriever()
            rag_res = retriever.retrieve(query=query_text, top_k=top_k)
            evidence: List[Dict[str, Any]] = []
            for chunk in rag_res.results:
                evidence.append(
                    {
                        "chunk_id": chunk.chunk_id,
                        "document": chunk.filename,
                        "page_number": chunk.page_number,
                        "content": chunk.content,
                        "similarity_score": round(chunk.score, 4),
                    }
                )
            return evidence
        except Exception as exc:
            logger.warning("RAG retrieval failed during context assembly: %s", str(exc))
            return []

    def retrieve_kg_facts(self, patient: PatientProfile) -> List[Dict[str, Any]]:
        """Retrieve topological relationships and contraindications from Neo4j."""
        facts: List[Dict[str, Any]] = []
        try:
            kg_svc = self._get_kg_service()
            # 1. Direct patient context if patient node exists
            context = kg_svc.get_context_for_patient(patient.patient_id)
            if context and context.relationships:
                for rel in context.relationships:
                    facts.append(
                        {
                            "source": rel.source_name or rel.source_id,
                            "relationship": rel.type,
                            "target": rel.target_name or rel.target_id,
                            "description": rel.description or "",
                            "confidence": rel.confidence,
                        }
                    )

            # 2. Check contraindications for patient's current medications
            for med in patient.current_medications:
                contraindications = kg_svc.find_contraindications(med)
                for item in contraindications:
                    other_ent = item.get("contraindicated_with")
                    other_name = (
                        other_ent.name
                        if hasattr(other_ent, "name")
                        else str(other_ent)
                    )
                    facts.append(
                        {
                            "source": med,
                            "relationship": "CONTRAINDICATED_WITH",
                            "target": other_name,
                            "description": item.get("description", ""),
                            "confidence": item.get("confidence", 1.0),
                        }
                    )

            # 3. Check indicated treatments for patient's conditions
            for cond in patient.conditions:
                treatments = kg_svc.find_treatments(cond)
                for t in treatments:
                    facts.append(
                        {
                            "source": cond,
                            "relationship": "TREATED_BY",
                            "target": t.name,
                            "description": t.description or "",
                            "confidence": t.confidence,
                        }
                    )

            # 4. Check risk factors for patient's conditions
            for cond in patient.conditions:
                risk_factors = kg_svc.find_risk_factors(cond)
                for rf in risk_factors:
                    facts.append(
                        {
                            "source": rf.name,
                            "relationship": "RISK_FACTOR_FOR",
                            "target": cond,
                            "description": rf.description or f"{rf.name} is a documented risk factor for {cond}.",
                            "confidence": rf.confidence,
                        }
                    )

            # Deduplicate facts preserving order
            seen = set()
            unique_facts: List[Dict[str, Any]] = []
            for f in facts:
                key = (f["source"].strip().lower(), f["relationship"].strip().lower(), f["target"].strip().lower())
                if key not in seen:
                    seen.add(key)
                    unique_facts.append(f)

            return unique_facts
        except Exception as exc:
            logger.warning("Knowledge Graph retrieval failed: %s", str(exc))
            return []

    def assemble(
        self,
        patient: Union[PatientProfile, Dict[str, Any]],
        include_rag: bool = True,
        include_kg: bool = True,
    ) -> Dict[str, Any]:
        """Assemble full multi-source context including symbolic reasoning output.

        Returns:
            Dict containing patient facts, RAG evidence, KG facts, symbolic findings,
            and an index of valid citation identifiers.
        """
        patient_obj = (
            patient
            if isinstance(patient, PatientProfile)
            else PatientProfile(**patient)
        )

        # 1. Run deterministic symbolic reasoning
        engine = self._get_rule_engine()
        reasoning_res: ReasoningResult = engine.evaluate(patient_obj)

        # 2. Retrieve RAG evidence
        rag_evidence = self.retrieve_rag_evidence(patient_obj) if include_rag else []

        # 3. Retrieve KG facts
        kg_facts = self.retrieve_kg_facts(patient_obj) if include_kg else []

        # 4. Extract valid citation references for guardrail verification
        valid_rule_ids = [r.rule_id for r in reasoning_res.triggered_rules]
        valid_documents = list(
            {chunk.get("document", "") for chunk in rag_evidence if chunk.get("document")}
        )
        for r in reasoning_res.triggered_rules:
            if r.guideline_source and r.guideline_source not in valid_documents:
                valid_documents.append(r.guideline_source)

        return {
            "patient_facts": patient_obj.model_dump(),
            "knowledge_graph_facts": kg_facts,
            "rag_evidence": rag_evidence,
            "symbolic_reasoning": {
                "rules_evaluated_count": reasoning_res.total_rules_evaluated,
                "triggered_rules_count": reasoning_res.triggered_rules_count,
                "triggered_rules": [r.model_dump() for r in reasoning_res.triggered_rules],
                "derived_facts": [f.model_dump() for f in reasoning_res.derived_facts],
                "critical_alerts": [a.model_dump() for a in reasoning_res.critical_alerts],
                "reasoning_trace": [t.model_dump() for t in reasoning_res.reasoning_trace],
            },
            "valid_citations": {
                "rule_ids": valid_rule_ids,
                "documents": valid_documents,
            },
        }


_cached_context_assembler: Optional[ClinicalContextAssembler] = None


def get_context_assembler() -> ClinicalContextAssembler:
    """Singleton getter for ClinicalContextAssembler."""
    global _cached_context_assembler
    if _cached_context_assembler is None:
        _cached_context_assembler = ClinicalContextAssembler()
    return _cached_context_assembler
