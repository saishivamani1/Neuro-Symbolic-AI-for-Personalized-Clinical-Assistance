"""
app/reasoning/evidence.py

Evidence and Clinical Fact Assembler.

Bridges Patient Facts, Neo4j Knowledge Graph contextual relations, and RAG guideline
chunks into a unified, evidence-grounded working memory for symbolic reasoning.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from app.core.logging import get_logger
from app.knowledge_graph.models import PatientGraphContext
from app.knowledge_graph.service import KnowledgeGraphService, get_kg_service
from app.rag.retriever import RAGRetriever, get_retriever
from app.reasoning.models import PatientProfile
from app.schemas.rag import RAGSearchResultChunk

logger = get_logger(__name__)


class EvidenceAssembler:
    """Assembles patient data, Knowledge Graph context, and RAG evidence."""

    def __init__(
        self,
        kg_service: Optional[KnowledgeGraphService] = None,
        retriever: Optional[RAGRetriever] = None,
    ) -> None:
        self.kg_service = kg_service
        self.retriever = retriever

    def _get_kg_service(self) -> KnowledgeGraphService:
        if self.kg_service is None:
            self.kg_service = get_kg_service()
        return self.kg_service

    def _get_retriever(self) -> RAGRetriever:
        if self.retriever is None:
            self.retriever = get_retriever()
        return self.retriever

    def assemble_patient_working_memory(
        self,
        patient: Union[PatientProfile, Dict[str, Any]],
        include_kg: bool = True,
        include_rag: bool = True,
    ) -> Dict[str, Any]:
        """Combine patient profile with graph relationships and retrieved clinical evidence.

        Parameters
        ----------
        patient : Union[PatientProfile, Dict[str, Any]]
            Patient profile.
        include_kg : bool
            Whether to enrich working memory with Neo4j Knowledge Graph facts.
        include_rag : bool
            Whether to retrieve guideline chunks from ChromaDB.

        Returns
        -------
        Dict[str, Any]
            Augmented working memory dictionary.
        """
        if isinstance(patient, PatientProfile):
            memory = patient.model_dump()
        else:
            memory = dict(patient)

        patient_id = memory.get("patient_id", "unknown_patient")
        conditions = memory.get("conditions", [])
        medications = memory.get("current_medications", [])
        symptoms = memory.get("symptoms", [])

        # 1. Knowledge Graph Context
        kg_facts: List[Dict[str, Any]] = []
        if include_kg:
            try:
                kg_svc = self._get_kg_service()
                # Direct patient context
                context = kg_svc.get_context_for_patient(patient_id)
                if context and context.relationships:
                    for rel in context.relationships:
                        kg_facts.append({
                            "source": rel.source_name or rel.source_id,
                            "relationship": rel.type,
                            "target": rel.target_name or rel.target_id,
                            "confidence": rel.confidence,
                        })

                # Check drug contraindications for patient's medications
                for med in medications:
                    contraindications = kg_svc.find_contraindications(med)
                    for item in contraindications:
                        other_ent = item.get("contraindicated_with")
                        other_name = other_ent.name if hasattr(other_ent, "name") else str(other_ent)
                        kg_facts.append({
                            "source": med,
                            "relationship": "CONTRAINDICATED_WITH",
                            "target": other_name,
                            "description": item.get("description"),
                            "confidence": item.get("confidence", 1.0),
                        })

                # Check treatments indicated for patient conditions
                for cond in conditions:
                    treatments = kg_svc.find_treatments(cond)
                    for t in treatments:
                        kg_facts.append({
                            "source": cond,
                            "relationship": "TREATED_BY",
                            "target": t.name,
                            "description": t.description or "",
                            "confidence": t.confidence,
                        })

                memory["kg_facts"] = kg_facts
                logger.info(
                    "Enriched patient %s with %d Knowledge Graph facts",
                    patient_id,
                    len(kg_facts),
                )
            except Exception as exc:
                logger.warning(
                    "Could not retrieve Knowledge Graph context for %s: %s",
                    patient_id,
                    str(exc),
                )
                memory["kg_facts"] = []

        # 2. RAG Guideline Evidence
        rag_evidence: List[Dict[str, Any]] = []
        if include_rag:
            try:
                rag_ret = self._get_retriever()
                search_terms: List[str] = []
                if conditions:
                    search_terms.append(" ".join(conditions[:2]))
                if medications:
                    search_terms.append(" ".join(medications[:2]))
                if symptoms:
                    search_terms.append(" ".join(symptoms[:2]))

                query_text = " ".join(search_terms)
                if query_text.strip():
                    rag_res = rag_ret.retrieve(query=query_text, top_k=3)
                    for chunk in rag_res.results:
                        rag_evidence.append({
                            "chunk_id": chunk.chunk_id,
                            "filename": chunk.filename,
                            "page_number": chunk.page_number,
                            "snippet": chunk.content[:250],
                            "similarity_score": chunk.score,
                        })

                memory["rag_evidence"] = rag_evidence
                logger.info(
                    "Enriched patient %s with %d RAG evidence chunks",
                    patient_id,
                    len(rag_evidence),
                )
            except Exception as exc:
                logger.warning(
                    "Could not retrieve RAG evidence for %s: %s",
                    patient_id,
                    str(exc),
                )
                memory["rag_evidence"] = []

        return memory


_cached_evidence_assembler: Optional[EvidenceAssembler] = None


def get_evidence_assembler() -> EvidenceAssembler:
    """Singleton getter for EvidenceAssembler."""
    global _cached_evidence_assembler
    if _cached_evidence_assembler is None:
        _cached_evidence_assembler = EvidenceAssembler()
    return _cached_evidence_assembler
