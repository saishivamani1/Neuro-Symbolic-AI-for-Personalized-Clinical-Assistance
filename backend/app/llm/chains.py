"""
app/llm/chains.py

LangChain Runnable pipeline for structured neuro-symbolic clinical assistance.

Orchestrates:
ChatPromptTemplate -> ChatOpenAI (with_structured_output) -> Guardrail Validation
"""

import json
import re
from typing import Any, Dict, Optional

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable, RunnableLambda

from app.core.exceptions import LLMError
from app.core.logging import get_logger
from app.llm.guardrails import ClinicalGuardrailValidator
from app.llm.models import get_llm_factory
from app.llm.prompts import get_clinical_prompt_template
from app.llm.schemas import ClinicalAssistantResponse

logger = get_logger(__name__)


def _clean_and_parse_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """Clean markdown code blocks and attempt robust JSON parsing."""
    if not raw_text:
        return None
    # Strip markdown fences ```json ... ```
    text = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)

    try:
        return json.loads(text)
    except Exception:
        pass

    # Find first { and matching last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            # Fix trailing commas before } or ]
            candidate = re.sub(r",\s*([\]}])", r"\1", candidate)
            try:
                return json.loads(candidate)
            except Exception:
                pass
    return None


def _build_compact_context_for_prompt(context: Dict[str, Any]) -> Dict[str, Any]:
    """Prune and compact context to prevent exceeding token-per-minute (TPM) limits."""
    # 1. Patient facts
    patient = context.get("patient_facts", {})
    compact_patient = {
        "id": patient.get("patient_id"),
        "name": patient.get("name"),
        "age": patient.get("age"),
        "gender": patient.get("gender"),
        "vitals": patient.get("vitals", {}),
        "labs": patient.get("lab_results", {}),
        "conditions": patient.get("conditions", []),
        "medications": patient.get("current_medications", []),
        "symptoms": patient.get("symptoms", []),
    }

    # 2. Symbolic Reasoning
    symbolic = context.get("symbolic_reasoning", {})
    compact_rules = []
    for r in symbolic.get("triggered_rules", []):
        compact_rules.append({
            "rule_id": r.get("rule_id"),
            "name": r.get("name"),
            "severity": r.get("severity"),
            "conclusion": (r.get("actions", [{}])[0].get("message") if r.get("actions") else r.get("description", "")),
            "guideline": r.get("guideline_citation") or r.get("guideline_source"),
            "page": r.get("guideline_page"),
        })

    compact_alerts = []
    for a in symbolic.get("critical_alerts", []):
        compact_alerts.append({
            "category": a.get("category"),
            "severity": a.get("severity"),
            "alert": a.get("explanation"),
        })

    compact_facts = []
    for f in symbolic.get("derived_facts", []):
        compact_facts.append({
            "fact_type": f.get("fact_type"),
            "desc": f.get("description"),
        })

    # 3. RAG Evidence (top 3 chunks, max 250 chars excerpt each)
    rag_evidence = context.get("rag_evidence", [])
    compact_rag = []
    for chunk in rag_evidence[:3]:
        content = str(chunk.get("content", ""))
        if len(content) > 250:
            content = content[:250] + "..."
        compact_rag.append({
            "document": chunk.get("document"),
            "page": chunk.get("page_number"),
            "excerpt": content,
        })

    # 4. KG Facts (top 6 facts)
    kg_facts = context.get("knowledge_graph_facts", [])
    compact_kg = []
    for f in kg_facts[:6]:
        compact_kg.append({
            "source": f.get("source"),
            "relationship": f.get("relationship"),
            "target": f.get("target"),
            "description": f.get("description", ""),
        })

    return {
        "patient": compact_patient,
        "symbolic_alerts": compact_alerts,
        "symbolic_rules": compact_rules,
        "derived_facts": compact_facts,
        "guidelines_rag": compact_rag,
        "knowledge_graph": compact_kg,
    }


def build_clinical_assistant_chain(
    model: Optional[Any] = None,
) -> Runnable[Dict[str, Any], ClinicalAssistantResponse]:
    """Construct the LangChain runnable chain for clinical decision assistance."""
    if model is None:
        factory = get_llm_factory()
        model = factory.get_model()

    prompt_template = get_clinical_prompt_template()

    # Bind Pydantic output schema.
    def _needs_function_calling(m: Any) -> bool:
        """Return True if this model needs explicit function_calling method (not json_schema)."""
        module = type(m).__module__ or ""
        cls_name = type(m).__name__ or ""
        # Native ChatGroq class always needs function calling
        if "groq" in module.lower() or "groq" in cls_name.lower():
            return True
        # ChatOpenAI pointed at a non-OpenAI base URL (HuggingFace router, Groq compat, etc.)
        # LangChain stores base_url as SecretStr or plain str — unwrap safely
        base_url = getattr(m, "openai_api_base", None) or getattr(m, "base_url", None)
        if base_url is not None:
            # SecretStr has get_secret_value(), plain str is used directly
            base_str = base_url.get_secret_value() if hasattr(base_url, "get_secret_value") else str(base_url)
            base_str = base_str.lower()
            if "huggingface" in base_str or "groq" in base_str or "router." in base_str:
                return True
        return False

    try:
        if _needs_function_calling(model):
            logger.info("Using function_calling structured output for %s", type(model).__name__)
            # Explicitly use function_calling — HuggingFace router does NOT support json_schema
            structured_model = model.with_structured_output(
                ClinicalAssistantResponse, method="function_calling"
            )
        else:
            structured_model = model.with_structured_output(
                ClinicalAssistantResponse, method="json_mode"
            )
    except Exception:
        # Final fallback: default tool calling
        structured_model = model.with_structured_output(ClinicalAssistantResponse)

    def _execute_chain_with_guardrails(input_dict: Dict[str, Any]) -> ClinicalAssistantResponse:
        context_dict = input_dict.get("context", {})
        question = input_dict.get("question", "What are the important clinical considerations for this patient?")

        # Compact context to minimize prompt size and avoid token limit errors
        compact_context = _build_compact_context_for_prompt(context_dict)

        prompt_input = {
            "structured_context_json": json.dumps(compact_context, separators=(',', ':')),
            "question": question,
        }

        # Format prompt and invoke LLM
        messages = prompt_template.format_messages(**prompt_input)
        parsed_response: Optional[ClinicalAssistantResponse] = None

        try:
            raw_response = structured_model.invoke(messages)
            if isinstance(raw_response, ClinicalAssistantResponse):
                parsed_response = raw_response
            elif isinstance(raw_response, dict):
                parsed_response = ClinicalAssistantResponse(**raw_response)
        except Exception as tool_err:
            logger.warning("Structured tool-use failed (%s); attempting direct JSON fallback.", str(tool_err))

        if parsed_response is None:
            # Resilient fallback: invoke raw model requesting pure JSON
            fallback_msgs = [
                *messages,
                HumanMessage(
                    content=(
                        "Output ONLY a single valid JSON object representing the ClinicalAssistantResponse. "
                        "Do not include markdown or explanations outside the JSON."
                    )
                ),
            ]
            try:
                raw_ai = model.invoke(fallback_msgs)
                raw_text = getattr(raw_ai, "content", "")
                data = _clean_and_parse_json(str(raw_text))
                if data and isinstance(data, dict):
                    parsed_response = ClinicalAssistantResponse.model_validate(data)
            except Exception as direct_err:
                logger.error("Direct JSON fallback also failed: %s", str(direct_err))

        if parsed_response is None:
            raise LLMError(
                detail="LLM failed to return structured ClinicalAssistantResponse.",
                context={"provider": type(model).__name__},
            )

        # Apply deterministic guardrails
        validated_response = ClinicalGuardrailValidator.validate_and_sanitize(
            response=parsed_response,
            context=context_dict,
        )

        return validated_response

    return RunnableLambda(_execute_chain_with_guardrails)
