"""
app/api/routes/chat.py

POST /api/v1/chat — Conversational patient health companion endpoint.

Features:
- Direct, conversational answers to specific patient inquiries without repetitive recaps.
- Awareness of patient clinical context (vitals, labs, conditions, medications, safety alerts).
- Multi-turn conversation history support to prevent repeating greetings or medical records.
- Robust deterministic fallback if LLM is offline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from fastapi import APIRouter, status
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.logging import get_logger
from app.llm.models import get_llm_factory
from app.api.routes.patients import SAMPLE_PATIENTS, find_sample_patient
from app.schemas.response import APIResponse

router = APIRouter()
logger = get_logger(__name__)


class ChatHistoryItem(BaseModel):
    role: Literal["patient", "assistant", "user", "system"]
    text: str


class ChatRequest(BaseModel):
    patient_id: Optional[str] = None
    patient: Optional[Dict[str, Any]] = None
    message: str = Field(..., description="The patient's current question or message.")
    history: List[ChatHistoryItem] = Field(
        default_factory=list,
        description="Previous messages in the conversation for multi-turn context.",
    )


class ChatResponse(BaseModel):
    reply: str
    patient_id: Optional[str] = None
    category: str = "advice"


def _build_patient_context_string(patient_data: Dict[str, Any]) -> str:
    """Format structured patient data into a clear clinical context block for the LLM."""
    name = patient_data.get("name") or "Patient"
    age = patient_data.get("age") or "Adult"
    gender = patient_data.get("gender") or "Not specified"
    conditions = ", ".join(patient_data.get("conditions") or []) or "None reported"
    medications = ", ".join(patient_data.get("current_medications") or []) or "None"
    symptoms = ", ".join(patient_data.get("symptoms") or []) or "None reported"

    vitals = patient_data.get("vitals") or {}
    vitals_str = ", ".join(f"{k}: {v}" for k, v in vitals.items()) or "None recorded"

    labs = patient_data.get("lab_results") or {}
    labs_str = ", ".join(f"{k}: {v}" for k, v in labs.items()) or "None recorded"

    return (
        f"Patient Name: {name}\n"
        f"Age / Gender: {age} / {gender}\n"
        f"Active Diagnoses / Conditions: {conditions}\n"
        f"Current Medications: {medications}\n"
        f"Recent Vital Signs: {vitals_str}\n"
        f"Recent Lab Results: {labs_str}\n"
        f"Reported Symptoms: {symptoms}"
    )


def _generate_fallback_response(query: str, patient_data: Dict[str, Any]) -> str:
    """Targeted deterministic response when LLM is offline or encounters an error."""
    q = query.lower()
    name = patient_data.get("name", "").split(" ")[0] if patient_data.get("name") else "there"
    labs = patient_data.get("lab_results") or {}
    vitals = patient_data.get("vitals") or {}
    conditions = [c.lower() for c in patient_data.get("conditions") or []]

    has_hypertension = any("hypertension" in c or "blood pressure" in c for c in conditions) or (vitals.get("systolic_bp", 0) >= 130)
    has_kidney = any("kidney" in c or "ckd" in c or "renal" in c for c in conditions) or (labs.get("egfr", 100) < 60)
    has_diabetes = any("diabetes" in c or "glycemia" in c for c in conditions) or (labs.get("hba1c", 0) >= 6.5)

    if any(w in q for w in ["food", "eat", "diet", "nutrition", "meal", "drink", "avoid"]):
        tips = []
        if has_hypertension or has_kidney:
            tips.append("• **Cut down on sodium (salt)**: Limit packaged meals, canned soups, processed meats, and avoid adding extra salt at the table to protect your blood pressure and kidneys.")
        if has_kidney:
            tips.append("• **Be mindful of potassium & phosphorus**: If your doctor or dietitian recommends it, moderate high-potassium foods (like bananas and potatoes) and high-phosphorus items (like colas and heavy dairy).")
            tips.append("• **Moderate protein**: Choose high-quality, lean proteins in measured portions rather than oversized heavy red meats.")
        if has_diabetes:
            tips.append("• **Choose complex carbohydrates**: Opt for whole grains (oats, brown rice) and non-starchy vegetables over refined carbs (white bread, pastries, sugary drinks).")
        if not tips:
            tips.append("• **Focus on balanced whole foods**: Fresh leafy greens, lean proteins, high-fiber legumes, and plenty of water.")

        return "Here are the most important dietary recommendations for your health:\n\n" + "\n".join(tips) + "\n\n💡 *Tip: It is always best to consult a registered renal/clinical dietitian for a personalized daily meal plan.*"

    if any(w in q for w in ["kidney", "renal", "filtration", "egfr"]):
        egfr = labs.get("egfr", "N/A")
        return (
            f"To keep your kidneys healthy (current filtration rate eGFR: {egfr}):\n\n"
            "1. **Control your blood pressure**: Keeping your pressure well-managed is the single most effective way to slow down kidney strain.\n"
            "2. **Stay appropriately hydrated**: Drink water consistently throughout the day (unless your doctor has placed you on fluid restriction).\n"
            "3. **Avoid NSAID painkillers**: Over-the-counter pain relievers like ibuprofen and naproxen can stress your kidneys; ask your pharmacist or doctor before taking them.\n"
            "4. **Take prescribed kidney-protective medications regularly** as guided by your care team."
        )

    if any(w in q for w in ["pressure", "bp", "hypertension", "heart"]):
        sbp = vitals.get("systolic_bp", "N/A")
        dbp = vitals.get("diastolic_bp", "N/A")
        return (
            f"To manage your blood pressure (recently recorded at {sbp}/{dbp} mmHg):\n\n"
            "1. **Low Sodium Intake**: Keep salt under 2,000 mg/day (about 1 teaspoon total across all meals).\n"
            "2. **Daily Routine**: Take your blood pressure medications at the same time every morning or evening without skipping doses.\n"
            "3. **Gentle Physical Activity**: Aim for 20-30 minutes of doctor-approved walking or light aerobic exercise on most days.\n"
            "4. **Stress & Sleep**: Prioritize 7-8 hours of restful sleep and mindfulness techniques."
        )

    if any(w in q for w in ["medication", "medicine", "pill", "drug", "dose"]):
        meds = patient_data.get("current_medications") or []
        meds_list = ", ".join(meds) if meds else "No current medications on file"
        return (
            f"You are currently taking: **{meds_list}**.\n\n"
            "**Key Safety Principles:**\n"
            "• Take every dose consistently as prescribed—do not stop suddenly without discussing with your doctor.\n"
            "• Keep an updated list of all medications, including vitamins and supplements, to share during clinic visits.\n"
            "• Check with your doctor or pharmacist before starting any new over-the-counter medication."
        )

    if any(w in q for w in ["lab", "result", "test", "hba1c", "creatinine"]):
        egfr = labs.get("egfr", "N/A")
        hba1c = labs.get("hba1c", "N/A")
        sbp = vitals.get("systolic_bp", "N/A")
        dbp = vitals.get("diastolic_bp", "N/A")
        return (
            f"Here is a summary of your recent key measurements:\n\n"
            f"• **Blood Pressure**: {sbp}/{dbp} mmHg\n"
            f"• **Kidney Filtration (eGFR)**: {egfr} (measures how well kidneys filter waste)\n"
            f"• **Average 3-Month Blood Sugar (HbA1c)**: {hba1c}%\n\n"
            "Feel free to ask about any specific number you'd like more details on!"
        )

    return (
        "I'm here to help with your health questions. You can ask me about:\n"
        "• Specific foods and nutrition recommendations\n"
        "• How your medications work and safe practices\n"
        "• Ways to protect your kidney health and manage blood pressure\n"
        "• Explanations of your latest lab results"
    )


@router.post(
    "/chat",
    response_model=APIResponse[ChatResponse],
    status_code=status.HTTP_200_OK,
    tags=["Chat"],
    summary="Patient-facing conversational assistant endpoint",
)
async def patient_chat(request: ChatRequest) -> APIResponse[ChatResponse]:
    """Provide empathetic, multi-turn, direct conversational replies to patient questions."""
    patient_data: Dict[str, Any] = {}
    patient_id = request.patient_id

    # 1. Resolve patient profile
    if request.patient and isinstance(request.patient, dict):
        patient_data = dict(request.patient)
        if not patient_id:
            patient_id = str(patient_data.get("patient_id", ""))

    # Find matching profile in synthetic registry if available
    matched_profile = find_sample_patient(patient_id) if patient_id else None
    if not matched_profile and patient_data.get("name"):
        matched_profile = find_sample_patient(patient_data["name"])

    if matched_profile:
        sample_dict = matched_profile.model_dump()
        # Merge sample data to enrich any missing fields (e.g. labs, vitals, medications)
        if not patient_data or not patient_data.get("lab_results") or not patient_data.get("vitals"):
            patient_data = {**sample_dict, **patient_data}
            if not patient_data.get("lab_results") and sample_dict.get("lab_results"):
                patient_data["lab_results"] = sample_dict["lab_results"]
            if not patient_data.get("vitals") and sample_dict.get("vitals"):
                patient_data["vitals"] = sample_dict["vitals"]
            if not patient_data.get("current_medications") and sample_dict.get("current_medications"):
                patient_data["current_medications"] = sample_dict["current_medications"]
            if not patient_data.get("conditions") and sample_dict.get("conditions"):
                patient_data["conditions"] = sample_dict["conditions"]
            if not patient_data.get("name") and sample_dict.get("name"):
                patient_data["name"] = sample_dict["name"]
        if not patient_id:
            patient_id = matched_profile.patient_id
    elif not patient_data and patient_id:
        patient_data = {"patient_id": patient_id}

    patient_name = patient_data.get("name", "there").split(" ")[0]
    patient_context = _build_patient_context_string(patient_data)

    system_prompt = f"""\
You are a warm, empathetic, and knowledgeable Personal Health Companion speaking directly with the patient ({patient_name}).

PATIENT CLINICAL CONTEXT:
======================================================================
{patient_context}
======================================================================

STRICT CONVERSATIONAL RULES:
1. FOCUS ONLY ON THE LATEST QUESTION: Answer strictly and directly the patient's newest question. Do NOT re-answer questions from previous conversation turns.
2. NO REPETITIVE GREETINGS OR FULL MEDICAL SUMMARIES:
   - Do NOT start your message with a repetitive greeting (like "Hi [Name], let's talk about..." or "I'm here to help explain..."). Jump straight into answering their question warmly and helpfully.
   - Do NOT recite their full medical background (e.g., blood pressure, kidney stage, blood sugar) on every turn unless they specifically ask for an overview of all their test results.
3. CONVERSATIONAL & JARGON-FREE:
   - Use simple, everyday terms (e.g., say "kidney filtration rate" rather than confusing formulas, "average 3-month blood sugar" for HbA1c).
   - Use clean bullet points, bold text, and concise paragraphs for high readability.
4. EVIDENCE-BASED & SAFE:
   - Tailor lifestyle, dietary, and medication safety tips specifically to their conditions (e.g. low-sodium for high blood pressure, kidney-safe choices if kidney function is low, complex carbs for blood sugar).
   - Always encourage consulting their doctor or care team before changing medications.
"""

    # Build LangChain message chain with multi-turn history
    llm_messages = [SystemMessage(content=system_prompt)]

    # Include recent conversation turns (up to 6 previous messages)
    recent_history = request.history[-6:] if request.history else []
    for item in recent_history:
        if item.role in ("patient", "user"):
            llm_messages.append(HumanMessage(content=item.text))
        elif item.role == "assistant":
            llm_messages.append(AIMessage(content=item.text))

    # Add the current user query
    llm_messages.append(HumanMessage(content=request.message))

    # Invoke LLM
    try:
        factory = get_llm_factory()
        if factory.is_configured():
            llm = factory.get_model(temperature=0.3)
            logger.info("Generating conversational patient reply using %s", factory.settings.llm_provider)
            ai_response = await llm.ainvoke(llm_messages)
            
            # Robust content text extraction across LangChain ChatModel variants
            raw_content = getattr(ai_response, "content", "")
            if isinstance(raw_content, str):
                reply_text = raw_content
            elif isinstance(raw_content, list):
                parts = []
                for part in raw_content:
                    if isinstance(part, dict) and "text" in part:
                        parts.append(str(part["text"]))
                    elif isinstance(part, str):
                        parts.append(part)
                    else:
                        parts.append(str(part))
                reply_text = "".join(parts)
            else:
                reply_text = str(raw_content)

            return APIResponse(
                success=True,
                data=ChatResponse(reply=reply_text, patient_id=patient_id, category="advice"),
            )
        else:
            logger.warning("LLM unconfigured, generating targeted deterministic fallback.")
            reply_text = _generate_fallback_response(request.message, patient_data)
            return APIResponse(
                success=True,
                data=ChatResponse(reply=reply_text, patient_id=patient_id, category="advice"),
            )
    except Exception as exc:
        logger.error("LLM patient chat invocation failed: %s — falling back to deterministic response", str(exc))
        reply_text = _generate_fallback_response(request.message, patient_data)
        return APIResponse(
            success=True,
            data=ChatResponse(reply=reply_text, patient_id=patient_id, category="advice"),
        )
