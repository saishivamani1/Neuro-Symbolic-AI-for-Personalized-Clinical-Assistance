
# backend/tet.py - Test configured Groq LLM model

import asyncio
from app.core.config import get_settings
from app.llm.models import get_llm_factory


def main() -> None:
    settings = get_settings()
    print(f"Current Provider : {settings.llm_provider}")
    print(f"Configured Model : {settings.groq_model}")

    factory = get_llm_factory()
    print(f"Is Configured    : {factory.is_configured()}")

    if not factory.is_configured():
        print("ERROR: Groq API key is not configured in backend/.env.")
        return

    from app.llm.schemas import ClinicalAssistantResponse
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    
    for model_name in ["qwen/qwen3.8-27b", "groq/compound"]:
        print(f"\n================ Testing {model_name} with with_structured_output ================")
        try:
            llm = ChatGroq(
                model=model_name,
                groq_api_key=settings.groq_api_key,
                temperature=0.1,
                max_tokens=4096,
            )
            structured = llm.with_structured_output(ClinicalAssistantResponse)
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a clinical decision support assistant. You must synthesize the patient data into structured format."),
                ("human", "Patient 68yo male with Stage 4 CKD (eGFR 24), Type 2 Diabetes (HbA1c 8.4%), Metformin 1000mg BID. Evaluate safety.")
            ])
            
            res = structured.invoke(prompt.format_messages())
            print(f"SUCCESS on {model_name}!")
            print("Summary:", str(res.summary)[:150])
            print("Risk flags:", len(res.risk_flags))
        except Exception as e:
            print(f"FAILED on {model_name}: {str(e)[:200]}")


if __name__ == "__main__":
    main()