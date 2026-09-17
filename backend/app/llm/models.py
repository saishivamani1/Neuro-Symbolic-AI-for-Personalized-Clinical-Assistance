"""
app/llm/models.py

LangChain Chat Model factory and configuration management.

Provides a unified interface for instantiating LangChain chat models (ChatOpenAI)
with deterministic settings, graceful fallback when unconfigured, and health probing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from app.core.config import Settings, get_settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger

logger = get_logger(__name__)


class LLMFactory:
    """Factory for LangChain chat models with validation and health checks."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def is_configured(self) -> bool:
        """Check if LLM credentials are provided."""
        if self.settings.llm_provider == "openai":
            return bool(self.settings.openai_api_key and self.settings.openai_api_key.strip())
        if self.settings.llm_provider in ("gemini", "google"):
            api_key = self.settings.google_api_key or self.settings.gemini_api_key or self.settings.openai_api_key
            return bool(api_key and api_key.strip())
        if self.settings.llm_provider == "groq":
            api_key = self.settings.groq_api_key or self.settings.openai_api_key
            return bool(api_key and api_key.strip())
        if self.settings.llm_provider == "ollama":
            return bool(self.settings.ollama_base_url)
        if self.settings.llm_provider == "huggingface":
            return bool(self.settings.hf_api_key and self.settings.hf_api_key.strip())
        return False

    def get_model(
        self,
        temperature: Optional[float] = None,
        streaming: bool = False,
    ) -> Any:
        """Create and return a configured LangChain ChatModel instance.

        Raises LLMError if the provider is unconfigured or required packages are missing.
        """
        temp = temperature if temperature is not None else self.settings.llm_temperature

        if self.settings.llm_provider in ("gemini", "google"):
            if not self.is_configured():
                raise LLMError(
                    detail="Google Gemini API key is not configured. Please set GOOGLE_API_KEY (or GEMINI_API_KEY) in .env",
                    context={"provider": self.settings.llm_provider, "model": self.settings.gemini_model},
                )
            api_key = self.settings.google_api_key or self.settings.gemini_api_key or self.settings.openai_api_key
            model_name = self.settings.gemini_model or "gemini-2.0-flash"
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI

                logger.info("Initializing Google Gemini Chat model: %s", model_name)
                return ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key,
                    temperature=temp,
                    timeout=self.settings.llm_timeout,
                    max_retries=self.settings.llm_max_retries,
                    streaming=streaming,
                )
            except ImportError as exc:
                logger.error("langchain-google-genai package is not installed: %s", str(exc))
                raise LLMError(
                    detail="langchain-google-genai package is required for Google Gemini models.",
                    context={"error": str(exc)},
                ) from exc
            except Exception as exc:
                logger.error("Failed to initialize Google Gemini model (%s): %s", model_name, str(exc))
                raise LLMError(
                    detail=f"Failed to instantiate Google Gemini model '{model_name}': {exc}",
                    context={"provider": self.settings.llm_provider, "model": model_name, "error": str(exc)},
                ) from exc

        if self.settings.llm_provider == "openai":
            if not self.is_configured():
                raise LLMError(
                    detail="OpenAI API key is not configured. Please set OPENAI_API_KEY in .env",
                    context={"provider": self.settings.llm_provider, "model": self.settings.llm_model},
                )
            try:
                from langchain_openai import ChatOpenAI

                return ChatOpenAI(
                    model=self.settings.llm_model,
                    api_key=self.settings.openai_api_key,
                    base_url=self.settings.openai_api_base if self.settings.openai_api_base != "https://api.openai.com/v1" else None,
                    temperature=temp,
                    timeout=self.settings.llm_timeout,
                    max_retries=self.settings.llm_max_retries,
                    streaming=streaming,
                )
            except ImportError as exc:
                logger.error("langchain-openai package is not installed: %s", str(exc))
                raise LLMError(
                    detail="langchain-openai package is required for OpenAI models.",
                    context={"error": str(exc)},
                ) from exc
            except Exception as exc:
                logger.error("Failed to initialize ChatOpenAI: %s", str(exc))
                raise LLMError(
                    detail=f"Failed to instantiate OpenAI chat model: {exc}",
                    context={"error": str(exc)},
                ) from exc

        if self.settings.llm_provider == "groq":
            if not self.is_configured():
                raise LLMError(
                    detail="Groq API key is not configured. Please set GROQ_API_KEY in .env",
                    context={"provider": self.settings.llm_provider, "model": self.settings.groq_model},
                )

            api_key = self.settings.groq_api_key or self.settings.openai_api_key
            # Always use groq_model when provider is groq — llm_model is for OpenAI-family providers
            model_name = self.settings.groq_model or "llama-3.3-70b-versatile"

            try:
                # 1. Primary: langchain-groq ChatGroq (preferred — uses native Groq SDK)
                from langchain_groq import ChatGroq

                logger.info("Initializing Groq ChatGroq model: %s", model_name)
                return ChatGroq(
                    model=model_name,
                    groq_api_key=api_key,
                    temperature=temp,
                    max_tokens=self.settings.llm_max_tokens,
                    timeout=self.settings.llm_timeout,
                    max_retries=self.settings.llm_max_retries,
                    streaming=streaming,
                )
            except ImportError:
                # 2. Fallback: langchain-openai ChatOpenAI via Groq's OpenAI-compatible endpoint
                logger.warning("langchain-groq not installed — falling back to ChatOpenAI with Groq endpoint")
                from langchain_openai import ChatOpenAI

                return ChatOpenAI(
                    model=model_name,
                    api_key=api_key,
                    base_url="https://api.groq.com/openai/v1",
                    temperature=temp,
                    max_tokens=self.settings.llm_max_tokens,
                    timeout=self.settings.llm_timeout,
                    max_retries=self.settings.llm_max_retries,
                    streaming=streaming,
                )
            except Exception as exc:
                logger.error("Failed to initialize Groq model (%s): %s", model_name, str(exc))
                raise LLMError(
                    detail=f"Failed to instantiate Groq model '{model_name}': {exc}",
                    context={"provider": "groq", "model": model_name, "error": str(exc)},
                ) from exc

        if self.settings.llm_provider == "huggingface":
            if not self.is_configured():
                raise LLMError(
                    detail="HuggingFace API token not configured. Set HF_API_KEY in .env",
                    context={"provider": "huggingface", "model": self.settings.hf_model},
                )
            model_name = self.settings.hf_model or "meta-llama/Llama-3.1-8B-Instruct"
            try:
                from langchain_openai import ChatOpenAI

                logger.info("Initializing HuggingFace Inference API model: %s", model_name)
                # HuggingFace exposes an OpenAI-compatible endpoint at /v1
                return ChatOpenAI(
                    model=model_name,
                    api_key=self.settings.hf_api_key,
                    base_url="https://router.huggingface.co/v1",
                    temperature=temp,
                    timeout=self.settings.llm_timeout,
                    max_retries=self.settings.llm_max_retries,
                    streaming=streaming,    
                )
            except ImportError as exc:
                raise LLMError(
                    detail="langchain-openai package is required for HuggingFace provider.",
                    context={"error": str(exc)},
                ) from exc
            except Exception as exc:
                logger.error("Failed to initialize HuggingFace model (%s): %s", model_name, str(exc))
                raise LLMError(
                    detail=f"Failed to instantiate HuggingFace model '{model_name}': {exc}",
                    context={"provider": "huggingface", "model": model_name, "error": str(exc)},
                ) from exc

        raise LLMError(
            detail=f"Unsupported LLM provider: '{self.settings.llm_provider}'",
            context={"provider": self.settings.llm_provider},
        )

    def health_check(self) -> Dict[str, Any]:
        """Probe LLM configuration status without making costly API calls."""
        if not self.is_configured():
            key_map = {
                "groq": "GROQ_API_KEY",
                "huggingface": "HF_API_KEY",
                "gemini": "GOOGLE_API_KEY",
                "google": "GOOGLE_API_KEY",
            }
            key_name = key_map.get(self.settings.llm_provider, "OPENAI_API_KEY")
            return {
                "status": "unconfigured",
                "provider": self.settings.llm_provider,
                "model": self._resolve_model_name(),
                "detail": f"{key_name} is not set in environment or .env file",
            }

        return {
            "status": "configured",
            "provider": self.settings.llm_provider,
            "model": self._resolve_model_name(),
            "temperature": self.settings.llm_temperature,
        }

    def _resolve_model_name(self) -> str:
        """Return the effective model name for the configured provider."""
        if self.settings.llm_provider in ("gemini", "google"):
            return self.settings.gemini_model or "gemini-2.0-flash"
        if self.settings.llm_provider == "groq":
            return self.settings.groq_model or "llama-3.3-70b-versatile"
        if self.settings.llm_provider == "ollama":
            return self.settings.ollama_model
        if self.settings.llm_provider == "huggingface":
            return self.settings.hf_model or "meta-llama/Llama-3.1-8B-Instruct"
        return self.settings.llm_model


_cached_llm_factory: Optional[LLMFactory] = None


def get_llm_factory() -> LLMFactory:
    """Singleton getter for LLMFactory."""
    global _cached_llm_factory
    if _cached_llm_factory is None:
        _cached_llm_factory = LLMFactory()
    return _cached_llm_factory
