"""
app/core/config.py

Centralised application configuration loaded from environment variables
via Pydantic Settings.  A single Settings instance is constructed once
and re-used throughout the application through the `get_settings`
dependency / singleton.

All secrets (API keys, passwords) MUST live in a .env file and are
never hard-coded here.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Application
    # ------------------------------------------------------------------ #
    app_name: str = Field(default="Neuro-Symbolic Healthcare Intelligence Platform")
    app_version: str = Field(default="0.1.0")
    app_env: Literal["development", "staging", "production"] = Field(
        default="development"
    )
    debug: bool = Field(default=False)
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # ------------------------------------------------------------------ #
    # LLM
    # ------------------------------------------------------------------ #
    llm_provider: Literal["openai", "ollama", "azure_openai", "groq", "huggingface", "gemini", "google"] = Field(
        default="openai"
    )
    llm_model: str = Field(default="gpt-4o-mini")
    llm_temperature: float = Field(
        default=0.1, description="Low temperature for deterministic clinical assistance."
    )
    llm_max_tokens: int = Field(
        default=4096, description="Maximum output tokens for LLM generation."
    )
    llm_timeout: float = Field(default=60.0, description="Request timeout in seconds.")
    llm_max_retries: int = Field(default=2, description="Max retries for LLM requests.")
    openai_api_key: str = Field(default="")
    openai_api_base: str = Field(default="https://api.openai.com/v1")
    # Gemini / Google settings
    google_api_key: str = Field(
        default="",
        description="Google Gemini API key from AI Studio.",
    )
    gemini_api_key: str = Field(
        default="",
        description="Alias for Google Gemini API key.",
    )
    gemini_model: str = Field(
        default="gemini-2.0-flash",
        description="Google Gemini model name (e.g. gemini-2.5-flash, gemini-2.0-flash, gemini-1.5-flash).",
    )
    # Groq settings (Llama & Mistral models)
    groq_api_key: str = Field(default="")
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Groq model ID (e.g., llama-3.3-70b-versatile, mixtral-8x7b-32768, mistral-saba-24b, llama-3.1-8b-instant).",
    )
    # Ollama / local Llama settings
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3")
    # HuggingFace Inference API settings
    hf_api_key: str = Field(
        default="",
        description="HuggingFace API token (from https://huggingface.co/settings/tokens).",
    )
    hf_model: str = Field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        description="HuggingFace model ID for serverless inference (must support tool calling).",
    )

    # ------------------------------------------------------------------ #
    # Embeddings
    # ------------------------------------------------------------------ #
    embedding_model: str = Field(
        default="BAAI/bge-large-en-v1.5",
        description="HuggingFace model name for BGE embeddings.",
    )
    embedding_device: str = Field(
        default="cpu",
        description="Device for sentence-transformers: 'cpu' or 'cuda'.",
    )

    # ------------------------------------------------------------------ #
    # ChromaDB
    # ------------------------------------------------------------------ #
    chroma_persist_directory: str = Field(default="./data/chroma")
    chroma_collection_name: str = Field(default="medical_documents")

    # ------------------------------------------------------------------ #
    # Neo4j
    # ------------------------------------------------------------------ #
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_username: str = Field(default="neo4j")
    neo4j_password: str = Field(default="password")
    neo4j_database: str = Field(default="neo4j")

    # ------------------------------------------------------------------ #
    # RAG / Retrieval
    # ------------------------------------------------------------------ #
    top_k: int = Field(default=5, description="Number of chunks to retrieve.")
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    log_level: str = Field(default="INFO")
    log_format: Literal["json", "text"] = Field(default="json")

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #
    data_directory: str = Field(default="./data")
    documents_directory: str = Field(default="./data/documents")
    sample_directory: str = Field(default="./data/sample")
    rules_config_path: str = Field(default="./data/rules.json")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton.

    Using lru_cache(maxsize=1) ensures the .env file is read only once
    per process lifetime, which is the desired behaviour for a long-running
    FastAPI application.
    """
    return Settings()
