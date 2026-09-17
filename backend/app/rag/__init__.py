"""
app/rag/__init__.py

RAG (Retrieval-Augmented Generation) package for clinical guidelines and medical documents.

Exports:
- PDFLoader: PyMuPDF PDF parser and text extractor
- MedicalChunker: Page-aware recursive text splitter
- BGEEmbeddings: Dense vector embeddings wrapper (BGE-large)
- ChromaVectorStore: Persistent ChromaDB collection manager
- RAGRetriever: Semantic retrieval service
- DocumentPipeline: End-to-end PDF ingestion pipeline
"""

from app.rag.chunking import MedicalChunker
from app.rag.embeddings import BGEEmbeddings, get_embeddings_service
from app.rag.loaders import PDFLoader
from app.rag.pipeline import DocumentPipeline, get_document_pipeline
from app.rag.retriever import RAGRetriever, get_retriever
from app.rag.vector_store import ChromaVectorStore, get_vector_store

__all__ = [
    "PDFLoader",
    "MedicalChunker",
    "BGEEmbeddings",
    "ChromaVectorStore",
    "RAGRetriever",
    "DocumentPipeline",
    "get_embeddings_service",
    "get_vector_store",
    "get_retriever",
    "get_document_pipeline",
]
