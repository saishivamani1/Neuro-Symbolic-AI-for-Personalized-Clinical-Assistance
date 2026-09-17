"""
app/utils/text.py

Text normalisation and cleaning utilities shared across the platform.

These helpers are intentionally free of external dependencies so that
they can be used in any layer (RAG, reasoning, KG extraction) without
introducing circular imports.
"""

from __future__ import annotations

import re
import unicodedata


def normalise_text(text: str) -> str:
    """Normalise unicode, collapse whitespace, and strip leading/trailing space.

    Parameters
    ----------
    text:
        Raw text extracted from a document or user input.

    Returns
    -------
    str
        Cleaned text suitable for embedding or NLP processing.
    """
    # NFKC normalisation: decompose then recompose canonical equivalents.
    text = unicodedata.normalize("NFKC", text)
    # Collapse any run of whitespace (spaces, tabs, newlines) to a single space.
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate_text(text: str, max_chars: int = 2000) -> str:
    """Truncate text to ``max_chars`` characters, appending an ellipsis if cut.

    Parameters
    ----------
    text:
        Input text.
    max_chars:
        Maximum number of characters to retain.

    Returns
    -------
    str
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def extract_sentences(text: str) -> list[str]:
    """Split text into sentences using a simple regex heuristic.

    A proper sentence splitter (spaCy, NLTK) is preferred for production
    NLP; this lightweight version avoids heavy dependencies in Stage 1.

    Parameters
    ----------
    text:
        Input paragraph or document text.

    Returns
    -------
    list[str]
        List of sentence strings, stripped of surrounding whitespace.
    """
    # Split on period/exclamation/question followed by whitespace + capital letter.
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", normalise_text(text))
    return [s.strip() for s in sentences if s.strip()]


def clean_medical_text(text: str) -> str:
    """Apply medical-document-specific cleaning on top of normalise_text.

    Removes common PDF artefacts found in clinical guidelines:
    * Page-number lines ("Page 12 of 45")
    * Running headers / footers (lines shorter than 15 chars)
    * Repeated dashes / underscores used as dividers

    Parameters
    ----------
    text:
        Raw extracted text from a medical PDF.

    Returns
    -------
    str
        Cleaned text.
    """
    lines = text.splitlines()
    cleaned: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Skip empty lines.
        if not stripped:
            continue
        # Skip page-number patterns.
        if re.match(r"^[Pp]age\s+\d+\s+of\s+\d+$", stripped):
            continue
        # Skip divider lines (only dashes/underscores/equals).
        if re.match(r"^[-_=]{3,}$", stripped):
            continue
        # Skip very short lines that are likely headers/footers.
        if len(stripped) < 15 and not re.search(r"\d", stripped):
            continue
        cleaned.append(stripped)
    return normalise_text(" ".join(cleaned))
