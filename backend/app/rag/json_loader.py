"""
app/rag/json_loader.py

Loader for structured medical disease JSON datasets.

Expected input:
[
    {
        "disease": "...",
        "Overview": "...",
        "Symptoms": "...",
        "Causes": "...",
        ...
    }
]

Creates section-level DocumentChunk objects suitable
for the existing embedding and ChromaDB pipeline.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.schemas.rag import DocumentChunk


KNOWLEDGE_FIELDS = [
    "Overview",
    "Symptoms",
    "When to see a doctor",
    "Causes",
    "Risk factors",
    "Complications",
    "Prevention",
    "Diagnosis",
    "Treatment",
    "Coping and support",
    "Preparing for your appointment",
    "Lifestyle and home remedies",
]


def clean_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).strip()

    if not text:
        return ""

    text = text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    # Fix common scraping artefacts
    replacements = {
        "AFibcan": "AFib can",
        "AFibmay": "AFib may",
        "AFiband": "AFib and",
        "AFibis": "AFib is",
        "AFibisn't": "AFib isn't",
        "withAFib": "with AFib",
        "forAFib": "for AFib",
        "ofAFib": "of AFib",
        "HIVis": "HIV is",
        "HIVcan": "HIV can",
        "HIVand": "HIV and",
        "withHIV": "with HIV",
        "forHIV": "for HIV",
        "medicalattention": "medical attention",
        "healthcareprofessional":
            "healthcare professional",
        "healthcareprovider":
            "healthcare provider",
        "healthcareteam":
            "healthcare team",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new,
        )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    text = re.sub(
        r"[ \t]+\n",
        "\n",
        text,
    )

    return text.strip()


def slugify(value: str) -> str:
    value = value.lower()

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value,
    )

    return value.strip("_")


def calculate_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class JSONDiseaseLoader:
    """Loads structured disease JSON into RAG chunks."""

    def load(
        self,
        file_path: str | Path,
    ) -> Tuple[
        str,
        str,
        List[DocumentChunk],
    ]:

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"JSON file not found: {path}"
            )

        data = path.read_bytes()

        content_hash = calculate_hash(data)

        try:
            records = json.loads(
                data.decode("utf-8")
            )

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON file '{path}': {exc}"
            ) from exc

        if not isinstance(records, list):
            raise ValueError(
                "Medical disease JSON must contain "
                "a top-level list."
            )

        document_id = (
            f"json_{content_hash[:24]}"
        )

        chunks = []

        chunk_index = 0

        for record_index, record in enumerate(
            records
        ):

            if not isinstance(record, dict):
                continue

            disease = clean_text(
                record.get("disease")
            )

            if not disease:
                continue

            disease_key = (
                record.get("id")
                or record.get("Unnamed: 0")
                or record_index
            )

            disease_key = slugify(
                str(disease_key)
            )

            keywords = record.get(
                "keywords",
                record.get(
                    "updated",
                    "",
                ),
            )

            if isinstance(
                keywords,
                list,
            ):
                keywords_text = ", ".join(
                    clean_text(item)
                    for item in keywords
                    if clean_text(item)
                )
            else:
                keywords_text = clean_text(
                    keywords
                )

            source = clean_text(
                record.get(
                    "source",
                    "Mayo Clinic",
                )
            )

            source_urls = []

            for field in [
                "main_link",
                "Diagnosis_treatment_link",
                "Doctors_departments_link",
            ]:

                url = record.get(field)

                if url:
                    source_urls.append(
                        str(url).strip()
                    )

            for section in KNOWLEDGE_FIELDS:

                text = clean_text(
                    record.get(section)
                )

                if not text:
                    continue

                section_slug = slugify(
                    section
                )

                chunk_id = (
                    f"{document_id}_"
                    f"{disease_key}_"
                    f"{section_slug}"
                )

                chunk_text = (
                    f"Disease: {disease}\n"
                    f"Section: {section}\n"
                    f"Source: {source}\n\n"
                    f"{text}"
                )

                metadata = {
                    "content_hash":
                        content_hash,
                    "document_type":
                        "medical_disease_json",
                    "source":
                        source,
                    "disease":
                        disease,
                    "section":
                        section,
                }

                if keywords_text:
                    metadata[
                        "keywords"
                    ] = keywords_text

                if source_urls:
                    metadata[
                        "source_urls"
                    ] = " | ".join(
                        source_urls
                    )

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    filename=path.name,
                    page_number=record_index + 1,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    metadata=metadata,
                )

                chunks.append(chunk)

                chunk_index += 1

        if not chunks:
            raise ValueError(
                f"No usable medical records found "
                f"in '{path}'."
            )

        return (
            document_id,
            content_hash,
            chunks,
        )