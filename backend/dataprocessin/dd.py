
import json
import re
from pathlib import Path
from urllib.parse import urlparse

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("medical_diseases.json")

CLEANED_JSON_FILE = Path("cleaned_diseases.json")
CHUNKS_JSONL_FILE = Path("disease_chunks.jsonl")

SOURCE_NAME = "Mayo Clinic"

# Fields that contain medical knowledge and should become
# individual RAG sections.
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


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean formatting problems without changing the medical meaning.
    """

    if text is None:
        return None

    if not isinstance(text, str):
        text = str(text)

    text = text.strip()

    if not text:
        return None

    # Normalize Windows / Mac line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Fix common missing spaces caused by scraping.
    # Example:
    # AFibcan -> AFib can
    # conditionis -> condition is
    # medicalattention -> medical attention

    text = re.sub(
        r'([a-zA-Z])([A-Z][a-z]+)',
        r'\1 \2',
        text
    )

    # Common medical abbreviations that frequently get attached
    # to surrounding words.
    replacements = {
        "AFibcan": "AFib can",
        "AFibmay": "AFib may",
        "AFibitself": "AFib itself",
        "AFiband": "AFib and",
        "AFibis": "AFib is",
        "AFibisn't": "AFib isn't",
        "withAFib": "with AFib",
        "ofAFib": "of AFib",
        "forAFib": "for AFib",
        "HIVis": "HIV is",
        "HIVcan": "HIV can",
        "HIVand": "HIV and",
        "withHIV": "with HIV",
        "forHIV": "for HIV",
        "COVID-19virus": "COVID-19 virus",
        "medicalattention": "medical attention",
        "healthcareprofessional": "healthcare professional",
        "healthcareprovider": "healthcare provider",
        "healthcareteam": "healthcare team",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove spaces before punctuation
    text = re.sub(r'\s+([,.;:!?])', r'\1', text)

    # Normalize multiple spaces while preserving newlines
    text = re.sub(r'[ \t]+', ' ', text)

    # Normalize excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove whitespace around line breaks
    text = re.sub(r'[ \t]+\n', '\n', text)
    text = re.sub(r'\n[ \t]+', '\n', text)

    return text.strip()


# ============================================================
# CLEAN DISEASE NAME
# ============================================================

def clean_disease_name(name):
    if name is None:
        return None

    name = str(name).strip()

    name = re.sub(r'\s+', ' ', name)

    return name


# ============================================================
# CLEAN URL
# ============================================================

def clean_url(url):
    if not url:
        return None

    url = str(url).strip()

    try:
        parsed = urlparse(url)

        if parsed.scheme in ("http", "https") and parsed.netloc:
            return url

    except Exception:
        pass

    return None


# ============================================================
# CLEAN KEYWORDS
# ============================================================

def clean_keywords(value):
    if not value:
        return []

    if isinstance(value, list):
        keywords = value
    else:
        keywords = str(value).split(",")

    cleaned = []

    for keyword in keywords:
        keyword = clean_text(keyword)

        if keyword:
            keyword = keyword.lower()

            if keyword not in cleaned:
                cleaned.append(keyword)

    return cleaned


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            "Expected the JSON file to contain a list of disease records."
        )

    return data


# ============================================================
# CLEAN ONE DISEASE
# ============================================================

def clean_disease(record, disease_id):
    disease = clean_disease_name(record.get("disease"))

    if not disease:
        return None

    cleaned = {
        "id": disease_id,
        "disease": disease,
        "source": SOURCE_NAME,

        "source_urls": {
            "main": clean_url(record.get("main_link")),
            "diagnosis_treatment": clean_url(
                record.get("Diagnosis_treatment_link")
            ),
            "doctors_departments": clean_url(
                record.get("Doctors_departments_link")
            ),
        },

        "keywords": clean_keywords(record.get("updated")),
    }

    # Add medical knowledge fields
    for field in KNOWLEDGE_FIELDS:
        value = clean_text(record.get(field))

        cleaned[field] = value

    return cleaned


# ============================================================
# REMOVE NULL SOURCE URLS
# ============================================================

def remove_empty_urls(record):
    urls = record.get("source_urls", {})

    urls = {
        key: value
        for key, value in urls.items()
        if value
    }

    record["source_urls"] = urls

    return record


# ============================================================
# CREATE RAG CHUNKS
# ============================================================

def create_chunks(disease):
    """
    Creates one RAG chunk per medical section.
    """

    chunks = []

    disease_id = disease["id"]
    disease_name = disease["disease"]

    for field in KNOWLEDGE_FIELDS:

        content = disease.get(field)

        if not content:
            continue

        section_name = field

        chunk_text = (
            f"Disease: {disease_name}\n"
            f"Section: {section_name}\n\n"
            f"{content}"
        )

        chunk = {
            "chunk_id": f"{disease_id}_{slugify(section_name)}",
            "disease_id": disease_id,
            "disease": disease_name,
            "section": section_name,
            "source": disease["source"],
            "text": chunk_text,
            "keywords": disease.get("keywords", []),
            "source_urls": disease.get("source_urls", {}),
        }

        chunks.append(chunk)

    return chunks


# ============================================================
# SLUGIFY
# ============================================================

def slugify(text):
    text = text.lower()

    text = re.sub(
        r'[^a-z0-9]+',
        '_',
        text
    )

    return text.strip("_")


# ============================================================
# DUPLICATE DETECTION
# ============================================================

def remove_duplicate_diseases(records):
    seen = set()
    unique = []

    for record in records:

        key = record["disease"].strip().lower()

        if key in seen:
            continue

        seen.add(key)
        unique.append(record)

    return unique


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(records):
    problems = []

    for record in records:

        if not record.get("disease"):
            problems.append(
                f"Missing disease name: {record.get('id')}"
            )

        knowledge_count = sum(
            1
            for field in KNOWLEDGE_FIELDS
            if record.get(field)
        )

        if knowledge_count == 0:
            problems.append(
                f"No medical content: {record.get('disease')}"
            )

    return problems


# ============================================================
# SAVE JSON
# ============================================================

def save_json(data, path):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


# ============================================================
# SAVE JSONL
# ============================================================

def save_jsonl(chunks, path):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        for chunk in chunks:

            f.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False
                )
                + "\n"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("MEDICAL DATASET CLEANER")
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print("\nLoading dataset...")

    raw_records = load_dataset(INPUT_FILE)

    print(
        f"Loaded {len(raw_records)} raw records."
    )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    cleaned_records = []

    for index, record in enumerate(raw_records):

        disease_id = f"disease_{index:04d}"

        cleaned = clean_disease(
            record,
            disease_id
        )

        if cleaned:
            cleaned = remove_empty_urls(cleaned)

            cleaned_records.append(cleaned)

    print(
        f"Cleaned {len(cleaned_records)} records."
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    before = len(cleaned_records)

    cleaned_records = remove_duplicate_diseases(
        cleaned_records
    )

    after = len(cleaned_records)

    print(
        f"Removed {before - after} duplicate diseases."
    )

    # --------------------------------------------------------
    # Reassign IDs after duplicate removal
    # --------------------------------------------------------

    for index, record in enumerate(cleaned_records):

        record["id"] = f"disease_{index:04d}"

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    problems = validate_dataset(
        cleaned_records
    )

    if problems:

        print("\nValidation warnings:")

        for problem in problems[:20]:
            print(
                " -",
                problem
            )

        if len(problems) > 20:
            print(
                f" ... and {len(problems) - 20} more"
            )

    else:

        print(
            "\nValidation passed."
        )

    # --------------------------------------------------------
    # Save cleaned disease-level JSON
    # --------------------------------------------------------

    save_json(
        cleaned_records,
        CLEANED_JSON_FILE
    )

    print(
        f"\nCreated: {CLEANED_JSON_FILE}"
    )

    # --------------------------------------------------------
    # Create RAG chunks
    # --------------------------------------------------------

    all_chunks = []

    for disease in cleaned_records:

        chunks = create_chunks(
            disease
        )

        all_chunks.extend(chunks)

    # --------------------------------------------------------
    # Save JSONL
    # --------------------------------------------------------

    save_jsonl(
        all_chunks,
        CHUNKS_JSONL_FILE
    )

    print(
        f"Created: {CHUNKS_JSONL_FILE}"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    print(
        f"Diseases       : {len(cleaned_records)}"
    )

    print(
        f"RAG chunks     : {len(all_chunks)}"
    )

    print(
        f"Average chunks : "
        f"{len(all_chunks) / max(len(cleaned_records), 1):.2f}"
    )

    # Section statistics

    section_counts = {}

    for chunk in all_chunks:

        section = chunk["section"]

        section_counts[section] = (
            section_counts.get(section, 0) + 1
        )

    print("\nChunks by section:")

    for section, count in sorted(
        section_counts.items()
    ):

        print(
            f"  {section:<35} {count}"
        )

    print("\nDone.")


if __name__ == "__main__":
    main()

