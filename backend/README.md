# Neuro-Symbolic Healthcare Intelligence Platform
### Personalized Clinical Assistance Backend

> **IMPORTANT MEDICAL DISCLAIMER**
> This system is a **research prototype** built as an academic mini-project.
> It is **not** a licensed medical device, clinical decision system, or
> substitute for professional medical advice, diagnosis, or treatment.
> Never use it to make real clinical decisions.

---

## 1. Project Overview

This backend implements a **Neuro-Symbolic Healthcare Intelligence Platform**
that combines:

| Component | Technology | Role |
|-----------|-----------|------|
| **RAG Pipeline** | LangChain + ChromaDB + BGE Embeddings | Retrieve relevant clinical evidence from medical documents |
| **Medical Knowledge Graph** | Neo4j | Encode structured medical relationships |
| **Symbolic Rule Engine** | Custom Python engine | Deterministic clinical rule evaluation |
| **LLM** | OpenAI / Ollama | Natural language generation with structured context |
| **FastAPI** | Python 3.11 | REST API layer |

The system is **not** a simple chatbot. It explicitly separates:

- **Neural intelligence** → understanding, retrieval, generation
- **Symbolic intelligence** → explicit rules, graph traversal, auditable reasoning

---

## 2. Architecture

```
                  USER / PATIENT
                       |
                       v
                FastAPI Backend
                       |
                       v
              Clinical NLP Layer
                       |
          +------------+------------+
          |                         |
          v                         v
     RAG Retrieval            Knowledge Graph
     (ChromaDB +              (Neo4j Cypher
      BGE Embeddings)          Traversal)
          |                         |
          |                         v
          |                  Graph Facts
          |                         |
          +------------+------------+
                       |
                       v
              Symbolic Rule Engine
              (Deterministic Rules)
                       |
                       v
               Derived Clinical Facts
                       |
                       v
               Evidence Assembly
                       |
                       v
                    LLM
              (Structured Prompt)
                       |
                       v
       Personalized Explainable Response
                       |
       +---------------+----------------+
       |               |                |
       v               v                v
     Answer         Evidence       Reasoning Trace
```

---

## 3. Technology Stack

- **Python 3.11+**
- **FastAPI 0.115** + **Uvicorn**
- **Pydantic v2** + **pydantic-settings**
- **LangChain** (document loading, chunking, retrieval, LLM orchestration)
- **ChromaDB** (vector store for BGE embeddings)
- **BGE Large English** (`BAAI/bge-large-en-v1.5`) via sentence-transformers
- **PyMuPDF** (PDF parsing)
- **Neo4j 5.x** (knowledge graph)
- **Custom Python symbolic rule engine** (deterministic, auditable)
- **OpenAI API** or **Ollama** (LLM provider)
- **pytest** (testing)
- **Docker Compose** (deployment)

---

## 4. Installation

### Prerequisites

- Python 3.11+
- pip
- Git
- (Optional) Neo4j 5.x running locally or via Docker
- (Optional) OpenAI API key or Ollama installed

### Steps

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd backend

# 2. Create a virtual environment
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# 3. Install Stage 1 dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your values (API keys, Neo4j credentials, etc.)
```

---

## 5. Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Description | Default |
|----------|-------------|---------|
| `APP_ENV` | `development` / `staging` / `production` | `development` |
| `DEBUG` | Enable debug mode | `false` |
| `LLM_PROVIDER` | `openai` / `ollama` | `openai` |
| `LLM_MODEL` | Model name | `gpt-4o-mini` |
| `OPENAI_API_KEY` | OpenAI API key | *(required for OpenAI)* |
| `EMBEDDING_MODEL` | HuggingFace BGE model ID | `BAAI/bge-large-en-v1.5` |
| `EMBEDDING_DEVICE` | `cpu` or `cuda` | `cpu` |
| `CHROMA_PERSIST_DIRECTORY` | ChromaDB storage path | `./data/chroma` |
| `NEO4J_URI` | Neo4j Bolt URI | `bolt://localhost:7687` |
| `NEO4J_USERNAME` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | *(required)* |
| `TOP_K` | RAG retrieval chunks | `5` |
| `CHUNK_SIZE` | Document chunk size (tokens) | `512` |
| `CHUNK_OVERLAP` | Chunk overlap (tokens) | `64` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `LOG_FORMAT` | `json` or `text` | `json` |

**Never commit `.env` to version control.**

---

## 6. Neo4j Setup

### Option A: Docker (recommended)

```bash
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your-password \
  neo4j:5.26-community
```

Access the Neo4j Browser at: http://localhost:7474

### Option B: Neo4j Desktop

1. Download from https://neo4j.com/download/
2. Create a new project and database
3. Set credentials matching your `.env` file

---

## 7. ChromaDB Setup

ChromaDB runs in **embedded persistent mode** by default — no separate
service is needed. The database is stored at `CHROMA_PERSIST_DIRECTORY`
(default: `./data/chroma`).

This directory is created automatically when the first document is ingested.

---

## 8. Document Ingestion

Place medical PDF documents in `./data/documents/`, then run:

```bash
python scripts/ingest_documents.py
```

*(Stage 2 — implementation coming soon)*

---

## 9. Running the Backend

### Development

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at:
- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc
- **Health**: http://localhost:8000/api/v1/health

---

## 10. API Endpoints

| Method | Endpoint | Description | Stage |
|--------|----------|-------------|-------|
| `GET` | `/api/v1/health` | Platform health check | ✅ Stage 1 |
| `POST` | `/api/v1/documents/upload` | Upload medical PDF | 🔜 Stage 2 |
| `POST` | `/api/v1/chat` | Clinical query (full pipeline) | 🔜 Stage 5/6 |
| `GET` | `/api/v1/patients/{id}` | Get patient context | 🔜 Stage 4 |
| `GET` | `/api/v1/knowledge-graph/entities` | List graph entities | 🔜 Stage 3 |

---

## 11. Example Requests

### Health Check

```bash
curl http://localhost:8000/api/v1/health
```

### Clinical Query (Stage 5/6)

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "patient": {
      "patient_id": "P001",
      "age": 58,
      "conditions": ["type2_diabetes", "hypertension"],
      "symptoms": ["fatigue", "increased_thirst", "blurred_vision"],
      "medications": ["metformin"],
      "allergies": ["penicillin"]
    },
    "query": "What clinical considerations are relevant given these symptoms and history?"
  }'
```

---

## 12. Example Response (Stage 5/6)

```json
{
  "answer": "Based on the patient profile and retrieved clinical evidence...",
  "summary": "Patient shows signs consistent with poorly controlled diabetes...",
  "personalization": {
    "factors_considered": ["age_58", "diabetes", "hypertension", "metformin"]
  },
  "evidence": [
    {
      "source": "ADA_Standards_of_Care_2024.pdf",
      "page": 12,
      "content": "HbA1c targets for patients with diabetes..."
    }
  ],
  "knowledge_graph": {
    "entities": ["Type2Diabetes", "Hypertension", "Metformin"],
    "relationships": [
      {"from": "Type2Diabetes", "rel": "TREATED_BY", "to": "Metformin"}
    ]
  },
  "reasoning": {
    "triggered_rules": ["elevated_cardiovascular_risk", "hba1c_monitoring"],
    "derived_facts": [
      {"fact": "cardiovascular_risk", "value": "elevated", "rule": "CVR_001"}
    ],
    "reasoning_trace": [
      {
        "rule_id": "CVR_001",
        "rule_name": "elevated_cardiovascular_risk",
        "input_facts": {"has_diabetes": true, "age": 58},
        "conclusion": {"cardiovascular_risk": "elevated"},
        "source": "ACC/AHA_2019_Guideline"
      }
    ]
  },
  "confidence": 0.82,
  "limitations": [
    "Lab results not provided — HbA1c and lipid panel recommended.",
    "This is informational only — consult a qualified clinician."
  ]
}
```

---

## 13. Testing

```bash
# Run all tests
pytest tests/ -v

# Run Stage 1 tests only
pytest tests/test_health.py -v

# With coverage
pytest tests/ --cov=app --cov-report=term-missing
```

---

## 14. Docker Setup

```bash
# Start all services (FastAPI + Neo4j)
docker-compose up -d

# View backend logs
docker-compose logs -f backend

# Stop all services
docker-compose down

# Stop and remove volumes (full reset)
docker-compose down -v
```

**Note on ChromaDB**: ChromaDB runs in embedded persistent mode inside the
backend container, using the `./data/chroma` volume mount. A standalone
ChromaDB server service is available in `docker-compose.yml` but commented
out — uncomment it if you prefer a dedicated server.

---

## 15. Project Limitations

1. **Academic prototype** — not validated for clinical use.
2. **BGE embeddings** require ~1.5 GB RAM (CPU) or GPU for reasonable throughput.
3. **Neo4j** requires a running instance; health check degrades gracefully if unavailable.
4. **No authentication** — add OAuth2 / JWT before any real-world deployment.
5. **Rule engine** is currently Python-only; a production system should use
   a standards-based clinical rule format (CDS Hooks, FHIR PlanDefinition).
6. **Knowledge graph** is manually populated — automated NLP extraction has
   error rates that require clinical expert review before trust.

---

## 16. Medical Safety Disclaimer

> ⚠️ **THIS SYSTEM IS NOT A MEDICAL DEVICE**
>
> This software is a **research prototype** created for educational purposes.
>
> - It does **not** provide medical diagnoses.
> - It does **not** replace clinical judgment.
> - It does **not** constitute medical advice.
> - All responses must be reviewed by a qualified healthcare professional.
> - Do **not** use this system to make real patient care decisions.
> - Synthetic/anonymised data must be used during development.
> - Real patient data must **never** be input into this system.
>
> Developers and users accept full responsibility for appropriate use.

---

## Development Stages

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | FastAPI foundation, config, health endpoint | ✅ Complete |
| 2 | PDF ingestion, chunking, BGE embeddings, ChromaDB | 🔜 Next |
| 3 | Neo4j knowledge graph (entities + relationships) | 🔜 Planned |
| 4 | Patient context, symbolic rule engine, reasoning trace | 🔜 Planned |
| 5 | LangChain LLM integration, prompt architecture | 🔜 Planned |
| 6 | Neuro-symbolic integration (full pipeline) | 🔜 Planned |
| 7 | Tests, logging, error handling, Docker, docs | 🔜 Planned |
