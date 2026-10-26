# Nawaloka Health Assistant

A multi-agent healthcare assistant built with FastAPI and LangGraph. It routes
requests to CRM, RAG, web-search, and direct-agent paths, with short-term and
semantic long-term memory.

This system has no application authentication and must not be exposed publicly
or used with real patient data.

## Technology

- Python 3.13 and `uv`
- FastAPI, Pydantic, and Uvicorn
- LangGraph, LangChain, and OpenRouter
- SQLAlchemy, Supabase PostgreSQL, and pgvector
- Qdrant Cloud for knowledge-base retrieval and CAG caching
- Tavily for optional real-time web search
- Langfuse for optional tracing and prompt retrieval
- Loguru for logging

## Current functionality

### API

- `GET /health` — application liveness/readiness response.
- `POST /chat` — runs the asynchronous LangGraph agent and returns the answer,
  route(s), action, tool output, memory context, and latency.
- `GET /graph` — returns Mermaid text and structured graph nodes/edges.

### Agent routes

- `direct` — greetings and general requests.
- `crm` — patient lookup, doctor/specialty search, upcoming booking lookup,
  and booking creation.
- `rag` — internal hospital knowledge-base questions through Qdrant.
- `web_search` — optional Tavily search for current external information.
- Multiple routes can run in parallel and are merged into one response.

### Data and memory

- CRM tables for locations, specialties, doctors, patients, and bookings.
- Deterministic CRM seed data from the checked-in SQL files.
- Parent-child chunking and Qdrant ingestion for 12 knowledge-base documents.
- CRAG confidence handling and CAG semantic answer caching.
- FAQ cache warming from `config/faqs.yaml`.
- Short-term conversation memory in `st_turns`, scoped by user and session.
- Semantic long-term facts in `mem_facts`, including query, distillation,
  scoring, deduplication, and post-response upsert.
- Optional Langfuse tracing with local prompt fallbacks.

## Architecture

```text
POST /chat
    |
FastAPI lifespan
    |
AgentOrchestrator
    |
    +--> recall --> supervisor --> route decisions
                              |
                              +--> direct_agent
                              +--> admin_agent --> CRM
                              +--> clinical_agent --> RAG/CRAG/CAG
                              +--> direct_agent --> Tavily (optional)
                                      |
                                      v
                              merge_responses
                                      |
                                      v
                                save_memory
                                      |
                                      v
                                   response
```

```text
Supabase PostgreSQL + pgvector
  -> CRM records, st_turns, and mem_facts

Qdrant Cloud
  -> nawaloka knowledge-base collection
  -> cag_cache semantic answer cache
```

Specialist branches write to the reducer-backed `agent_outputs` field. The
fan-in node writes scalar `final_answer` and `tool_output` values, avoiding
concurrent graph-state updates during multi-route execution.

## Setup and run

### Prerequisites

Install Python 3.13 and [`uv`](https://docs.astral.sh/uv/).

Required for the complete system:

- OpenRouter — router, chat, extraction, and embedding models.
- Supabase PostgreSQL — CRM records and durable memory.
- Qdrant — knowledge-base retrieval and CAG cache.

Optional integrations:

- Tavily — real-time web search.
- Langfuse — tracing and managed prompt overrides.

The API can start without optional integrations, but their features will be
unavailable. CRM, memory, and RAG functionality require their corresponding
database service.

### Install dependencies

From the repository root:

```bash
uv sync --extra dev
```

### Configure environment

PowerShell:

```powershell
Copy-Item .env.example .env
```

Set these required values in `.env`:

```env
OPENROUTER_API_KEY=...
SUPABASE_DB_URL=...
QDRANT_URL=...
QDRANT_API_KEY=...
QDRANT_COLLECTION_NAME=nawaloka
```

Optional integrations:

```env
TAVILY_API_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=...
```

Never commit `.env` or provider keys.

### Initialise data services

Apply `sql/001_crm_schema.sql` in the Supabase SQL editor. Apply
`sql/007_short_term_memory.sql` and `sql/008_long_term_memory.sql` when using
the memory path. Then seed CRM data:

```bash
PYTHONPATH="$PWD/src" uv run python scripts/seed_crm.py
```

Ingest the knowledge base and warm the FAQ cache:

```bash
PYTHONPATH="$PWD/src" uv run python scripts/ingest_to_qdrant.py
PYTHONPATH="$PWD/src" uv run python scripts/warm_cag_cache.py
```

The ingestion script uses the checked-in documents and parent-child chunking.
The cache script uses `config/faqs.yaml`.

### Start the API

```bash
PYTHONPATH="$PWD/src" uv run uvicorn api.main:app --app-dir src --reload --port 8000
```

PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\src"
uv run uvicorn api.main:app --app-dir src --reload --port 8000
```

Check the API:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/graph
```

Send a chat request:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_message":"What cardiac services does the hospital offer?","user_id":"demo-user","session_id":"demo-session"}'
```
