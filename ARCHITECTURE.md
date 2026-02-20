# Medical Imaging Multimodal RAG — Architecture Guide

## System Overview

This is a Retrieval-Augmented Generation (RAG) system for radiology reports. Users can ask medical imaging questions (optionally with an uploaded chest X-ray), and the system retrieves relevant radiology reports from a vector database, then uses an LLM to generate evidence-based answers with citations.

### Data Flow

```
User Question (+ optional X-ray image)
        │
        ▼
┌─ retriever.py ───────────────────┐
│  1. Image → Gemini Vision description │
│  2. Text query → embedding            │
│  3. ChromaDB vector search            │
└───────────────────────────────────┘
        │ top-K relevant documents
        ▼
┌─ generator.py ───────────────────┐
│  Gemini 2.0 Flash generates answer    │
│  with source citations + disclaimer   │
└───────────────────────────────────┘
        │
        ▼
    Gradio Web UI (app.py)
```

---

## File-by-File Explanation

### `config.py` — Centralized Configuration

**What it does:**
Defines all project-wide settings in one place: file paths, API keys, model names, and RAG hyperparameters.

```python
CHUNK_SIZE = 512          # tokens per chunk
CHUNK_OVERLAP = 50        # overlap between chunks
TOP_K = 5                 # number of retrieved documents
EMBEDDING_BACKEND = "local"  # "local" (TF-IDF) or "gemini"
```

**Why it's designed this way:**
- Having a single config file avoids hardcoded values scattered across modules.
- `EMBEDDING_BACKEND` allows switching between offline (TF-IDF) and production (Gemini) embeddings via an environment variable, without changing any code.
- API keys are loaded from `.env` via `python-dotenv`, keeping secrets out of source code.

---

### `src/data_loader.py` — Data Loading & Parsing

**What it does:**
- Defines the `MedicalReport` dataclass (uid, indication, findings, impression, images).
- Loads data from two sources:
  1. **HuggingFace** (`ykumards/open-i` dataset) — the Indiana Chest X-ray dataset.
  2. **Local JSON** — cached or sample data for offline use.
- Provides save/load functions for JSON caching.

**Why it's designed this way:**
- `MedicalReport` as a dataclass gives a clean, typed structure. The `full_text` property combines all sections into one string, which is useful for the "full" chunk.
- HuggingFace loading has a `max_samples` parameter so development and testing can use a small subset (e.g., 300) instead of the full dataset.
- JSON caching (`save_reports_to_json` / `load_reports_from_json`) avoids re-downloading from HuggingFace on every run.
- Reports without findings or impression are filtered out — empty reports add noise to the vector store without value.

---

### `src/chunking.py` — Text Chunking Strategy

**What it does:**
Splits each radiology report into chunks by its natural sections:
1. **Indication** — why the exam was ordered
2. **Findings** — what the radiologist observed
3. **Impression** — the conclusion/diagnosis
4. **Full text** — all sections combined

Each chunk carries metadata (report uid, section name, whether images exist).

**Why it's designed this way:**
- Radiology reports have a well-defined structure. Cutting along these natural semantic boundaries keeps each chunk self-contained and meaningful, unlike fixed-size chunking (e.g., every 512 tokens) which can split a sentence in half.
- Section-level chunks improve retrieval precision: a question about diagnosis will match `impression` chunks; a question about observations will match `findings` chunks.
- The full-text chunk provides broader context for general questions that span multiple sections.
- Metadata enables filtering and source citation in the final answer.

**Example:** One report produces ~4 chunks:
```
Chunk 1: "Indication: Cough and fever for 3 days."
Chunk 2: "Findings: The heart size is normal..."
Chunk 3: "Impression: Right lower lobe pneumonia."
Chunk 4: "Indication: Cough and fever...\nFindings: ...\nImpression: ..."
```

---

### `src/embedding.py` — Embedding Module (Dual Backend)

**What it does:**
Converts text into numerical vectors (embeddings) for vector search. Supports two backends:

| Backend | Method | Use Case |
|---|---|---|
| `tfidf` (default) | TF-IDF with scikit-learn | Offline development, no API needed |
| `gemini` | Gemini `text-embedding-004` | Production, higher quality |

Public API:
- `embed_texts(texts)` — embed a list of documents
- `embed_query(query)` — embed a single search query

**Why it's designed this way:**
- The dual backend lets you develop and test entirely offline with TF-IDF, then switch to Gemini embeddings in production by setting `EMBEDDING_BACKEND=gemini` in `.env`.
- TF-IDF vectors are padded to 768 dimensions to match ChromaDB's expected fixed dimension, ensuring the same vector store works with both backends.
- Gemini embeddings are batched (100 per request) to respect API rate limits.
- `task_type` parameter distinguishes between document indexing (`RETRIEVAL_DOCUMENT`) and query embedding (`RETRIEVAL_QUERY`), which Gemini's embedding model uses to optimize the vectors for asymmetric search.

---

### `src/vector_store.py` — ChromaDB Vector Store

**What it does:**
- Manages a persistent ChromaDB database at `data/chroma_db/`.
- `index_chunks()` — takes chunks, generates embeddings, and stores them in ChromaDB.
- `search()` — takes a query string, embeds it, and returns the top-K most similar documents.

**Why it's designed this way:**
- **ChromaDB** is chosen for simplicity — it's an embedded vector database that runs in-process with no external server, ideal for a self-contained project.
- **Persistent storage** (`PersistentClient`) means the index survives restarts; you only need to run ingestion once.
- **Cosine similarity** (`hnsw:space: cosine`) is used because it measures directional similarity between vectors, which works well for text embeddings regardless of document length.
- **Batch indexing** (default 50 per batch) prevents memory issues with large datasets.
- The skip-if-indexed check avoids accidental duplicate indexing.

---

### `src/retriever.py` — Multimodal Retrieval

**What it does:**
1. If the user uploads an image → calls Gemini Vision to generate a text description of the medical image.
2. Combines the text query with the image description (if any).
3. Searches the vector store for the most relevant documents.
4. Returns a `RetrievalResult` with documents, metadata, distances, and the image description.

**Why it's designed this way:**
- The knowledge base is text-only (radiology reports), so images must be converted to text before searching. Gemini Vision acts as a "bridge" between modalities.
- Combining `query + image description` into one search string lets the vector search consider both what the user asked and what's in the image.
- `RetrievalResult.context` property formats the retrieved documents with source labels, ready to be injected into the LLM prompt.
- The `describe_image()` prompt is specifically tuned for radiology: it asks for "clinical terminology," "abnormalities," and "anatomical structures."

---

### `src/generator.py` — Answer Generation

**What it does:**
- Builds a prompt that combines the user's question + retrieved documents + image analysis.
- Sends the prompt to **Gemini 2.0 Flash** with a medical system prompt.
- Falls back to returning raw retrieved results if the API is unavailable.
- Appends a medical disclaimer if the LLM didn't include one.

**Why it's designed this way:**
- The **system prompt** enforces critical rules: only answer from provided references, always cite sources, include a disclaimer. This reduces hallucination and keeps answers grounded.
- **Temperature 0.3** is intentionally low — medical information requires factual, conservative responses rather than creative ones.
- **Local fallback** (`_generate_local`) ensures the app still works without an API key — it just shows the retrieved documents instead of an LLM-generated summary. This is valuable for development and demo.
- The disclaimer auto-append is a safety net: if the LLM forgets to include a disclaimer, the code adds one.

---

### `src/rag_pipeline.py` — Pipeline Orchestrator

**What it does:**
The `MedicalImagingRAG` class ties everything together:
- `__init__()` — checks if an index already exists in ChromaDB.
- `ingest(max_samples)` — loads data → chunks → indexes (one-time setup).
- `query(question, image)` — retrieves → generates (per-query).

**Why it's designed this way:**
- A single class provides a clean, two-method API: `ingest()` then `query()`. This makes it easy to use from both the Gradio UI and the CLI script.
- Data loading has a priority chain: cache → local sample JSON → HuggingFace download. This minimizes network calls and supports offline use.
- `_is_ingested` flag prevents querying before data is indexed, giving a clear error message instead of a confusing empty result.

---

### `app.py` — Gradio Web Interface

**What it does:**
- Builds a web UI with Gradio on port 7860.
- Left panel: image upload (optional) + text question + example questions.
- Right panel: AI-generated answer + retrieved sources with relevance scores.
- Auto-runs ingestion on first launch if no index exists.

**Why it's designed this way:**
- **Gradio** provides a polished web UI with minimal code — no need for a separate frontend.
- The two-column layout separates input (left) from output (right), making the workflow intuitive.
- **Example questions** lower the barrier to entry — users can click to try without thinking of a question.
- Source display shows report UIDs, sections, and relevance percentages, so users can verify the answer's basis.

---

### `ingest.py` — CLI Ingestion Script

**What it does:**
A standalone script to run data ingestion from the command line. Checks if an index exists, offers to re-index, runs ingestion, and does a quick test query.

**Why it's designed this way:**
- Separating ingestion from the app lets you index data without starting the web server.
- The re-index confirmation prevents accidental data loss.
- The quick test query at the end verifies the pipeline works end-to-end.

---

### `data/sample_reports.json` — Sample Dataset

**What it does:**
Contains ~20 synthetic radiology reports covering common conditions: pneumonia, CHF, pneumothorax, lung nodules, COPD, tuberculosis, etc.

**Why it exists:**
- Allows the project to run completely offline without downloading from HuggingFace.
- Covers a diverse range of findings for meaningful demo queries.
- Small enough to index in seconds, making development fast.

---

## Key Design Decisions Summary

| Decision | Rationale |
|---|---|
| Section-based chunking (not fixed-size) | Radiology reports have natural semantic boundaries |
| Dual embedding backend (TF-IDF / Gemini) | Offline development + production quality |
| ChromaDB (embedded, not client-server) | Simple deployment, no external dependencies |
| Gemini Vision for image → text | Bridges image modality to text-only knowledge base |
| Low temperature (0.3) for generation | Medical domain requires conservative, factual answers |
| Local fallback for generation | Works without API key for development/demo |
| JSON caching for loaded data | Avoids repeated HuggingFace downloads |
| Custom RAGAS metrics over `ragas` library | Avoids ~15 OpenAI/LangChain dependencies; metrics are simple LLM prompts |
| Gemini as eval judge (not OpenAI) | Reuses existing API key; no additional provider needed |
| Graceful degradation for eval | Retrieval metrics work without API key; LLM metrics are optional |

---

## Evaluation System

### Overview

The evaluation module (`src/evaluation/`) measures RAG quality using 7 metrics across two categories, tested against a golden QA dataset of 40 manually crafted question-answer pairs.

### Architecture

```
data/golden_qa.json (40 QA pairs)
        │
        ▼
┌─ runner.py ─────────────────────────┐
│  For each question:                       │
│  1. Run RAG pipeline → get answer         │
│  2. Compute retrieval metrics             │
│  3. Run LLM-based metrics (if API key)    │
│  4. Run LLM judge (if API key)            │
└───────────────────────────────────────┘
        │ EvaluationReport
        ▼
┌─ visualization.py ──────────────────┐
│  Plotly HTML report:                      │
│  - Radar chart (7 metrics)                │
│  - Bar chart (by category)                │
│  - Summary + detail tables                │
└───────────────────────────────────────┘
        │
        ▼
    data/eval_report.html
    data/eval_results.json
```

### Metrics

**RAGAS-Style (metrics.py):**

| Metric | Method | What it measures |
|---|---|---|
| Context Precision | Set intersection | % of retrieved docs that are relevant |
| Context Recall | Set intersection | % of ground truth docs that were retrieved |
| Faithfulness | Gemini LLM call | Is the answer grounded in retrieved context? |
| Answer Relevancy | Gemini LLM call | Does the answer address the question? |

**LLM-as-Judge (llm_judge.py):**

| Criterion | What it measures |
|---|---|
| Medical Appropriateness | Correct terminology, clinically accurate |
| Citation Accuracy | Sources cited and match content |
| Answer Completeness | Covers key points from ground truth |

### Golden QA Dataset

40 pairs from the 20 sample reports, stratified by difficulty:
- 10 factual/easy — single report lookup
- 10 factual/medium — requires finding the right report
- 10 comparative/medium — synthesize across reports
- 5 diagnostic/hard — match patient presentation to reports
- 5 procedural/easy — about devices/tubes/lines

### File Breakdown

- `src/evaluation/metrics.py` — `MetricResult` dataclass + 4 metric functions
- `src/evaluation/llm_judge.py` — `JudgeResult` dataclass + `judge_answer()` via single Gemini call
- `src/evaluation/runner.py` — `run_evaluation()` orchestrator, `EvaluationReport` with aggregation
- `src/evaluation/visualization.py` — `generate_html_report()` using Plotly CDN
- `src/evaluation/__init__.py` — public API exports
- `run_eval.py` — CLI entry point with `--retrieval-only` flag for API-free mode
