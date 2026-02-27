# RAG Playground

A modular Retrieval-Augmented Generation framework with pluggable components. Swap embedding backends, retrieval strategies (dense, BM25 hybrid, cross-encoder reranking), generation models, and evaluation metrics — then benchmark across datasets like SQuAD v2, SciFact, and custom QA sets.

## Features

### RAG Pipeline
- **Multimodal retrieval** — text queries + optional image input (Gemini Vision converts images to text for search)
- **Hybrid retrieval** — Dense (ChromaDB) + Sparse (BM25) with Reciprocal Rank Fusion, then cross-encoder reranking
- **Triple embedding backend** — sentence-transformers (default), TF-IDF (offline), Gemini `text-embedding-004` (API)
- **ChromaDB vector store** — persistent, embedded database with cosine similarity search
- **Multi-LLM generation** — Gemini 2.0 Flash, Ollama (local), with automatic fallback chain
- **Local fallback** — works without API key by returning raw retrieved documents

### Evaluation System
- **6 retrieval metrics** — Context Precision, Context Recall, MRR (Mean Reciprocal Rank), nDCG (Normalized Discounted Cumulative Gain), Faithfulness, Answer Relevancy
- **Combined LLM evaluation** — 5 metrics in a single API call (faithfulness, relevancy, domain appropriateness, citation accuracy, completeness)
- **Multi-backend eval** — Groq (fast, free) > OpenRouter > Gemini > Ollama fallback chain
- **Multiple benchmark datasets** — custom medical QA (40 pairs), SQuAD v2 (2000 pairs), SciFact (300 queries)
- **Checkpoint/resume** — saves progress per question, resumes from last checkpoint on interruption
- **Experiment management** — YAML configs, auto-named experiment folders, config snapshots
- **Interactive HTML report** — radar chart, grouped bar chart, metric descriptions, per-question detail table (Plotly)
- **Cross-experiment comparison** — overlaid radar charts, delta tables, config diff
- **Graceful degradation** — retrieval metrics work without API key; LLM metrics are optional

### Web Interface
- **Gradio UI** on port 7860 — image upload, text input, example questions, source display with relevance scores

## Experiment Results

All experiments use `all-MiniLM-L12-v2` (sentence-transformers) for embedding, ChromaDB for dense retrieval, and `cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking.

### Retrieval Strategy Comparison (SQuAD v2, 200 questions)

| Configuration | Recall | MRR | nDCG | Faithfulness | Relevancy | Completeness |
|---|---|---|---|---|---|---|
| Baseline (dense only) | 0.635 | 0.555 | 0.576 | 0.900 | 0.700 | 0.650 |
| + Cross-encoder reranking | 0.810 | 0.779 | 0.787 | 0.875 | 0.875 | 0.750 |
| + BM25 hybrid + RRF + reranking | **0.895** | **0.858** | **0.868** | — | — | — |

### Earlier Experiments

| Experiment | Dataset | K | Questions | Mode | Recall | MRR | nDCG |
|---|---|---|---|---|---|---|---|
| scifact_baseline | scifact | 3 | 300 | Retrieval | 0.614 | 0.550 | 0.559 |
| squad_v2_baseline | squad_v2 | 3 | 2000 | Retrieval | 0.737 | 0.644 | 0.668 |
| squad_v2_rerank | squad_v2 | 3 | 2000 | Retrieval | 0.893 | 0.849 | 0.861 |
| squad_v2_bm25_hybrid | squad_v2 | 3 | 2000 | Retrieval | 0.941 | 0.894 | 0.906 |

**Key observations:**
- **BM25 hybrid + reranking** is the best retrieval strategy: +41% recall over dense-only baseline
- **Cross-encoder reranking** alone gives a major boost (+27.6% recall), BM25 adds another +10.5%
- **Faithfulness 0.875–0.900**: generated answers are well-grounded in retrieved context
- **Answer completeness** improves with better retrieval (0.650 → 0.750)

Each experiment generates an interactive HTML report in `experiments/{date}_{name}/eval_report.html`.

## Project Structure

```
rag-playground/
├── app.py                    # Gradio web interface
├── ingest.py                 # CLI data ingestion script
├── run_eval.py               # CLI evaluation entry point
├── config.py                 # Centralized configuration
├── requirements.txt
├── ARCHITECTURE.md           # Detailed architecture documentation
│
├── src/
│   ├── data_loader.py        # Load reports from HuggingFace or local JSON
│   ├── chunking.py           # Section-based text chunking
│   ├── embedding.py          # ST / TF-IDF / Gemini embedding backends
│   ├── vector_store.py       # ChromaDB indexing & search
│   ├── retriever.py          # Dense + BM25 hybrid retrieval with reranking
│   ├── bm25_index.py         # BM25 sparse keyword index
│   ├── generator.py          # Gemini / Ollama answer generation with citations
│   ├── rag_pipeline.py       # Orchestrator (ingest + query)
│   └── evaluation/
│       ├── metrics.py        # Retrieval metrics + LLM-based metrics
│       ├── llm_judge.py      # 3-criteria LLM judge
│       ├── combined_eval.py  # 5-metric combined LLM eval (Groq/OpenRouter/Gemini/Ollama)
│       ├── runner.py         # Evaluation orchestrator with checkpoint/resume
│       └── visualization.py  # Plotly HTML report + cross-experiment comparison
│
├── scripts/
│   └── download_datasets.py  # Download SQuAD v2 / SciFact / RadQA benchmarks
│
├── experiments/              # Auto-generated experiment results
│   └── {date}_{name}/
│       ├── config.yaml       # Parameter snapshot
│       ├── eval_results.json # Full scores + generated answers
│       └── eval_report.html  # Interactive HTML report
│
└── data/
    ├── openi_synthetic/      # 20 medical reports + 40 QA pairs
    ├── squad_v2/             # SQuAD v2 benchmark (1204 passages, 2000 QA)
    ├── scifact/              # SciFact benchmark (5183 docs, 300 queries)
    └── chroma_db/            # Vector store (rebuilt per dataset)
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up API key (optional)

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
EMBEDDING_BACKEND=gemini    # or "local" for offline TF-IDF (default)
```

Without an API key the system still works — it uses TF-IDF embeddings and returns retrieved documents directly instead of generating LLM answers.

### 3. Ingest data

```bash
python ingest.py
```

This loads the 20 sample radiology reports, chunks them by section, generates embeddings, and stores them in ChromaDB. Only needs to run once (data persists in `data/chroma_db/`).

### 4. Launch the web UI

```bash
python app.py
```

Open http://localhost:7860 in your browser. Type a question or click an example, optionally upload a chest X-ray image.

### 5. Run evaluation

```bash
# Full evaluation (requires GEMINI_API_KEY)
python run_eval.py

# Retrieval metrics only (no API key needed)
python run_eval.py --retrieval-only

# Run on benchmark datasets
python scripts/download_datasets.py squad_v2    # download first
python run_eval.py --dataset squad_v2 --retrieval-only --experiment-name squad_test

# Quick test with limited questions
python run_eval.py --dataset squad_v2 --max-samples 20 --experiment-name quick_test

# YAML config for reproducible experiments
python run_eval.py --config experiments/configs/example.yaml --experiment-name my_exp
```

Results are saved to `experiments/{date}_{name}/` with an interactive HTML report.

**CLI options:**

| Flag | Description |
|---|---|
| `--retrieval-only` | Skip LLM-based metrics (no API key needed) |
| `--dataset NAME` | Switch dataset (squad_v2, scifact, openi_synthetic) |
| `--max-samples N` | Limit to first N questions (useful for quick tests) |
| `--top-k N` | Override number of retrieved documents |
| `--experiment-name NAME` | Save results to `experiments/{date}_{name}/` |
| `--config PATH` | Load YAML config for parameter overrides |
| `--embedding-backend` | sentence-transformers, tfidf, or gemini |
| `--generation-backend` | gemini or ollama |
| `--qa PATH` | Custom golden QA file |
| `--output-dir PATH` | Output directory (default: `data/`) |
| `--quiet` | Suppress progress output |

## How It Works

```
User Question (+ optional image)
        │
        ▼
┌─ Retriever ───────────────────────────┐
│  1. Image → Vision desc. (optional)   │
│  2. Query → embedding                 │
│  3. Dense (ChromaDB) + BM25 (sparse)  │
│  4. Reciprocal Rank Fusion            │
│  5. Cross-encoder reranking           │
└───────────────────────────────────────┘
        │ top-k relevant chunks
        ▼
┌─ Generator ───────────────────────────┐
│  Gemini / Ollama (with fallback)      │
│  + source citations                   │
└───────────────────────────────────────┘
        │
        ▼
    Answer with references
```

## Evaluation Metrics

| Metric | Type | What it measures |
|---|---|---|
| Context Precision | Retrieval | % of retrieved docs that are relevant |
| Context Recall | Retrieval | % of ground truth docs that were retrieved |
| MRR (Reciprocal Rank) | Retrieval | 1 / position of first relevant doc |
| nDCG | Retrieval | Ranking quality, weighted by position |
| Faithfulness | Generation (LLM) | Is the answer grounded in retrieved context? |
| Answer Relevancy | Generation (LLM) | Does the answer address the question? |
| Domain Appropriateness | Judge (LLM) | Correct terminology, domain-appropriate |
| Citation Accuracy | Judge (LLM) | Sources cited and match content |
| Answer Completeness | Judge (LLM) | Covers key points from ground truth |

## Data

The project supports multiple datasets out of the box:
- **openi_synthetic** — 20 medical radiology reports + 40 QA pairs (custom)
- **squad_v2** — 1204 passages + 2000 QA pairs (Wikipedia-based reading comprehension)
- **scifact** — 5183 scientific abstracts + 300 claim queries (retrieval-only)

Use `python scripts/download_datasets.py <name>` to download benchmark datasets.

## Tech Stack

- **LLM**: Gemini 2.0 Flash (generation + evaluation)
- **Embeddings**: sentence-transformers `all-MiniLM-L12-v2` (default), Gemini `text-embedding-004`, or TF-IDF (offline)
- **Vector DB**: ChromaDB (embedded, persistent)
- **Web UI**: Gradio
- **Visualization**: Plotly
- **Data**: HuggingFace Datasets
