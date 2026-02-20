# Medical Imaging Multimodal RAG

A Retrieval-Augmented Generation system for radiology reports. Ask questions about chest X-ray findings — optionally upload an image — and get AI-generated answers grounded in a knowledge base of radiology reports, with source citations.

## Features

### RAG Pipeline
- **Multimodal retrieval** — text queries + optional chest X-ray image input (Gemini Vision converts images to text for search)
- **Section-aware chunking** — splits radiology reports by natural boundaries (indication / findings / impression) instead of fixed token windows
- **Dual embedding backend** — TF-IDF for offline development, Gemini `text-embedding-004` for production
- **ChromaDB vector store** — persistent, embedded database with cosine similarity search
- **Grounded generation** — Gemini 2.0 Flash generates answers with source citations and medical disclaimers
- **Local fallback** — works without API key by returning raw retrieved documents

### Evaluation System
- **4 RAGAS-style metrics** — Context Precision, Context Recall, Faithfulness, Answer Relevancy
- **3-criteria LLM Judge** — Medical Appropriateness, Citation Accuracy, Answer Completeness
- **40 golden QA pairs** — manually crafted from 20 radiology reports, stratified by difficulty and category
- **Interactive HTML report** — radar chart, grouped bar chart, summary and per-question tables (Plotly)
- **Graceful degradation** — retrieval metrics work without API key; LLM metrics are optional

### Web Interface
- **Gradio UI** on port 7860 — image upload, text input, example questions, source display with relevance scores

## Project Structure

```
medical-imaging-rag/
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
│   ├── embedding.py          # TF-IDF / Gemini embedding backends
│   ├── vector_store.py       # ChromaDB indexing & search
│   ├── retriever.py          # Multimodal retrieval (text + image)
│   ├── generator.py          # Gemini answer generation with citations
│   ├── rag_pipeline.py       # Orchestrator (ingest + query)
│   └── evaluation/
│       ├── metrics.py        # RAGAS-style metrics (precision, recall, faithfulness, relevancy)
│       ├── llm_judge.py      # 3-criteria LLM judge (medical, citation, completeness)
│       ├── runner.py         # Evaluation orchestrator
│       └── visualization.py  # Plotly HTML report generator
│
└── data/
    ├── sample_reports.json   # 20 synthetic radiology reports
    ├── golden_qa.json        # 40 QA pairs for evaluation
    ├── chroma_db/            # Vector store (generated after ingestion)
    ├── eval_results.json     # Evaluation scores (generated after eval)
    └── eval_report.html      # Interactive HTML report (generated after eval)
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
```

After completion, open `data/eval_report.html` in a browser to see the interactive report. Raw scores are saved to `data/eval_results.json`.

**CLI options:**

| Flag | Description |
|---|---|
| `--retrieval-only` | Skip LLM-based metrics (no API key needed) |
| `--qa PATH` | Custom golden QA file (default: `data/golden_qa.json`) |
| `--output-dir PATH` | Output directory (default: `data/`) |
| `--quiet` | Suppress progress output |

## How It Works

```
User Question (+ optional X-ray)
        │
        ▼
┌─ Retriever ──────────────────────┐
│  1. Image → Gemini Vision desc.  │
│  2. Query → embedding            │
│  3. ChromaDB cosine search       │
└──────────────────────────────────┘
        │ top-5 relevant chunks
        ▼
┌─ Generator ──────────────────────┐
│  Gemini 2.0 Flash                │
│  + source citations              │
│  + medical disclaimer            │
└──────────────────────────────────┘
        │
        ▼
    Answer with references
```

## Evaluation Metrics

| Metric | Type | What it measures |
|---|---|---|
| Context Precision | Retrieval | % of retrieved docs that are relevant |
| Context Recall | Retrieval | % of ground truth docs that were retrieved |
| Faithfulness | Generation (LLM) | Is the answer grounded in retrieved context? |
| Answer Relevancy | Generation (LLM) | Does the answer address the question? |
| Medical Appropriateness | Judge (LLM) | Correct terminology, clinically accurate |
| Citation Accuracy | Judge (LLM) | Sources cited and match content |
| Answer Completeness | Judge (LLM) | Covers key points from ground truth |

## Data

The project uses 20 synthetic radiology reports (`data/sample_reports.json`) covering common chest X-ray findings: pneumonia, CHF, pneumothorax, COPD, lung masses, tuberculosis, ARDS, rib fractures, sarcoidosis, and more. For a larger dataset, the system can download the [Indiana University Chest X-ray (OpenI)](https://huggingface.co/datasets/ykumards/open-i) dataset from HuggingFace.

## Tech Stack

- **LLM**: Gemini 2.0 Flash (generation + evaluation)
- **Embeddings**: Gemini `text-embedding-004` or TF-IDF (offline)
- **Vector DB**: ChromaDB (embedded, persistent)
- **Web UI**: Gradio
- **Visualization**: Plotly
- **Data**: HuggingFace Datasets
