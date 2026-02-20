"""Centralized configuration for Medical Imaging RAG."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CHROMA_DIR = DATA_DIR / "chroma_db"

# Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "text-embedding-004"

# Embedding backend: "local" (offline, default) or "gemini" (requires API)
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local")

# Generation backend: "gemini" (default) or "ollama" (local)
GENERATION_BACKEND = os.getenv("GENERATION_BACKEND", "gemini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# RAG Settings
CHUNK_SIZE = 512          # tokens per chunk (approx)
CHUNK_OVERLAP = 50        # overlap between chunks
TOP_K = 5                 # number of retrieved documents
COLLECTION_NAME = "medical_reports"

# Embedding
EMBEDDING_DIMENSION = 768  # text-embedding-004 output dimension

# Evaluation
GOLDEN_QA_PATH = DATA_DIR / "golden_qa.json"
EVAL_RESULTS_PATH = DATA_DIR / "eval_results.json"
EVAL_REPORT_PATH = DATA_DIR / "eval_report.html"
