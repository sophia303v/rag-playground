"""
Embedding module with multiple backend support:
- sentence-transformers: local sentence-transformers model (default, semantic)
- tfidf: TF-IDF (fully offline, no downloads needed)
- gemini: text-embedding-004 (production, requires API key)
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import config

# --- TF-IDF backend (fully offline) ---
_tfidf_vectorizer = None
_tfidf_corpus = None

def _fit_tfidf(corpus: list[str]):
    """Fit TF-IDF on the full corpus. Must be called before queries."""
    global _tfidf_vectorizer, _tfidf_corpus
    _tfidf_vectorizer = TfidfVectorizer(max_features=768, stop_words="english")
    _tfidf_vectorizer.fit(corpus)
    _tfidf_corpus = corpus

def _embed_texts_tfidf(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    global _tfidf_vectorizer
    if _tfidf_vectorizer is None:
        _fit_tfidf(texts)
    vectors = _tfidf_vectorizer.transform(texts).toarray()
    # Pad or truncate to fixed dimension for ChromaDB
    target_dim = 768
    if vectors.shape[1] < target_dim:
        pad = np.zeros((vectors.shape[0], target_dim - vectors.shape[1]))
        vectors = np.hstack([vectors, pad])
    return vectors.tolist()


# --- Sentence-Transformers backend ---
_st_model = None

def _get_st_model():
    """Load and cache the SentenceTransformer model."""
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        _st_model = SentenceTransformer(
            config.SENTENCE_TRANSFORMER_MODEL,
            cache_folder=str(config.MODELS_DIR),
        )
    return _st_model

def _embed_texts_st(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    model = _get_st_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


# --- Gemini backend ---
def _get_gemini_client():
    from google import genai
    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set.")
    return genai.Client(api_key=config.GEMINI_API_KEY)

def _embed_texts_gemini(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    from google.genai import types
    client = _get_gemini_client()
    embeddings = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i + 100]
        result = client.models.embed_content(
            model=config.EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        embeddings.extend([e.values for e in result.embeddings])
    return embeddings


# --- Public API ---
def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    backend = getattr(config, "EMBEDDING_BACKEND", "sentence-transformers")
    if backend == "gemini":
        return _embed_texts_gemini(texts, task_type)
    elif backend == "sentence-transformers":
        return _embed_texts_st(texts, task_type)
    else:
        return _embed_texts_tfidf(texts, task_type)

def embed_query(query: str) -> list[float]:
    return embed_texts([query], task_type="RETRIEVAL_QUERY")[0]

def get_client():
    return _get_gemini_client()
