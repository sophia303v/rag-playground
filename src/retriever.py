"""Retrieval module: handles text and image queries."""
import base64
import io
from dataclasses import dataclass

import requests
from PIL import Image

import config
from src.embedding import get_client
from src.vector_store import search, get_chroma_client, get_or_create_collection
from src.prompt_loader import get as get_prompt


_cross_encoder = None


def _rrf_merge(dense_docs, dense_metas, dense_dists,
               bm25_docs, bm25_metas, bm25_scores,
               k: int = 60) -> tuple[list[str], list[dict], list[float]]:
    """Merge dense and BM25 results using Reciprocal Rank Fusion.

    Each document gets score = sum(1 / (k + rank)) across the lists it appears in.
    Documents are identified by their text content for deduplication.
    """
    doc_scores: dict[str, float] = {}
    doc_meta: dict[str, dict] = {}
    doc_text: dict[str, str] = {}  # id -> text

    # Process dense results
    for rank, (doc, meta) in enumerate(zip(dense_docs, dense_metas)):
        doc_id = f"{meta.get('uid', '')}:{meta.get('section', '')}:{hash(doc)}"
        doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        doc_meta[doc_id] = meta
        doc_text[doc_id] = doc

    # Process BM25 results
    for rank, (doc, meta) in enumerate(zip(bm25_docs, bm25_metas)):
        doc_id = f"{meta.get('uid', '')}:{meta.get('section', '')}:{hash(doc)}"
        doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        doc_meta[doc_id] = meta
        doc_text[doc_id] = doc

    # Sort by fused score descending
    ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    fused_docs = [doc_text[did] for did, _ in ranked]
    fused_metas = [doc_meta[did] for did, _ in ranked]
    fused_scores = [score for _, score in ranked]

    return fused_docs, fused_metas, fused_scores


def _get_cross_encoder():
    """Load and cache the cross-encoder model."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        print(f"Loading cross-encoder: {config.CROSS_ENCODER_MODEL}")
        _cross_encoder = CrossEncoder(
            config.CROSS_ENCODER_MODEL,
            cache_folder=str(config.MODELS_DIR),
        )
    return _cross_encoder


@dataclass
class RetrievalResult:
    """Container for retrieval results."""
    query: str
    image_description: str | None
    documents: list[str]
    metadatas: list[dict]
    distances: list[float]

    @property
    def context(self) -> str:
        """Format retrieved documents as context for the generator."""
        context_parts = []
        for i, (doc, meta) in enumerate(zip(self.documents, self.metadatas)):
            source = f"Report {meta.get('uid', 'unknown')} ({meta.get('section', 'unknown')})"
            context_parts.append(f"[Source {i+1}: {source}]\n{doc}")
        return "\n\n".join(context_parts)


def _describe_image_gemini(image: Image.Image) -> str:
    """Use Gemini Vision to generate a medical description of an image."""
    client = get_client()

    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=[get_prompt("image_description_prompt"), image],
    )

    return response.text


def _describe_image_ollama(image: Image.Image) -> str:
    """Use Ollama vision model (e.g. llava) to describe a medical image."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json={
            "model": config.OLLAMA_VISION_MODEL,
            "prompt": get_prompt("image_description_prompt"),
            "images": [b64],
            "stream": False,
            "options": {"temperature": 0.3},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def describe_image(image: Image.Image) -> str:
    """
    Generate a medical description of an image using the configured vision backend.

    Fallback: primary backend fails → try the other → return empty string.
    """
    backend = config.VISION_BACKEND
    primary = _describe_image_ollama if backend == "ollama" else _describe_image_gemini
    fallback = _describe_image_gemini if backend == "ollama" else _describe_image_ollama

    try:
        return primary(image)
    except Exception as e:
        print(f"{backend.title()} vision unavailable ({e}), trying fallback...")
        try:
            return fallback(image)
        except Exception as e2:
            print(f"All vision backends failed ({e2}), skipping image description.")
            return ""


def retrieve(query: str, image: Image.Image | None = None, top_k: int = None) -> RetrievalResult:
    """
    Retrieve relevant medical reports based on text query and optional image.

    For multimodal queries:
    1. If image provided: generate description via configured vision backend
    2. Combine text query + image description
    3. Search vector store for relevant reports

    Args:
        query: Text query from user
        image: Optional uploaded medical image
        top_k: Number of results to return

    Returns:
        RetrievalResult with relevant documents and metadata
    """
    if top_k is None:
        top_k = config.TOP_K

    image_description = None

    # If image is provided, generate a description
    if image is not None:
        backend = config.VISION_BACKEND
        print(f"Generating image description with {backend.title()} Vision...")
        image_description = describe_image(image)
        if image_description:
            print(f"Image description: {image_description[:200]}...")
            # Combine text query with image description for richer retrieval
            combined_query = f"{query}\n\nImage findings: {image_description}"
        else:
            combined_query = query
    else:
        combined_query = query

    # Determine how many candidates to fetch from each source
    need_fusion = config.BM25_ENABLED or config.RERANK_ENABLED
    if need_fusion:
        n_candidates = max(config.RERANK_CANDIDATES, top_k)
    else:
        n_candidates = top_k

    # Dense retrieval (ChromaDB)
    results = search(combined_query, n_results=n_candidates)
    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    dists = results["distances"][0] if results["distances"] else []

    # BM25 sparse retrieval + RRF fusion
    if config.BM25_ENABLED:
        from src.bm25_index import get_or_build_index
        client = get_chroma_client()
        collection = get_or_create_collection(client)
        bm25 = get_or_build_index(collection)
        bm25_docs, bm25_metas, bm25_scores = bm25.search(combined_query, n_results=n_candidates)
        docs, metas, dists = _rrf_merge(
            docs, metas, dists,
            bm25_docs, bm25_metas, bm25_scores,
            k=config.RRF_K,
        )

    # Re-rank with cross-encoder
    if config.RERANK_ENABLED and len(docs) > top_k:
        ce = _get_cross_encoder()
        pairs = [[combined_query, doc] for doc in docs]
        scores = ce.predict(pairs)

        ranked = sorted(
            zip(scores, docs, metas, dists),
            key=lambda x: x[0],
            reverse=True,
        )
        ranked = ranked[:top_k]
        _, docs, metas, dists = zip(*ranked)
        docs, metas, dists = list(docs), list(metas), list(dists)
    elif len(docs) > top_k:
        # Trim to top_k if no re-ranking (e.g. BM25-only fusion)
        docs, metas, dists = docs[:top_k], metas[:top_k], dists[:top_k]

    return RetrievalResult(
        query=query,
        image_description=image_description,
        documents=docs,
        metadatas=metas,
        distances=dists,
    )
