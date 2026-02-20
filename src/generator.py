"""
Generation module: produces answers using retrieved context.
- Gemini backend: full LLM generation with multimodal support
- Ollama backend: local LLM generation via Ollama REST API
- Local backend: returns retrieved context directly (for testing without API)
"""
from dataclasses import dataclass
import requests
from PIL import Image

import config
from src.retriever import RetrievalResult


SYSTEM_PROMPT = """You are a medical imaging AI assistant. Your role is to help analyze
radiology reports and medical images based on retrieved reference documents.

IMPORTANT RULES:
1. Only answer based on the provided reference documents and image analysis.
2. If the reference documents don't contain relevant information, say so clearly.
3. Always cite which source(s) you used in your answer (e.g., [Source 1], [Source 2]).
4. Include a medical disclaimer that this is for educational/research purposes only.
5. Use clear, professional medical terminology.
6. If an image is provided, integrate your image analysis with the retrieved references."""

DISCLAIMER = "\n\n---\n*Disclaimer: This analysis is for educational and research purposes only. Always consult qualified medical professionals for clinical decisions.*"


@dataclass
class GenerationResult:
    """Container for generation results."""
    answer: str
    retrieval: RetrievalResult
    prompt_used: str


def _build_prompt(retrieval: RetrievalResult) -> str:
    """Build the user prompt from retrieval results."""
    prompt = f"""Based on the following reference documents from our medical knowledge base,
please answer the user's question.

--- RETRIEVED REFERENCES ---
{retrieval.context}
--- END REFERENCES ---

User Question: {retrieval.query}"""

    if retrieval.image_description:
        prompt += f"""

Image Analysis: The uploaded image shows: {retrieval.image_description}

Please integrate the image findings with the reference documents to provide a comprehensive answer."""

    return prompt


def _generate_gemini(prompt: str, image: Image.Image | None = None) -> str:
    """Generate answer using Gemini API."""
    from src.embedding import get_client
    client = get_client()

    contents = [prompt]
    if image is not None:
        contents.append(image)

    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=contents,
        config={
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 0.3,
            "max_output_tokens": 1024,
        },
    )
    return response.text


def _generate_ollama(prompt: str) -> str:
    """Generate answer using a local Ollama model."""
    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json={
            "model": config.OLLAMA_MODEL,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 1024},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def _generate_local(prompt: str, retrieval: RetrievalResult) -> str:
    """
    Local fallback: format retrieved context as an answer.
    Used when Gemini API is not available (offline testing).
    """
    answer = f"**Query:** {retrieval.query}\n\n"
    answer += "**Retrieved Evidence:**\n\n"

    for i, (doc, meta, dist) in enumerate(zip(
        retrieval.documents, retrieval.metadatas, retrieval.distances
    )):
        relevance = f"{(1 - dist) * 100:.1f}%"
        uid = meta.get("uid", "unknown")
        section = meta.get("section", "unknown")
        answer += f"**[Source {i+1}]** Report {uid} ({section}) — Relevance: {relevance}\n"
        answer += f"> {doc}\n\n"

    answer += "\n*[Local mode — connect Gemini API for AI-generated analysis]*"
    return answer


def generate_answer(
    retrieval: RetrievalResult,
    image: Image.Image | None = None,
) -> GenerationResult:
    """Generate an answer using retrieved context."""
    prompt = _build_prompt(retrieval)

    backend = config.GENERATION_BACKEND
    try:
        if backend == "ollama":
            answer = _generate_ollama(prompt)
        else:
            answer = _generate_gemini(prompt, image)
    except Exception as e:
        # Fallback chain: try the other LLM backend, then local
        print(f"{backend.title()} unavailable ({e}), trying fallback...")
        try:
            if backend == "ollama":
                answer = _generate_gemini(prompt, image)
            else:
                answer = _generate_ollama(prompt)
        except Exception as e2:
            print(f"All LLM backends failed ({e2}), using local fallback.")
            answer = _generate_local(prompt, retrieval)

    if "disclaimer" not in answer.lower() and "educational" not in answer.lower():
        answer += DISCLAIMER

    return GenerationResult(
        answer=answer,
        retrieval=retrieval,
        prompt_used=prompt,
    )
