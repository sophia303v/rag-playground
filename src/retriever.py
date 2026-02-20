"""Retrieval module: handles text and image queries."""
from dataclasses import dataclass
from PIL import Image
from google import genai

import config
from src.embedding import get_client
from src.vector_store import search


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


def describe_image(image: Image.Image) -> str:
    """
    Use Gemini Vision to generate a medical description of an image.

    This converts an image query into a text description that can be
    used for vector search against the text-based knowledge base.
    """
    client = get_client()

    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=[
            "You are a radiologist. Describe the key findings in this medical image "
            "in clinical terminology. Focus on abnormalities, anatomical structures, "
            "and any notable observations. Be concise but thorough.",
            image,
        ],
    )

    return response.text


def retrieve(query: str, image: Image.Image | None = None, top_k: int = None) -> RetrievalResult:
    """
    Retrieve relevant medical reports based on text query and optional image.

    For multimodal queries:
    1. If image provided: generate description via Gemini Vision
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
        print("Generating image description with Gemini Vision...")
        image_description = describe_image(image)
        print(f"Image description: {image_description[:200]}...")

        # Combine text query with image description for richer retrieval
        combined_query = f"{query}\n\nImage findings: {image_description}"
    else:
        combined_query = query

    # Search vector store
    print(f"Searching for top-{top_k} relevant documents...")
    results = search(combined_query, n_results=top_k)

    return RetrievalResult(
        query=query,
        image_description=image_description,
        documents=results["documents"][0] if results["documents"] else [],
        metadatas=results["metadatas"][0] if results["metadatas"] else [],
        distances=results["distances"][0] if results["distances"] else [],
    )
