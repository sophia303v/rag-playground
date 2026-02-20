"""
RAGAS-style evaluation metrics for the Medical Imaging RAG system.

Implements 4 metrics:
- Context Precision: % of retrieved docs that are relevant (pure computation)
- Context Recall: % of ground truth docs that were retrieved (pure computation)
- Faithfulness: Is the answer grounded in retrieved context? (LLM call)
- Answer Relevancy: Does the answer address the question? (LLM call)
"""
from dataclasses import dataclass
import json
import re

import requests

import config


@dataclass
class MetricResult:
    """Result of a single metric evaluation."""
    name: str
    score: float  # 0.0 to 1.0
    explanation: str


def context_precision(
    retrieved_uids: list[str],
    relevant_uids: list[str],
) -> MetricResult:
    """
    What fraction of retrieved documents are actually relevant?

    Precision = |retrieved ∩ relevant| / |retrieved|
    """
    if not retrieved_uids:
        return MetricResult(
            name="context_precision",
            score=0.0,
            explanation="No documents were retrieved.",
        )

    retrieved_set = set(retrieved_uids)
    relevant_set = set(relevant_uids)
    hits = retrieved_set & relevant_set
    score = len(hits) / len(retrieved_set)

    return MetricResult(
        name="context_precision",
        score=round(score, 4),
        explanation=f"{len(hits)}/{len(retrieved_set)} retrieved docs were relevant. "
                    f"Relevant hits: {sorted(hits) if hits else 'none'}.",
    )


def context_recall(
    retrieved_uids: list[str],
    relevant_uids: list[str],
) -> MetricResult:
    """
    What fraction of ground truth relevant documents were retrieved?

    Recall = |retrieved ∩ relevant| / |relevant|
    """
    if not relevant_uids:
        return MetricResult(
            name="context_recall",
            score=1.0,
            explanation="No relevant documents specified (vacuous truth).",
        )

    retrieved_set = set(retrieved_uids)
    relevant_set = set(relevant_uids)
    hits = retrieved_set & relevant_set
    score = len(hits) / len(relevant_set)

    return MetricResult(
        name="context_recall",
        score=round(score, 4),
        explanation=f"{len(hits)}/{len(relevant_set)} relevant docs were retrieved. "
                    f"Missing: {sorted(relevant_set - retrieved_set) if relevant_set - retrieved_set else 'none'}.",
    )


def faithfulness(
    question: str,
    answer: str,
    context: str,
) -> MetricResult:
    """
    Is the answer grounded in the retrieved context?

    Uses Gemini to check whether each claim in the answer can be
    attributed to the provided context.
    """
    prompt = f"""You are an evaluation judge for a medical RAG system.

Given the CONTEXT (retrieved documents), QUESTION, and ANSWER below,
evaluate whether the ANSWER is faithful to the CONTEXT.

An answer is faithful if every factual claim it makes can be traced back
to information in the context. The answer should not hallucinate facts
that are not in the context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
{answer}

Respond in JSON format:
{{
  "score": <float between 0.0 and 1.0>,
  "explanation": "<brief explanation of your score>"
}}

Scoring guide:
- 1.0: Every claim in the answer is supported by the context
- 0.7-0.9: Most claims are supported, minor unsupported details
- 0.4-0.6: Mix of supported and unsupported claims
- 0.1-0.3: Mostly unsupported claims
- 0.0: Answer contradicts the context or is entirely fabricated"""

    return _call_llm_metric("faithfulness", prompt)


def answer_relevancy(
    question: str,
    answer: str,
) -> MetricResult:
    """
    Does the answer actually address the question?

    Uses Gemini to evaluate whether the answer is on-topic and
    responsive to what was asked.
    """
    prompt = f"""You are an evaluation judge for a medical RAG system.

Given the QUESTION and ANSWER below, evaluate whether the ANSWER
is relevant to the QUESTION.

A relevant answer directly addresses what was asked, stays on topic,
and provides the type of information the question is seeking.

QUESTION:
{question}

ANSWER:
{answer}

Respond in JSON format:
{{
  "score": <float between 0.0 and 1.0>,
  "explanation": "<brief explanation of your score>"
}}

Scoring guide:
- 1.0: Answer directly and completely addresses the question
- 0.7-0.9: Answer mostly addresses the question with minor tangents
- 0.4-0.6: Answer partially addresses the question
- 0.1-0.3: Answer is mostly off-topic
- 0.0: Answer is completely irrelevant to the question"""

    return _call_llm_metric("answer_relevancy", prompt)


def _call_llm_metric(metric_name: str, prompt: str) -> MetricResult:
    """Helper to call LLM (Gemini or Ollama) and parse a JSON metric response."""
    try:
        if config.GENERATION_BACKEND == "ollama":
            text = _call_ollama(prompt)
        else:
            from src.embedding import get_client
            client = get_client()
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=[prompt],
                config={"temperature": 0.0, "max_output_tokens": 256},
            )
            text = response.text.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        parsed = json.loads(text)
        score = float(parsed["score"])
        score = max(0.0, min(1.0, score))

        return MetricResult(
            name=metric_name,
            score=round(score, 4),
            explanation=parsed.get("explanation", ""),
        )

    except Exception as e:
        return MetricResult(
            name=metric_name,
            score=-1.0,
            explanation=f"LLM evaluation failed: {e}",
        )


def _call_ollama(prompt: str) -> str:
    """Call Ollama API and return the response text."""
    response = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json={
            "model": config.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 256},
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"].strip()
