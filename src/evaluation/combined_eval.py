"""
Combined LLM evaluation: faithfulness + relevancy + judge in a single API call.

Replaces 3 separate LLM calls with 1, reducing latency ~3x.
"""
import json
import re
from dataclasses import dataclass

import requests

import config
from src.prompt_loader import get as get_prompt


@dataclass
class CombinedEvalResult:
    """All 5 LLM-based metric scores from a single call."""
    faithfulness: float          # 0.0 to 1.0
    answer_relevancy: float      # 0.0 to 1.0
    domain_appropriateness: float  # 0.0 to 1.0
    citation_accuracy: float     # 0.0 to 1.0
    answer_completeness: float   # 0.0 to 1.0
    explanation: str


def _call_eval_llm(prompt: str) -> str:
    """Call the best available LLM for evaluation.

    Priority: Groq (fast, free) > Gemini > Ollama.
    """
    # Try Groq first (fastest, no throttle)
    if config.GROQ_API_KEY:
        resp = requests.post(
            f"{config.GROQ_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
            json={
                "model": config.GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 512,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    # Try OpenRouter (OpenAI-compatible, wide model selection)
    if config.OPENROUTER_API_KEY:
        resp = requests.post(
            f"{config.OPENROUTER_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {config.OPENROUTER_API_KEY}"},
            json={
                "model": config.OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 512,
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()

    # Fallback to Gemini
    if config.GEMINI_API_KEY:
        from src.embedding import get_client
        client = get_client()
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=[prompt],
            config={"temperature": 0.0, "max_output_tokens": 512},
        )
        return response.text.strip()

    # Fallback to Ollama
    resp = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json={
            "model": config.OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 512},
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def combined_eval(
    question: str,
    answer: str,
    ground_truth: str,
    context: str,
) -> CombinedEvalResult:
    """
    Evaluate an answer on all 5 LLM metrics in a single API call.

    Returns CombinedEvalResult with all scores, or all -1.0 on failure.
    """
    prompt = get_prompt("combined_eval_prompt").format(
        question=question,
        ground_truth=ground_truth,
        answer=answer,
        context=context,
    )

    try:
        text = _call_eval_llm(prompt)

        # Strip markdown code fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)

        parsed = json.loads(text)

        def clamp(v):
            return round(max(0.0, min(1.0, float(v))), 4)

        return CombinedEvalResult(
            faithfulness=clamp(parsed["faithfulness"]),
            answer_relevancy=clamp(parsed["answer_relevancy"]),
            domain_appropriateness=clamp(parsed["domain_appropriateness"]),
            citation_accuracy=clamp(parsed["citation_accuracy"]),
            answer_completeness=clamp(parsed["answer_completeness"]),
            explanation=parsed.get("explanation", ""),
        )

    except Exception as e:
        return CombinedEvalResult(
            faithfulness=-1.0,
            answer_relevancy=-1.0,
            domain_appropriateness=-1.0,
            citation_accuracy=-1.0,
            answer_completeness=-1.0,
            explanation=f"Combined eval failed: {e}",
        )
