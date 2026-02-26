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
        if config.GENERATION_BACKEND == "ollama":
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
            text = resp.json()["response"].strip()
        else:
            from src.embedding import get_client
            client = get_client()
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=[prompt],
                config={"temperature": 0.0, "max_output_tokens": 512},
            )
            text = response.text.strip()

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
