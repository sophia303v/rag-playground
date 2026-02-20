"""
LLM-as-Judge evaluation for medical domain quality.

Uses a single LLM call per question to evaluate 3 criteria:
- Medical Appropriateness: correct terminology, clinically accurate
- Citation Accuracy: sources cited and match content
- Answer Completeness: covers key points from ground truth
"""
from dataclasses import dataclass
import json
import re

import requests

import config


@dataclass
class JudgeResult:
    """Result of LLM judge evaluation for a single question."""
    medical_appropriateness: float  # 0.0 to 1.0
    citation_accuracy: float        # 0.0 to 1.0
    answer_completeness: float      # 0.0 to 1.0
    explanation: str

    @property
    def average_score(self) -> float:
        scores = [
            self.medical_appropriateness,
            self.citation_accuracy,
            self.answer_completeness,
        ]
        valid = [s for s in scores if s >= 0]
        return round(sum(valid) / len(valid), 4) if valid else -1.0


JUDGE_PROMPT = """You are a senior radiologist evaluating an AI medical assistant's answer.

Given the QUESTION, GROUND TRUTH answer, AI ANSWER, and the RETRIEVED CONTEXT,
score the AI answer on three criteria.

QUESTION:
{question}

GROUND TRUTH (reference answer):
{ground_truth}

AI ANSWER:
{answer}

RETRIEVED CONTEXT:
{context}

Score each criterion from 0.0 to 1.0:

1. **Medical Appropriateness** (0-1): Does the answer use correct medical
   terminology? Is it clinically accurate? Would a radiologist find it acceptable?
   - 1.0: Terminology and clinical accuracy are excellent
   - 0.5: Some terminology issues or minor inaccuracies
   - 0.0: Seriously incorrect medical information

2. **Citation Accuracy** (0-1): Does the answer cite its sources (e.g., [Source 1],
   report UIDs)? Do the citations match the actual content of the retrieved documents?
   - 1.0: All claims are properly cited with correct source references
   - 0.5: Some citations present but incomplete or partially incorrect
   - 0.0: No citations or completely wrong citations

3. **Answer Completeness** (0-1): Does the answer cover the key points from
   the ground truth? Are important findings mentioned?
   - 1.0: All key points from ground truth are covered
   - 0.5: Some key points covered, some missing
   - 0.0: Major key points missing entirely

Respond in JSON format only:
{{
  "medical_appropriateness": <float>,
  "citation_accuracy": <float>,
  "answer_completeness": <float>,
  "explanation": "<brief justification covering all three criteria>"
}}"""


def judge_answer(
    question: str,
    answer: str,
    ground_truth: str,
    context: str,
) -> JudgeResult:
    """
    Evaluate an answer using LLM-as-Judge with 3 medical domain criteria.

    Args:
        question: The original question
        answer: The AI-generated answer to evaluate
        ground_truth: The reference/correct answer
        context: The retrieved context used to generate the answer

    Returns:
        JudgeResult with scores for each criterion
    """
    prompt = JUDGE_PROMPT.format(
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

        return JudgeResult(
            medical_appropriateness=clamp(parsed["medical_appropriateness"]),
            citation_accuracy=clamp(parsed["citation_accuracy"]),
            answer_completeness=clamp(parsed["answer_completeness"]),
            explanation=parsed.get("explanation", ""),
        )

    except Exception as e:
        return JudgeResult(
            medical_appropriateness=-1.0,
            citation_accuracy=-1.0,
            answer_completeness=-1.0,
            explanation=f"LLM judge failed: {e}",
        )
