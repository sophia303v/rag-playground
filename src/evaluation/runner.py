"""
Evaluation runner: orchestrates RAGAS metrics and LLM judge across all golden QA pairs.

Produces an EvaluationReport with per-question and aggregate scores.
"""
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import config
from src.rag_pipeline import MedicalImagingRAG
from src.evaluation.metrics import (
    MetricResult,
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from src.evaluation.llm_judge import JudgeResult, judge_answer


@dataclass
class QuestionResult:
    """All evaluation results for a single question."""
    question_id: str
    question: str
    category: str
    difficulty: str
    answer: str
    ground_truth: str
    retrieved_uids: list[str]
    relevant_uids: list[str]
    # RAGAS metrics
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float
    # LLM judge metrics
    medical_appropriateness: float
    citation_accuracy: float
    answer_completeness: float

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "category": self.category,
            "difficulty": self.difficulty,
            "answer": self.answer,
            "ground_truth": self.ground_truth,
            "retrieved_uids": self.retrieved_uids,
            "relevant_uids": self.relevant_uids,
            "scores": {
                "context_precision": self.context_precision,
                "context_recall": self.context_recall,
                "faithfulness": self.faithfulness,
                "answer_relevancy": self.answer_relevancy,
                "medical_appropriateness": self.medical_appropriateness,
                "citation_accuracy": self.citation_accuracy,
                "answer_completeness": self.answer_completeness,
            },
        }


@dataclass
class EvaluationReport:
    """Complete evaluation report across all questions."""
    results: list[QuestionResult] = field(default_factory=list)
    run_time_seconds: float = 0.0

    def aggregate_scores(self) -> dict[str, dict[str, float]]:
        """Compute mean, min, max for each metric across all questions."""
        metric_names = [
            "context_precision", "context_recall", "faithfulness",
            "answer_relevancy", "medical_appropriateness",
            "citation_accuracy", "answer_completeness",
        ]
        agg = {}
        for name in metric_names:
            values = [
                getattr(r, name) for r in self.results
                if getattr(r, name) >= 0  # skip failed metrics (-1)
            ]
            if values:
                agg[name] = {
                    "mean": round(sum(values) / len(values), 4),
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                    "count": len(values),
                }
            else:
                agg[name] = {"mean": -1.0, "min": -1.0, "max": -1.0, "count": 0}
        return agg

    def scores_by_category(self) -> dict[str, dict[str, float]]:
        """Compute mean scores grouped by question category."""
        categories: dict[str, list[QuestionResult]] = {}
        for r in self.results:
            categories.setdefault(r.category, []).append(r)

        metric_names = [
            "context_precision", "context_recall", "faithfulness",
            "answer_relevancy", "medical_appropriateness",
            "citation_accuracy", "answer_completeness",
        ]
        result = {}
        for cat, items in sorted(categories.items()):
            cat_scores = {}
            for name in metric_names:
                values = [getattr(r, name) for r in items if getattr(r, name) >= 0]
                cat_scores[name] = round(sum(values) / len(values), 4) if values else -1.0
            result[cat] = cat_scores
        return result

    def to_dict(self) -> dict:
        return {
            "run_time_seconds": round(self.run_time_seconds, 2),
            "num_questions": len(self.results),
            "aggregate_scores": self.aggregate_scores(),
            "scores_by_category": self.scores_by_category(),
            "results": [r.to_dict() for r in self.results],
        }

    def save_json(self, path: Path):
        """Save report to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Results saved to {path}")


def _extract_uids(metadatas: list[dict]) -> list[str]:
    """Extract unique report UIDs from retrieval metadata."""
    uids = []
    seen = set()
    for meta in metadatas:
        uid = meta.get("uid", "")
        if uid and uid not in seen:
            uids.append(uid)
            seen.add(uid)
    return uids


def run_evaluation(
    golden_qa_path: Path | None = None,
    use_llm_metrics: bool = True,
    verbose: bool = True,
) -> EvaluationReport:
    """
    Run full evaluation on the golden QA dataset.

    Args:
        golden_qa_path: Path to golden QA JSON file
        use_llm_metrics: Whether to run LLM-based metrics (faithfulness,
                         answer_relevancy, LLM judge). Set False to only
                         run retrieval metrics without an API key.
        verbose: Print progress

    Returns:
        EvaluationReport with all results
    """
    if golden_qa_path is None:
        golden_qa_path = config.DATA_DIR / "golden_qa.json"

    with open(golden_qa_path) as f:
        qa_pairs = json.load(f)

    if verbose:
        print(f"Loaded {len(qa_pairs)} QA pairs from {golden_qa_path}")

    # Initialize RAG pipeline
    rag = MedicalImagingRAG()
    if not rag._is_ingested:
        if verbose:
            print("No index found. Running ingestion...")
        rag.ingest()

    report = EvaluationReport()
    start_time = time.time()

    for i, qa in enumerate(qa_pairs):
        qid = qa["id"]
        question = qa["question"]
        ground_truth = qa["ground_truth_answer"]
        relevant_uids = qa["relevant_report_uids"]
        category = qa["category"]
        difficulty = qa["difficulty"]

        if verbose:
            print(f"\n[{i+1}/{len(qa_pairs)}] {qid}: {question[:60]}...")

        # Run RAG query
        try:
            gen_result = rag.query(question)
            answer = gen_result.answer
            retrieval = gen_result.retrieval
            retrieved_uids = _extract_uids(retrieval.metadatas)
            context_text = retrieval.context
        except Exception as e:
            if verbose:
                print(f"  Query failed: {e}")
            answer = f"[Query failed: {e}]"
            retrieved_uids = []
            context_text = ""

        # --- Retrieval metrics (always run) ---
        cp = context_precision(retrieved_uids, relevant_uids)
        cr = context_recall(retrieved_uids, relevant_uids)

        if verbose:
            print(f"  Context Precision: {cp.score:.2f}  Recall: {cr.score:.2f}")

        # --- LLM metrics (optional) ---
        faith_score = -1.0
        relevancy_score = -1.0
        med_approp = -1.0
        cite_acc = -1.0
        completeness = -1.0

        if use_llm_metrics:
            faith = faithfulness(question, answer, context_text)
            faith_score = faith.score

            relevancy = answer_relevancy(question, answer)
            relevancy_score = relevancy.score

            judge = judge_answer(question, answer, ground_truth, context_text)
            med_approp = judge.medical_appropriateness
            cite_acc = judge.citation_accuracy
            completeness = judge.answer_completeness

            if verbose:
                print(f"  Faithfulness: {faith_score:.2f}  Relevancy: {relevancy_score:.2f}")
                print(f"  Medical: {med_approp:.2f}  Citation: {cite_acc:.2f}  Completeness: {completeness:.2f}")

        result = QuestionResult(
            question_id=qid,
            question=question,
            category=category,
            difficulty=difficulty,
            answer=answer,
            ground_truth=ground_truth,
            retrieved_uids=retrieved_uids,
            relevant_uids=relevant_uids,
            context_precision=cp.score,
            context_recall=cr.score,
            faithfulness=faith_score,
            answer_relevancy=relevancy_score,
            medical_appropriateness=med_approp,
            citation_accuracy=cite_acc,
            answer_completeness=completeness,
        )
        report.results.append(result)

    report.run_time_seconds = time.time() - start_time

    if verbose:
        print(f"\n{'='*60}")
        print(f"Evaluation complete in {report.run_time_seconds:.1f}s")
        print(f"{'='*60}")
        agg = report.aggregate_scores()
        for name, stats in agg.items():
            if stats["count"] > 0:
                print(f"  {name:30s}  mean={stats['mean']:.3f}  "
                      f"min={stats['min']:.3f}  max={stats['max']:.3f}")
            else:
                print(f"  {name:30s}  (no data)")

    return report
