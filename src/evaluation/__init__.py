"""
Evaluation module for Medical Imaging RAG.

Provides RAGAS-style metrics, LLM-as-Judge scoring, and HTML visualization.
"""
from src.evaluation.metrics import (
    MetricResult,
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from src.evaluation.llm_judge import JudgeResult, judge_answer
from src.evaluation.runner import (
    QuestionResult,
    EvaluationReport,
    run_evaluation,
)
from src.evaluation.visualization import generate_html_report

__all__ = [
    # Metrics
    "MetricResult",
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
    # LLM Judge
    "JudgeResult",
    "judge_answer",
    # Runner
    "QuestionResult",
    "EvaluationReport",
    "run_evaluation",
    # Visualization
    "generate_html_report",
]
