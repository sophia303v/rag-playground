"""
CLI entry point for running RAG evaluation.

Usage:
    python run_eval.py                   # Full evaluation (needs Gemini API key)
    python run_eval.py --retrieval-only  # Retrieval metrics only (no API key needed)
    python run_eval.py --qa data/golden_qa.json  # Custom QA file
"""
import argparse
from pathlib import Path

import config
from src.evaluation.runner import run_evaluation
from src.evaluation.visualization import generate_html_report


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument(
        "--qa", type=str, default=None,
        help="Path to golden QA JSON file (default: data/golden_qa.json)",
    )
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="Only run retrieval metrics (no Gemini API key needed)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for results (default: data/)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    args = parser.parse_args()

    qa_path = Path(args.qa) if args.qa else None
    output_dir = Path(args.output_dir) if args.output_dir else config.DATA_DIR
    use_llm = not args.retrieval_only

    if use_llm and not config.GEMINI_API_KEY:
        print("Warning: No GEMINI_API_KEY set. Falling back to retrieval-only mode.")
        print("Set GEMINI_API_KEY in .env for full evaluation (faithfulness, relevancy, LLM judge).\n")
        use_llm = False

    # Run evaluation
    report = run_evaluation(
        golden_qa_path=qa_path,
        use_llm_metrics=use_llm,
        verbose=not args.quiet,
    )

    # Save results
    json_path = output_dir / "eval_results.json"
    report.save_json(json_path)

    # Generate HTML report
    html_path = output_dir / "eval_report.html"
    generate_html_report(report, html_path)

    print(f"\nDone! Open {html_path} in a browser to view the report.")


if __name__ == "__main__":
    main()
