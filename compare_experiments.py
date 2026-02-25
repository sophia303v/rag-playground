"""
CLI tool: compare multiple experiment results side by side.

Usage:
    # Compare two experiments
    python compare_experiments.py experiments/2026-02-20_baseline experiments/2026-02-23_topk5

    # Compare with custom output path
    python compare_experiments.py exp1/ exp2/ exp3/ -o experiments/my_comparison.html

    # Compare with custom display names
    python compare_experiments.py exp1/ exp2/ --names "Baseline" "TOP_K=5"
"""
import argparse
import json
import sys
from pathlib import Path

# Direct file load to avoid pulling in the full pipeline (chromadb, etc.)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "visualization",
    Path(__file__).parent / "src" / "evaluation" / "visualization.py",
)
_viz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_viz)
generate_comparison_report = _viz.generate_comparison_report


def _load_experiment(exp_dir: Path, name: str | None = None) -> dict:
    """Load an experiment's results and config from a directory."""
    results_path = exp_dir / "eval_results.json"
    config_path = exp_dir / "config.yaml"

    if not results_path.exists():
        print(f"Error: {results_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    config_data = {}
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config_data = yaml.safe_load(f) or {}

    # Default name: directory name
    if name is None:
        name = exp_dir.name

    return {"name": name, "results": results, "config": config_data}


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple experiment results",
    )
    parser.add_argument(
        "experiments", nargs="+", type=str,
        help="Paths to experiment directories (each containing eval_results.json)",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None,
        help="Output HTML path (default: experiments/comparison.html)",
    )
    parser.add_argument(
        "--names", nargs="+", type=str, default=None,
        help="Display names for experiments (must match number of experiment paths)",
    )
    args = parser.parse_args()

    if len(args.experiments) < 2:
        print("Error: need at least 2 experiments to compare.", file=sys.stderr)
        sys.exit(1)

    if args.names and len(args.names) != len(args.experiments):
        print(
            f"Error: --names count ({len(args.names)}) doesn't match "
            f"experiment count ({len(args.experiments)})",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load experiments
    experiments = []
    for i, exp_path in enumerate(args.experiments):
        exp_dir = Path(exp_path)
        name = args.names[i] if args.names else None
        experiments.append(_load_experiment(exp_dir, name))
        print(f"Loaded: {experiments[-1]['name']}")

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        names_slug = "_vs_".join(e["name"] for e in experiments)
        output_path = Path("experiments") / f"comparison_{names_slug}.html"

    # Generate comparison report
    generate_comparison_report(experiments, output_path)
    print(f"\nDone! Open {output_path} in a browser to view the comparison.")


if __name__ == "__main__":
    main()
