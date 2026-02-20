"""
Evaluation report visualization: generates a self-contained HTML file
with interactive Plotly charts.

Charts included:
1. Radar chart — all 7 metrics at a glance
2. Bar chart — scores grouped by question category
3. Summary table — mean, min, max per metric
4. Detail table — per-question scores with color coding
"""
import json
from pathlib import Path

from src.evaluation.runner import EvaluationReport


# Metric display names (short labels for charts)
METRIC_LABELS = {
    "context_precision": "Ctx Precision",
    "context_recall": "Ctx Recall",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Relevancy",
    "medical_appropriateness": "Medical",
    "citation_accuracy": "Citation",
    "answer_completeness": "Completeness",
}

METRIC_ORDER = list(METRIC_LABELS.keys())


def generate_html_report(
    report: EvaluationReport,
    output_path: Path,
) -> Path:
    """
    Generate a self-contained HTML report with Plotly charts.

    Args:
        report: EvaluationReport with evaluation results
        output_path: Where to save the HTML file

    Returns:
        Path to the generated HTML file
    """
    agg = report.aggregate_scores()
    by_cat = report.scores_by_category()

    # --- Build chart data ---
    radar_data = _build_radar_data(agg)
    bar_data = _build_bar_data(by_cat)
    summary_rows = _build_summary_table(agg)
    detail_rows = _build_detail_table(report)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RAG Evaluation Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #f5f7fa; color: #1a1a2e; padding: 2rem; }}
  .container {{ max-width: 1200px; margin: 0 auto; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.3rem; }}
  .subtitle {{ color: #666; margin-bottom: 2rem; font-size: 0.95rem; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }}
  .card {{ background: #fff; border-radius: 12px; padding: 1.5rem;
           box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  .card h2 {{ font-size: 1.1rem; margin-bottom: 1rem; color: #16213e; }}
  .full-width {{ grid-column: 1 / -1; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ background: #16213e; color: #fff; padding: 0.6rem 0.5rem; text-align: left;
       position: sticky; top: 0; }}
  td {{ padding: 0.5rem; border-bottom: 1px solid #e8e8e8; }}
  tr:hover {{ background: #f0f4ff; }}
  .score-cell {{ text-align: center; font-weight: 600; border-radius: 4px; }}
  .score-high {{ background: #d4edda; color: #155724; }}
  .score-mid {{ background: #fff3cd; color: #856404; }}
  .score-low {{ background: #f8d7da; color: #721c24; }}
  .score-na {{ background: #e2e3e5; color: #6c757d; }}
  .stats {{ display: flex; gap: 1.5rem; margin-bottom: 1.5rem; flex-wrap: wrap; }}
  .stat-box {{ background: #fff; border-radius: 10px; padding: 1rem 1.5rem;
               box-shadow: 0 2px 8px rgba(0,0,0,0.08); text-align: center; min-width: 140px; }}
  .stat-box .value {{ font-size: 1.6rem; font-weight: 700; color: #0f3460; }}
  .stat-box .label {{ font-size: 0.8rem; color: #888; margin-top: 0.2rem; }}
  .detail-scroll {{ max-height: 500px; overflow-y: auto; }}
</style>
</head>
<body>
<div class="container">
  <h1>Medical Imaging RAG — Evaluation Report</h1>
  <p class="subtitle">{len(report.results)} questions evaluated in {report.run_time_seconds:.1f}s</p>

  <!-- Quick stats -->
  <div class="stats">
    {_build_stat_boxes(agg, report)}
  </div>

  <!-- Charts row -->
  <div class="grid">
    <div class="card">
      <h2>Overall Metrics (Radar)</h2>
      <div id="radar-chart"></div>
    </div>
    <div class="card">
      <h2>Scores by Category</h2>
      <div id="bar-chart"></div>
    </div>
  </div>

  <!-- Summary table -->
  <div class="card full-width" style="margin-bottom:1.5rem;">
    <h2>Metric Summary</h2>
    <table>
      <tr><th>Metric</th><th>Mean</th><th>Min</th><th>Max</th><th>Count</th></tr>
      {summary_rows}
    </table>
  </div>

  <!-- Detail table -->
  <div class="card full-width">
    <h2>Per-Question Results</h2>
    <div class="detail-scroll">
    <table>
      <tr>
        <th>ID</th><th>Category</th><th>Difficulty</th>
        <th>Ctx Prec</th><th>Ctx Rec</th><th>Faith</th><th>Relev</th>
        <th>Medical</th><th>Citation</th><th>Complete</th>
      </tr>
      {detail_rows}
    </table>
    </div>
  </div>
</div>

<script>
// Radar chart
{radar_data}

// Bar chart
{bar_data}
</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"HTML report saved to {output_path}")
    return output_path


def _score_class(score: float) -> str:
    """CSS class for color coding a score."""
    if score < 0:
        return "score-na"
    if score >= 0.7:
        return "score-high"
    if score >= 0.4:
        return "score-mid"
    return "score-low"


def _fmt(score: float) -> str:
    """Format score for display."""
    if score < 0:
        return "N/A"
    return f"{score:.2f}"


def _build_stat_boxes(agg: dict, report: EvaluationReport) -> str:
    """Build quick-stat HTML boxes."""
    # Overall mean across all metrics
    all_means = [v["mean"] for v in agg.values() if v["mean"] >= 0]
    overall = sum(all_means) / len(all_means) if all_means else 0

    retrieval_metrics = ["context_precision", "context_recall"]
    ret_means = [agg[m]["mean"] for m in retrieval_metrics if agg[m]["mean"] >= 0]
    retrieval_avg = sum(ret_means) / len(ret_means) if ret_means else 0

    gen_metrics = ["faithfulness", "answer_relevancy"]
    gen_means = [agg[m]["mean"] for m in gen_metrics if agg[m]["mean"] >= 0]
    gen_avg = sum(gen_means) / len(gen_means) if gen_means else 0

    judge_metrics = ["medical_appropriateness", "citation_accuracy", "answer_completeness"]
    judge_means = [agg[m]["mean"] for m in judge_metrics if agg[m]["mean"] >= 0]
    judge_avg = sum(judge_means) / len(judge_means) if judge_means else 0

    return f"""
    <div class="stat-box"><div class="value">{overall:.2f}</div><div class="label">Overall Score</div></div>
    <div class="stat-box"><div class="value">{retrieval_avg:.2f}</div><div class="label">Retrieval Quality</div></div>
    <div class="stat-box"><div class="value">{gen_avg:.2f}</div><div class="label">Generation Quality</div></div>
    <div class="stat-box"><div class="value">{judge_avg:.2f}</div><div class="label">Medical Judge</div></div>
    <div class="stat-box"><div class="value">{len(report.results)}</div><div class="label">Questions</div></div>
    """


def _build_radar_data(agg: dict) -> str:
    """Build Plotly radar chart JS."""
    labels = [METRIC_LABELS[m] for m in METRIC_ORDER]
    values = [max(0, agg[m]["mean"]) for m in METRIC_ORDER]
    # Close the polygon
    labels_js = json.dumps(labels + [labels[0]])
    values_js = json.dumps(values + [values[0]])

    return f"""
Plotly.newPlot('radar-chart', [{{
  type: 'scatterpolar',
  r: {values_js},
  theta: {labels_js},
  fill: 'toself',
  fillcolor: 'rgba(15,52,96,0.15)',
  line: {{ color: '#0f3460', width: 2 }},
  marker: {{ size: 6, color: '#0f3460' }}
}}], {{
  polar: {{
    radialaxis: {{ visible: true, range: [0, 1], tickvals: [0.2, 0.4, 0.6, 0.8, 1.0] }}
  }},
  margin: {{ t: 30, b: 30, l: 60, r: 60 }},
  height: 350,
  showlegend: false
}}, {{ responsive: true }});
"""


def _build_bar_data(by_cat: dict) -> str:
    """Build Plotly grouped bar chart JS."""
    categories = list(by_cat.keys())
    traces = []
    colors = [
        "#0f3460", "#1a73e8", "#4CAF50", "#FF9800",
        "#9C27B0", "#E91E63", "#00BCD4",
    ]
    for i, metric in enumerate(METRIC_ORDER):
        values = [max(0, by_cat[cat].get(metric, 0)) for cat in categories]
        traces.append({
            "x": categories,
            "y": values,
            "name": METRIC_LABELS[metric],
            "type": "bar",
            "marker": {"color": colors[i % len(colors)]},
        })

    return f"""
Plotly.newPlot('bar-chart', {json.dumps(traces)}, {{
  barmode: 'group',
  yaxis: {{ range: [0, 1], title: 'Score' }},
  xaxis: {{ title: 'Category' }},
  margin: {{ t: 30, b: 60, l: 50, r: 20 }},
  height: 350,
  legend: {{ orientation: 'h', y: -0.25, x: 0.5, xanchor: 'center', font: {{ size: 10 }} }}
}}, {{ responsive: true }});
"""


def _build_summary_table(agg: dict) -> str:
    """Build HTML rows for the summary table."""
    rows = []
    for metric in METRIC_ORDER:
        stats = agg[metric]
        label = METRIC_LABELS[metric]
        mean_cls = _score_class(stats["mean"])
        rows.append(
            f"<tr><td>{label}</td>"
            f"<td class='score-cell {mean_cls}'>{_fmt(stats['mean'])}</td>"
            f"<td class='score-cell'>{_fmt(stats['min'])}</td>"
            f"<td class='score-cell'>{_fmt(stats['max'])}</td>"
            f"<td>{stats['count']}</td></tr>"
        )
    return "\n".join(rows)


def _build_detail_table(report: EvaluationReport) -> str:
    """Build HTML rows for the per-question detail table."""
    rows = []
    for r in report.results:
        scores = [
            r.context_precision, r.context_recall, r.faithfulness,
            r.answer_relevancy, r.medical_appropriateness,
            r.citation_accuracy, r.answer_completeness,
        ]
        cells = "".join(
            f"<td class='score-cell {_score_class(s)}'>{_fmt(s)}</td>"
            for s in scores
        )
        rows.append(
            f"<tr><td>{r.question_id}</td>"
            f"<td>{r.category}</td><td>{r.difficulty}</td>"
            f"{cells}</tr>"
        )
    return "\n".join(rows)
