"""Standalone single-file HTML report generation."""

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import RunConfig
from .models import RunResult

__all__ = ["save_report"]


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _result_label(row: RunResult) -> str:
    return f"{row.provider}/{row.model} ({row.reasoning_effort or 'off'})"


def _format_usd(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 0.0001:
        return f"${value:.2e}"
    return f"${value:.6f}"


def _format_number(value: float | int | None, suffix: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:,.2f}{suffix}"
    return f"{value:,}{suffix}"


def _status_badge(row: RunResult) -> tuple[str, str]:
    if row.error:
        return "error", "Error"
    return "ok", "OK"


def _output_body(row: RunResult) -> str:
    if row.error:
        return _escape(str(row.error))
    if isinstance(row.output, (dict, list)):
        return _escape(json.dumps(row.output, ensure_ascii=False, indent=2))
    return _escape(str(row.output or ""))


def _render_config_items(config: RunConfig) -> str:
    items = [
        ("Prompt", config.prompt),
        ("File", config.file),
        ("Image", config.image),
        ("Structured output", json.dumps(config.structured_output, ensure_ascii=False, indent=2) if config.structured_output else None),
    ]
    return "".join(
        f"<div class='config-item'><span class='config-key'>{_escape(label)}</span>"
        f"<div class='config-value'>{_escape(str(value)) if value else '<span class=\"muted\">—</span>'}</div></div>"
        for label, value in items
    )


def _render_summary_cards(results: list[RunResult]) -> str:
    ok_count = sum(1 for row in results if row.ok)
    total_cost = sum(row.total_cost_usd or 0 for row in results)
    latencies = [row.latency_ms for row in results if row.latency_ms is not None and row.ok]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    total_tokens = sum((row.input_tokens or 0) + (row.output_tokens or 0) + (row.reasoning_tokens or 0) for row in results)
    cards = [
        ("Models", str(len(results)), "Runs in this matrix"),
        ("Success", f"{ok_count}/{len(results)}", "Completed without error"),
        ("Total cost", _format_usd(total_cost), "Sum across all runs"),
        ("Avg latency", _format_number(avg_latency, " ms"), "Mean end-to-end time"),
        ("Total tokens", _format_number(total_tokens), "Input + output + reasoning"),
    ]
    return "".join(
        f"<div class='card stat-card'><div class='stat-label'>{_escape(label)}</div>"
        f"<div class='stat-value'>{_escape(value)}</div>"
        f"<div class='stat-hint'>{_escape(hint)}</div></div>"
        for label, value, hint in cards
    )


def _render_comparison_table(results: list[RunResult]) -> str:
    rows = []
    for row in results:
        status_class, status_text = _status_badge(row)
        rows.append(
            "<tr>"
            f"<td><span class='badge {status_class}'>{status_text}</span></td>"
            f"<td>{_escape(row.provider)}</td>"
            f"<td>{_escape(row.model)}</td>"
            f"<td>{_escape(row.reasoning_effort or 'off')}</td>"
            f"<td>{_format_number(row.latency_ms, ' ms')}</td>"
            f"<td>{_format_number(row.time_to_first_token_ms, ' ms')}</td>"
            f"<td>{_format_number(row.tokens_per_second, ' t/s')}</td>"
            f"<td>{_format_number(row.input_tokens)}</td>"
            f"<td>{_format_number(row.output_tokens)}</td>"
            f"<td>{_format_number(row.reasoning_tokens)}</td>"
            f"<td>{_format_number(row.cached_tokens)}</td>"
            f"<td>{_format_usd(row.total_cost_usd)}</td>"
            f"<td>{_escape(row.finish_reason or '—')}</td>"
            "</tr>"
        )
    return (
        "<table class='comparison-table'>"
        "<thead><tr>"
        "<th>Status</th><th>Provider</th><th>Model</th><th>Reasoning</th>"
        "<th>Latency</th><th>TTFT</th><th>Throughput</th>"
        "<th>In</th><th>Out</th><th>Reason</th><th>Cached</th>"
        "<th>Cost</th><th>Finish</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_result_cards(results: list[RunResult]) -> str:
    cards = []
    for index, row in enumerate(results, start=1):
        status_class, status_text = _status_badge(row)
        metrics = [
            ("Latency", _format_number(row.latency_ms, " ms")),
            ("TTFT", _format_number(row.time_to_first_token_ms, " ms")),
            ("Throughput", _format_number(row.tokens_per_second, " tok/s")),
            ("Input tokens", _format_number(row.input_tokens)),
            ("Output tokens", _format_number(row.output_tokens)),
            ("Reasoning tokens", _format_number(row.reasoning_tokens)),
            ("Cached tokens", _format_number(row.cached_tokens)),
            ("Input cost", _format_usd(row.input_cost_usd)),
            ("Output cost", _format_usd(row.output_cost_usd)),
            ("Reasoning cost", _format_usd(row.reasoning_cost_usd)),
            ("Cached cost", _format_usd(row.cached_cost_usd)),
            ("Total cost", _format_usd(row.total_cost_usd)),
            ("Finish reason", row.finish_reason or "—"),
            ("Response ID", row.response_id or "—"),
        ]
        metric_grid = "".join(
            f"<div class='metric'><span class='metric-key'>{_escape(label)}</span>"
            f"<span class='metric-value'>{_escape(str(value))}</span></div>"
            for label, value in metrics
        )
        cards.append(
            f"<article class='result-card {status_class}'>"
            f"<div class='result-head'>"
            f"<div><h2>#{index} {_escape(row.provider)} / {_escape(row.model)}</h2>"
            f"<p class='meta'>Reasoning effort: {_escape(row.reasoning_effort or 'off')}</p></div>"
            f"<span class='badge {status_class}'>{status_text}</span>"
            f"</div>"
            f"<div class='metric-grid'>{metric_grid}</div>"
            f"<pre class='output-block'>{_output_body(row)}</pre>"
            f"</article>"
        )
    return "".join(cards)


def _chart_payload(results: list[RunResult]) -> dict[str, Any]:
    labels = [_result_label(row) for row in results]
    return {
        "labels": labels,
        "latency_ms": [row.latency_ms or 0 for row in results],
        "tokens_per_second": [row.tokens_per_second or 0 for row in results],
        "input_tokens": [row.input_tokens or 0 for row in results],
        "output_tokens": [row.output_tokens or 0 for row in results],
        "reasoning_tokens": [row.reasoning_tokens or 0 for row in results],
        "cached_tokens": [row.cached_tokens or 0 for row in results],
        "input_cost_usd": [row.input_cost_usd or 0 for row in results],
        "output_cost_usd": [row.output_cost_usd or 0 for row in results],
        "reasoning_cost_usd": [row.reasoning_cost_usd or 0 for row in results],
        "cached_cost_usd": [row.cached_cost_usd or 0 for row in results],
        "total_cost_usd": [row.total_cost_usd or 0 for row in results],
        "statuses": ["error" if row.error else "ok" for row in results],
    }


def render_html_report(config: RunConfig, results: list[RunResult], created_at: str) -> str:
    chart_data = json.dumps(_chart_payload(results), ensure_ascii=False)
    ok_count = sum(1 for row in results if row.ok)
    overall_status = "ok" if ok_count == len(results) else ("partial" if ok_count else "error")
    overall_label = {"ok": "All passed", "partial": "Partial success", "error": "All failed"}[overall_status]
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>fasteval report</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #eef2f8;
      --panel: #ffffff;
      --text: #152033;
      --muted: #667085;
      --line: #dbe3ef;
      --accent: #2563eb;
      --ok: #15803d;
      --ok-bg: #ecfdf3;
      --error: #b42318;
      --error-bg: #fef3f2;
      --warn: #b54708;
      --warn-bg: #fffaeb;
      --shadow: 0 10px 30px rgba(21, 32, 51, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font: 15px/1.5 Inter, ui-sans-serif, system-ui, sans-serif;
      color: var(--text);
      background: linear-gradient(180deg, #f8fbff 0%, var(--bg) 220px, var(--bg) 100%);
    }}
    .page {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    .hero, .card, .result-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
    }}
    .hero {{
      padding: 28px;
      margin-bottom: 20px;
    }}
    .hero-top {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      flex-wrap: wrap;
    }}
    h1, h2 {{ margin: 0 0 8px; }}
    .meta {{ color: var(--muted); margin: 0; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }}
    .badge.ok {{ background: var(--ok-bg); color: var(--ok); }}
    .badge.error {{ background: var(--error-bg); color: var(--error); }}
    .badge.partial {{ background: var(--warn-bg); color: var(--warn); }}
    .grid {{
      display: grid;
      gap: 16px;
    }}
    .stats-grid {{
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      margin-bottom: 20px;
    }}
    .charts-grid {{
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      margin-bottom: 20px;
    }}
    .card {{
      padding: 20px;
    }}
    .stat-card .stat-label {{
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .stat-card .stat-value {{
      font-size: 28px;
      font-weight: 700;
      margin: 8px 0 4px;
    }}
    .stat-card .stat-hint {{
      color: var(--muted);
      font-size: 13px;
    }}
    .section-title {{
      margin: 0 0 14px;
      font-size: 18px;
    }}
    .config-grid {{
      display: grid;
      gap: 14px;
    }}
    .config-item {{
      display: grid;
      gap: 6px;
    }}
    .config-key {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 700;
    }}
    .config-value {{
      white-space: pre-wrap;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 14px;
    }}
    .muted {{ color: var(--muted); }}
    .comparison-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    .comparison-table th, .comparison-table td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    .comparison-table th {{
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .comparison-table tbody tr:hover {{
      background: #f8fafc;
    }}
    .table-wrap {{
      overflow: auto;
    }}
    .result-card {{
      padding: 22px;
      margin-bottom: 16px;
    }}
    .result-card.error {{
      border-color: #fecdca;
    }}
    .result-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 16px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .metric {{
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
    }}
    .metric-key {{
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 4px;
    }}
    .metric-value {{
      font-weight: 600;
      word-break: break-word;
    }}
    .output-block {{
      white-space: pre-wrap;
      margin: 0;
      background: #101827;
      color: #e8eef8;
      border-radius: 12px;
      padding: 16px;
      overflow: auto;
    }}
    canvas {{
      width: 100% !important;
      height: 280px !important;
    }}
  </style>
</head>
<body>
  <div class="page">
    <header class="hero">
      <div class="hero-top">
        <div>
          <h1>fasteval report</h1>
          <p class="meta">{len(results)} model run(s) · {_escape(created_at)}</p>
        </div>
        <span class="badge {overall_status}">{overall_label}</span>
      </div>
    </header>

    <section class="grid stats-grid">
      {_render_summary_cards(results)}
    </section>

    <section class="card" style="margin-bottom:20px">
      <h2 class="section-title">Run configuration</h2>
      <div class="config-grid">{_render_config_items(config)}</div>
    </section>

    <section class="grid charts-grid">
      <div class="card"><h2 class="section-title">Latency</h2><canvas id="latencyChart"></canvas></div>
      <div class="card"><h2 class="section-title">Throughput</h2><canvas id="throughputChart"></canvas></div>
      <div class="card"><h2 class="section-title">Token usage</h2><canvas id="tokensChart"></canvas></div>
      <div class="card"><h2 class="section-title">Cost breakdown</h2><canvas id="costChart"></canvas></div>
    </section>

    <section class="card" style="margin-bottom:20px">
      <h2 class="section-title">Comparison table</h2>
      <div class="table-wrap">{_render_comparison_table(results)}</div>
    </section>

    <section>
      <h2 class="section-title">Detailed results</h2>
      {_render_result_cards(results)}
    </section>
  </div>

  <script>
    const chartData = {chart_data};
    const statusColors = chartData.statuses.map(status => status === "ok" ? "#2563eb" : "#ef4444");

    Chart.defaults.font.family = "Inter, ui-sans-serif, system-ui, sans-serif";
    Chart.defaults.color = "#667085";

    new Chart(document.getElementById("latencyChart"), {{
      type: "bar",
      data: {{
        labels: chartData.labels,
        datasets: [{{
          label: "Latency (ms)",
          data: chartData.latency_ms,
          backgroundColor: statusColors,
          borderRadius: 8,
        }}],
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: "ms" }} }} }},
      }},
    }});

    new Chart(document.getElementById("throughputChart"), {{
      type: "bar",
      data: {{
        labels: chartData.labels,
        datasets: [{{
          label: "Tokens / second",
          data: chartData.tokens_per_second,
          backgroundColor: "#059669",
          borderRadius: 8,
        }}],
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{ y: {{ beginAtZero: true, title: {{ display: true, text: "tok/s" }} }} }},
      }},
    }});

    new Chart(document.getElementById("tokensChart"), {{
      type: "bar",
      data: {{
        labels: chartData.labels,
        datasets: [
          {{ label: "Input", data: chartData.input_tokens, backgroundColor: "#2563eb", stack: "tokens" }},
          {{ label: "Output", data: chartData.output_tokens, backgroundColor: "#7c3aed", stack: "tokens" }},
          {{ label: "Reasoning", data: chartData.reasoning_tokens, backgroundColor: "#d97706", stack: "tokens" }},
          {{ label: "Cached", data: chartData.cached_tokens, backgroundColor: "#94a3b8", stack: "tokens" }},
        ],
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
          x: {{ stacked: true }},
          y: {{ stacked: true, beginAtZero: true, title: {{ display: true, text: "tokens" }} }},
        }},
      }},
    }});

    new Chart(document.getElementById("costChart"), {{
      type: "bar",
      data: {{
        labels: chartData.labels,
        datasets: [
          {{ label: "Input", data: chartData.input_cost_usd, backgroundColor: "#2563eb", stack: "cost" }},
          {{ label: "Output", data: chartData.output_cost_usd, backgroundColor: "#7c3aed", stack: "cost" }},
          {{ label: "Reasoning", data: chartData.reasoning_cost_usd, backgroundColor: "#d97706", stack: "cost" }},
          {{ label: "Cached", data: chartData.cached_cost_usd, backgroundColor: "#94a3b8", stack: "cost" }},
        ],
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
          x: {{ stacked: true }},
          y: {{ stacked: true, beginAtZero: true, title: {{ display: true, text: "USD" }} }},
        }},
      }},
    }});
  </script>
</body>
</html>"""


def save_report(
    config: RunConfig,
    results: list[RunResult],
    output_dir: str | Path,
    include_html: bool = True,
) -> tuple[Path, Path | None]:
    """Persist ``run.json`` and the standalone HTML report for one run."""
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "prompt": config.prompt,
        "file": config.file,
        "image": config.image,
        "structured_output": config.structured_output,
        "created_at": created_at,
        "results": [asdict(row) for row in results],
    }
    json_path = run_dir / "run.json"
    html_path = run_dir / "report.html"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    if include_html:
        html_path.write_text(render_html_report(config, results, created_at))
    return json_path, html_path if include_html else None
