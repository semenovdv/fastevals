from pathlib import Path

from fastevals.config import RunConfig
from fastevals.models import RunResult
from fastevals.report import render_html_report, save_report


def make_result(**overrides) -> RunResult:
    defaults: dict = {
        "provider": "openai",
        "model": "gpt-test",
        "reasoning_effort": "none",
        "output": "Hello",
        "latency_ms": 1200.0,
        "time_to_first_token_ms": None,
        "input_tokens": 10,
        "output_tokens": 5,
        "reasoning_tokens": 1,
        "cached_tokens": 2,
        "input_cost_usd": 0.00001,
        "output_cost_usd": 0.00002,
        "reasoning_cost_usd": 0.0,
        "cached_cost_usd": 0.0,
        "tokens_per_second": 4.2,
        "finish_reason": "stop",
        "response_id": "resp-1",
    }
    defaults.update(overrides)
    return RunResult(**defaults)


def test_render_html_report_includes_metrics_and_charts():
    results = [make_result()]
    config = RunConfig(prompt="HI")
    html = render_html_report(config, results, "2026-08-18T00:00:00+00:00")
    assert "Latency" in html
    assert "Comparison table" in html
    assert "Detailed results" in html
    assert "chart.js" in html
    assert "resp-1" in html
    assert "4.20 tok/s" in html


def test_render_escapes_model_output():
    results = [make_result(output="<script>alert(1)</script>")]
    config = RunConfig(prompt="HI")
    html = render_html_report(config, results, "now")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


def test_save_report_writes_json_and_html(tmp_path: Path):
    results = [make_result(provider="mock", model="mock-model", reasoning_effort="off", output="ok")]
    config = RunConfig(prompt="test")
    json_path, html_path = save_report(config, results, tmp_path)
    assert json_path.exists()
    assert html_path is not None
    assert html_path.exists()
    html = html_path.read_text()
    assert "Run configuration" in html
    assert "mock / mock-model" in html
