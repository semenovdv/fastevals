import asyncio
import json
import tomllib
import uuid
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .llm import complete


@dataclass
class RunResult:
    provider: str
    model: str
    reasoning_effort: str

    output: Any

    time_to_first_token_ms: float | None = None
    latency_ms: float | None = None


    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None 
    cached_tokens: int | None = None

    input_cost_usd: float | None = None
    output_cost_usd: float | None = None
    reasoning_cost_usd: float | None = None
    cached_cost_usd: float | None = None

    tokens_per_second: float | None = None
        
    error: str | None = None
    finish_reason: str | None = None
    response_id: str | None = None

    @property
    def total_cost_usd(self) -> float | None:
        costs = (
            self.input_cost_usd,
            self.output_cost_usd,
            self.reasoning_cost_usd,
            self.cached_cost_usd,
        )
        known_costs = [cost for cost in costs if cost is not None]
        return sum(known_costs) if known_costs else None


async def _call_model(model: dict[str, Any], prompt: str, file_path: str | None = None, image_path: str | None = None):
    if model.get("type") == "mock" and "connector" not in model:
        model = {**model, "connector": "mock"}
    file_paths = [path for path in (file_path, image_path) if path]
    response = await asyncio.to_thread(complete, prompt, model, file_paths=file_paths or None)
    return response


def _expand_reasoning_efforts(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded = []
    for model in models:
        raw_efforts = model.get("reasoning_efforts")
        efforts = [item.strip() for item in raw_efforts.split("|") if item.strip()] if isinstance(raw_efforts, str) else [model.get("reasoning_effort", "off")]
        for effort in efforts:
            expanded.append({**model, "reasoning_effort": effort, "registry_id": f"{model.get('provider', 'unknown')}:{model.get('model', 'unknown')}:{effort}"})
    return expanded


def _load_registry(path: Path) -> dict[str, dict[str, Any]]:
    """Load the flat TOML registry keyed by ``provider:model``."""
    with path.open("rb") as config_file:
        data = tomllib.load(config_file)
    if not data:
        raise ValueError(f"Model registry is empty: {path}")
    invalid = [key for key, value in data.items() if not isinstance(value, dict) or "provider" not in value or "model" not in value]
    if invalid:
        raise ValueError(f"Invalid model registry entries: {', '.join(invalid)}")
    return data


async def run(config: dict[str, Any]) -> list[RunResult]:
    prompt = config["prompt"]
    file_path = config.get("file")
    image_path = config.get("image")
    registry_path = Path(__file__).resolve().parents[1] / "config" / "models.toml"
    model_registry = _load_registry(registry_path) if registry_path.exists() else {}
    
    requested = config.get("providers", set())
    if "all" in requested or not requested:
        requested = {model.get("provider") for model in (model_registry or {}).values()}
    models = [dict(model, id=model_id) for model_id, model in (model_registry or {}).items() if model.get("provider") in requested]
    models = _expand_reasoning_efforts(models)
    # Keep registry metadata out of the provider request while preserving it for reports.

    async def run_model(model: dict[str, Any]) -> RunResult:
        started = time.perf_counter()
        try:
            response = await _call_model(model, prompt, file_path=file_path, image_path=image_path)
            latency_ms = (time.perf_counter() - started) * 1000
            input_tokens = response.input_tokens or 0
            output_tokens = response.output_tokens or 0
            reasoning_tokens = response.reasoning_tokens or 0
            cached_tokens = response.cached_tokens or 0
            per_million = lambda key: (input_tokens if key == "input" else output_tokens if key == "output" else reasoning_tokens if key == "reasoning" else cached_tokens) / 1_000_000 * model.get(f"{key}_cost_usd_per_mtok", 0)
            return RunResult(provider=model.get("provider", "unknown"), model=model.get("model", model.get("name", "unknown")), reasoning_effort=model.get("reasoning_effort", "off"), output=response.text, input_tokens=input_tokens, output_tokens=output_tokens, reasoning_tokens=reasoning_tokens, cached_tokens=cached_tokens, input_cost_usd=per_million("input"), output_cost_usd=per_million("output"), reasoning_cost_usd=per_million("reasoning"), cached_cost_usd=per_million("cached"), finish_reason=response.finish_reason or "completed", response_id=response.response_id or "", error="", latency_ms=latency_ms, time_to_first_token_ms=latency_ms, tokens_per_second=(output_tokens / (latency_ms / 1000)) if latency_ms else 0)
        except Exception as exc:  # Keep the matrix report useful when one provider fails.
            return RunResult(provider=model.get("provider", "unknown"), model=model.get("model", model.get("name", "unknown")), reasoning_effort=model.get("reasoning_effort", "off"), output=None, latency_ms=(time.perf_counter() - started) * 1000, error=str(exc))

    return await asyncio.gather(*(run_model(model) for model in models))


def save_report(config: dict[str, Any], results: list[RunResult], output_dir: Path, include_html: bool = True) -> tuple[Path, Path | None]:
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"prompt": config["prompt"], "file": config.get("file"), "image": config.get("image"), "structured_output": config.get("structured_output"), "created_at": datetime.now(timezone.utc).isoformat(), "results": [asdict(row) for row in results]}
    json_path = run_dir / "run.json"
    html_path = run_dir / "report.html"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    rows = "".join(
        f"<article><h2>{_escape(row.provider)} / {_escape(row.model)}</h2><p class='meta'>{(row.latency_ms or 0):.0f} ms · reasoning: {_escape(row.reasoning_effort or 'off')}</p><pre>{_escape(str(row.error or row.output or ''))}</pre></article>"
        for row in results
    )
    if include_html:
        html_path.write_text(f"""<!doctype html><meta charset='utf-8'><title>fasteval report</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:40px auto;background:#f5f7fb;color:#172033}}header,article{{background:white;border:1px solid #e2e7f0;border-radius:14px;padding:22px;margin:16px 0;box-shadow:0 4px 20px #1720330b}}h1{{margin-top:0}}.prompt{{white-space:pre-wrap;background:#f0f4fa;padding:16px;border-radius:10px}}pre{{white-space:pre-wrap;line-height:1.5;background:#101827;color:#e8eef8;padding:18px;border-radius:10px;overflow:auto}}.meta{{color:#65738b}}</style>
<header><h1>fasteval report</h1><p class='meta'>{len(results)} model(s) · {payload['created_at']}</p><div class='prompt'>{_escape(config['prompt'])}</div></header>{rows}""")
    return json_path, html_path if include_html else None


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
