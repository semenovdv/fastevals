"""MCP server exposing fastevals to AI assistants.

Run locally with ``fastevals-mcp`` (stdio transport) and register it from any
MCP client, e.g. Claude Desktop or Claude Code.
"""

import asyncio
from pathlib import Path
from typing import Any

from mcp.server.mcpserver.server import MCPServer

from .config import ALL_PROVIDERS, SUPPORTED_PROVIDERS, RunConfig
from .exceptions import FastEvalError
from .registry import default_registry_path, describe_registry, load_registry, select_specs
from .report import save_report
from .runner import run
from .structured import shorthand_to_schema
from .tags import load_tags, save_tag

__all__ = ["build_server", "main"]

mcp = MCPServer(
    name="fastevals",
    instructions=(
        "fastevals runs one prompt across a matrix of LLM models and providers, "
        "saves every response and returns a comparison summary with cost, "
        "latency and token metrics."
    ),
)


@mcp.tool()
async def run_evaluation(
    prompt: str = "",
    providers: str = ALL_PROVIDERS,
    models: str | None = None,
    tag: str | None = None,
    structured_output: str | None = None,
    dataset: str | None = None,
    file: str | None = None,
    image: str | None = None,
    nruns: int = 1,
    registry: str | None = None,
    out: str = "runs",
) -> dict[str, Any]:
    """Run an evaluation matrix and save a JSON + HTML report.

    Args:
        prompt: The task prompt (omit when ``dataset`` supplies prompts).
        providers: Pipe-separated provider list, e.g. ``openai|openrouter`` or ``all``.
        models: Optional pipe-separated selectors narrowing the matrix. Required
            form: ``provider/model[@efforts]`` using the exact model id, e.g.
            ``openai/gpt-5.6-luna@none,high`` or ``openai/gpt-5.6-sol@low``.
            The provider is mandatory — the same model string is often served
            by many providers. Use the list_models tool to discover ids.
            Mutually exclusive with ``tag``.
        tag: Optional name of a saved model suite (see the add_tag and
            list_tags tools). Preferred over typing models every time.
        structured_output: Optional compact schema like ``name:str,age:int``.
        dataset: Optional JSONL/CSV path with cases (prompt, expected, evaluator, pattern).
        file: Optional document attachment (image, PDF or text file).
        image: Optional image attachment.
        nruns: Repeat every case this many times for consistency checks.
        registry: Optional path to an alternative TOML registry.
        out: Directory where reports are written.
    """
    try:
        if tag and models:
            return {"ok": False, "error": "Use tag or models, not both"}
        schema = shorthand_to_schema(structured_output) if structured_output else None
        config = RunConfig(
            prompt=prompt,
            providers=frozenset(part.strip().lower() for part in providers.split("|") if part.strip()),
            models=frozenset(part.strip() for part in models.split("|") if part.strip()) if models else None,
            tag=tag,
            structured_output=schema,
            dataset=dataset,
            file=file,
            image=image,
            nruns=max(1, nruns),
            registry=registry,
            out=out,
        )
        results = await run(config)
        json_path, html_path = save_report(config, results, out)
    except FastEvalError as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": all(row.ok for row in results),
        "json_path": str(json_path),
        "html_path": str(html_path),
        "total_cost_usd": sum(row.total_cost_usd or 0 for row in results),
        "results": [
            {
                "case_id": row.case_id,
                "provider": row.provider,
                "model": row.model,
                "reasoning_effort": row.reasoning_effort,
                "latency_ms": row.latency_ms,
                "total_cost_usd": row.total_cost_usd,
                "output": row.output,
                "error": row.error or None,
                "evaluation": row.evaluation,
            }
            for row in results
        ],
    }


@mcp.tool()
def list_models(registry: str | None = None) -> dict[str, Any]:
    """List models available in the fastevals registry.

    Args:
        registry: Optional path to an alternative TOML registry.
    """
    path = Path(registry) if registry else default_registry_path()
    try:
        entries = load_registry(path)
    except FastEvalError as exc:
        return {"ok": False, "error": str(exc), "models": []}
    return {
        "registry": str(path),
        "supported_providers": list(SUPPORTED_PROVIDERS),
        "models": describe_registry(entries),
    }


@mcp.tool()
def add_tag(
    name: str,
    models: str,
    description: str | None = None,
    registry: str | None = None,
) -> dict[str, Any]:
    """Save a named model suite ("tag") for reuse by you and agents.

    Selectors are validated against the registry before saving, so a tag
    can never contain unresolvable models.

    Args:
        name: Short suite name, e.g. ``cheap`` or ``nightly``.
        models: Pipe-separated ``provider/model[@efforts]`` selectors.
        description: Optional human-readable purpose of the suite.
        registry: Optional path used only for validating the selectors now.
    """
    selectors = [part.strip() for part in models.split("|") if part.strip()]
    if not selectors:
        return {"ok": False, "error": "Provide at least one model selector"}
    path = Path(registry) if registry else default_registry_path()
    try:
        entries = load_registry(path)
        cells = len(select_specs(entries, {"all"}, selectors=selectors))
        save_tag(name, selectors, description)
    except FastEvalError as exc:
        return {"ok": False, "name": name, "error": str(exc)}
    return {
        "ok": True,
        "name": name,
        "description": description,
        "models": selectors,
        "cells_in_current_registry": cells,
    }


@mcp.tool()
def list_tags() -> dict[str, Any]:
    """List saved model suites (tags) usable as run_evaluation's ``tag`` argument."""
    tags = load_tags()
    return {
        "ok": True,
        "tags": {
            name: {"description": tag["description"], "models": tag["models"]} for name, tag in sorted(tags.items())
        },
    }


@mcp.tool()
def get_run(json_path: str) -> dict[str, Any]:
    """Summarize a saved fastevals run from its ``run.json`` file."""
    path = Path(json_path)
    if not path.exists():
        return {"ok": False, "error": f"Run file not found: {json_path}"}
    import json

    payload = json.loads(path.read_text())
    results: list[dict[str, Any]] = payload.get("results", [])
    scored = [row for row in results if (row.get("evaluation") or {}).get("passed") is not None]
    return {
        "ok": bool(results) and all(not row.get("error") for row in results),
        "created_at": payload.get("created_at"),
        "runs": len(results),
        "errors": sum(1 for row in results if row.get("error")),
        "pass_rate": (sum(1 for row in scored if row["evaluation"]["passed"]) / len(scored) if scored else None),
        "total_cost_usd": sum(row.get("total_cost_usd") or 0 for row in results),
        "html_report": str(path.parent / "report.html"),
    }


def build_server() -> MCPServer:
    """Return the configured MCP server instance."""
    return mcp


def main() -> int:
    asyncio.run(mcp.run_stdio_async())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
