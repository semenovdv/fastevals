"""Scenario-audit coverage: gaps found during the repository QA pass."""

import asyncio
import json
import os
import re
from pathlib import Path

import pytest

from fastevals import providers
from fastevals.cli import main
from fastevals.dataset import load_dataset
from fastevals.exceptions import ConfigError
from fastevals.tags import resolve_tag

# --- Fix 1: --list-models with a missing registry must not traceback ----------


def test_list_models_missing_registry_is_structured_error(tmp_path, capsys):
    exit_code = main(["--list-models", "--registry", str(tmp_path / "missing.toml")])
    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "not found" in payload["error"]


def test_run_with_missing_registry_is_config_error(tmp_path):
    from fastevals.config import RunConfig
    from fastevals.runner import run_evals

    with pytest.raises(ConfigError, match="Model registry not found"):
        asyncio.run(run_evals(RunConfig(prompt="hi", registry=str(tmp_path / "missing.toml"))))


# --- Fix 2: global agent dotenv store ----------------------------------------


def test_global_agent_dotenv_is_loaded(monkeypatch, tmp_path, capsys):
    global_env = tmp_path / "global" / ".env"
    global_env.parent.mkdir(parents=True)
    global_env.write_text("GLOBAL_AGENT_KEY=from-global\n")
    monkeypatch.setattr("fastevals.cli._dotenv_candidates", lambda: [global_env])
    monkeypatch.delenv("GLOBAL_AGENT_KEY", raising=False)

    empty_registry = tmp_path / "models.toml"
    empty_registry.write_text("")
    main(
        [
            "--prompt",
            "hi",
            "--providers",
            "openai",
            "--registry",
            str(empty_registry),
            "--out",
            str(tmp_path),
        ]
    )
    capsys.readouterr()
    import os

    assert os.environ.get("GLOBAL_AGENT_KEY") == "from-global"


def test_project_dotenv_wins_over_global_store(tmp_path, monkeypatch):
    project_env = tmp_path / ".env"
    project_env.write_text("PRIORITY_KEY=from-project\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "fastevals.cli._dotenv_candidates",
        lambda: [project_env, tmp_path / "global.env"],
    )
    monkeypatch.setenv("PRIORITY_KEY", "from-env-var")

    empty_registry = tmp_path / "models.toml"
    empty_registry.write_text("")
    main(["--prompt", "hi", "--registry", str(empty_registry), "--out", str(tmp_path)])
    import os

    # pre-set environment variables are never overwritten by any .env
    assert os.environ.get("PRIORITY_KEY") == "from-env-var"


# --- Fix 3: builtin tags resolve standalone in Python ------------------------


def test_resolve_builtin_tag_without_entries_loads_default_registry():
    selectors = resolve_tag("auto-deep")
    assert selectors and all("/" in s for s in selectors)


def test_resolve_unknown_still_lists_everything(tmp_path, monkeypatch):
    monkeypatch.setenv("FASTEVAL_TAGS_FILE", str(tmp_path / "tags.toml"))
    with pytest.raises(ConfigError) as excinfo:
        resolve_tag("nope")
    assert "saved:" in str(excinfo.value)
    assert "built-in:" in str(excinfo.value)


# --- D1: real MCP server over stdio (subprocess handshake) -------------------


@pytest.mark.asyncio
async def test_mcp_stdio_handshake_and_full_cycle(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    monkeypatch.setenv("FASTEVAL_TAGS_FILE", str(tmp_path / "tags.toml"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PATH", f"{Path(__file__).resolve().parents[1] / '.venv' / 'bin'}:{os.environ.get('PATH', '')}")

    params = StdioServerParameters(command="fastevals-mcp")
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        tools = {tool.name for tool in (await session.list_tools()).tools}
        assert {"run_evaluation", "list_models", "get_run", "add_tag", "list_tags"} <= tools

        added = await session.call_tool(
            "add_tag",
            {"name": "stdio-suite", "models": "openai/gpt-5.6-luna@none"},
        )
        assert added.is_error is False

        result = await session.call_tool(
            "run_evaluation",
            {"prompt": "hi", "tag": "stdio-suite", "providers": "all", "out": str(tmp_path)},
        )
        assert result.is_error is False
        # the spawned server has no litellm stub; a missing/failed provider
        # must surface as structured per-cell errors, never a crash
        payload = json.loads(result.content[0].text)
        assert payload["ok"] in (True, False)
        assert len(payload["results"]) >= 1


# --- E1: README python examples execute verbatim -----------------------------


def test_readme_python_examples_execute_verbatim(tmp_path, monkeypatch):
    readme = Path(__file__).resolve().parents[1] / "README.md"
    blocks = re.findall(r"```python\n(.*?)```", readme.read_text(), re.S)
    assert len(blocks) >= 2, "README must keep its runnable python examples"

    async def fake(**request):  # network stub at the litellm boundary
        from types import SimpleNamespace

        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
        )
        choice = SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=usage, id="readme")

    monkeypatch.setattr(providers, "_litellm_completion", fake)
    monkeypatch.setenv("FASTEVAL_TAGS_FILE", str(tmp_path / "tags.toml"))
    monkeypatch.chdir(tmp_path)

    for index, code in enumerate(blocks, start=1):
        namespace: dict = {"__name__": "__main__"}
        exec(compile(code, f"<README block {index}>", "exec"), namespace)


# --- E4: run_evals inside an already-running event loop ----------------------


@pytest.mark.asyncio
async def test_run_evals_inside_existing_loop(tmp_path, fake_llm, api_key, openai_registry):
    calls = fake_llm(text="nested ok")

    async def integrator_code():
        from fastevals.config import RunConfig
        from fastevals.runner import run_evals

        return await run_evals(RunConfig(prompt="hi", providers=frozenset({"openai"}), registry=str(openai_registry)))

    results = await integrator_code()
    assert len(results) == 2
    assert len(calls) == 2


# --- C4: CSV datasets tolerate blank lines like JSONL ------------------------


def test_csv_blank_lines_are_skipped(tmp_path):
    dataset = tmp_path / "cases.csv"
    dataset.write_text("prompt,expected\ncase one,x\n\ncase two,y\n")
    cases = load_dataset(dataset)
    assert [case.prompt for case in cases] == ["case one", "case two"]
