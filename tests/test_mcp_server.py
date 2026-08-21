import json
from pathlib import Path

import pytest

from fastevals.mcp_server import build_server


def parse(result):
    assert result.is_error is False, result.content
    return json.loads(result.content[0].text)


@pytest.mark.asyncio
async def test_server_exposes_documented_tools():
    tools = await build_server().list_tools()
    by_name = {tool.name for tool in tools}
    assert {"run_evaluation", "list_models", "get_run"} <= by_name
    assert all(tool.description for tool in tools)


@pytest.mark.asyncio
async def test_run_evaluation_end_to_end(tmp_path: Path, fake_llm, api_key, openai_registry):
    fake_llm(text="hello mcp")
    result = await build_server().call_tool(
        "run_evaluation",
        {
            "prompt": "hello mcp",
            "providers": "openai",
            "registry": str(openai_registry),
            "out": str(tmp_path),
        },
    )
    payload = parse(result)
    assert payload["ok"] is True
    json_path = Path(payload["json_path"])
    html_path = Path(payload["html_path"])
    assert json_path.exists() and html_path.exists()
    saved = json.loads(json_path.read_text())
    assert {row["output"] for row in saved["results"]} == {"hello mcp"}


@pytest.mark.asyncio
async def test_run_evaluation_structured_output(tmp_path: Path, fake_llm, api_key, openai_registry):
    fake_llm(text=json.dumps({"name": "Ada"}))
    result = await build_server().call_tool(
        "run_evaluation",
        {
            "prompt": "extract name",
            "providers": "openai",
            "registry": str(openai_registry),
            "structured_output": 'name:str("A name")',
            "out": str(tmp_path),
        },
    )
    payload = parse(result)
    outputs = [row["output"] for row in payload["results"]]
    assert {"name": "Ada"} in outputs


@pytest.mark.asyncio
async def test_run_evaluation_with_model_selectors(tmp_path: Path, fake_llm, api_key, openai_registry):
    calls = fake_llm(text="hi")
    result = await build_server().call_tool(
        "run_evaluation",
        {
            "prompt": "hi",
            "providers": "openai",
            "registry": str(openai_registry),
            "models": "openai/gpt-test@low",
            "out": str(tmp_path),
        },
    )
    payload = parse(result)
    assert payload["ok"] is True
    assert [row["reasoning_effort"] for row in payload["results"]] == ["low"]
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_run_evaluation_reports_config_errors(tmp_path: Path):
    result = await build_server().call_tool(
        "run_evaluation",
        {"prompt": "", "providers": "openai", "dataset": str(tmp_path / "missing.jsonl")},
    )
    payload = parse(result)
    assert payload["ok"] is False
    assert "not found" in payload["error"]


@pytest.mark.asyncio
async def test_list_models_reads_committed_registry():
    result = await build_server().call_tool("list_models", {})
    payload = parse(result)
    ids = [model["id"] for model in payload["models"]]
    assert any(model_id.startswith("openai:") for model_id in ids)
    assert set(payload["supported_providers"]) == {"openai", "gemini", "openrouter"}


@pytest.mark.asyncio
async def test_list_models_with_custom_registry(tmp_path: Path):
    registry = tmp_path / "custom.toml"
    registry.write_text('["openai:gpt-x"]\nprovider = "openai"\nmodel = "gpt-x"\ninput_cost_usd_per_mtok = 2.0\n')
    result = await build_server().call_tool("list_models", {"registry": str(registry)})
    payload = parse(result)
    assert payload["registry"] == str(registry)
    assert payload["models"][0]["model"] == "gpt-x"


@pytest.mark.asyncio
async def test_list_models_with_broken_registry(tmp_path: Path):
    registry = tmp_path / "broken.toml"
    registry.write_text("")
    result = await build_server().call_tool("list_models", {"registry": str(registry)})
    payload = parse(result)
    assert payload["ok"] is False


@pytest.mark.asyncio
async def test_get_run_summarizes_saved_run(tmp_path: Path, fake_llm, api_key, openai_registry):
    server = build_server()
    fake_llm(text="hi")
    run_result = await server.call_tool(
        "run_evaluation",
        {
            "prompt": "hi",
            "providers": "openai",
            "registry": str(openai_registry),
            "out": str(tmp_path),
        },
    )
    json_path = parse(run_result)["json_path"]
    summary = parse(await server.call_tool("get_run", {"json_path": json_path}))
    assert summary["runs"] == 2
    assert summary["errors"] == 0
    assert summary["pass_rate"] is None  # no evaluator configured
    assert summary["html_report"].endswith("report.html")


@pytest.mark.asyncio
async def test_run_evaluation_saves_dataset_cases(tmp_path: Path, fake_llm, api_key, openai_registry):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps({"id": "a", "prompt": "hi"}))
    fake_llm(text="hi")
    result = await build_server().call_tool(
        "run_evaluation",
        {
            "providers": "openai",
            "registry": str(openai_registry),
            "dataset": str(dataset),
            "out": str(tmp_path),
        },
    )
    payload = parse(result)
    saved = json.loads(Path(payload["json_path"]).read_text())
    assert saved["cases"][0]["id"] == "a"


@pytest.mark.asyncio
async def test_get_run_missing_file():
    result = await build_server().call_tool("get_run", {"json_path": "/nope/run.json"})
    payload = parse(result)
    assert payload["ok"] is False


@pytest.mark.asyncio
async def test_add_and_use_tag_via_mcp(tmp_path: Path, fake_llm, api_key, openai_registry, monkeypatch):
    monkeypatch.setenv("FASTEVAL_TAGS_FILE", str(tmp_path / "tags.toml"))
    server = build_server()

    added = parse(
        await server.call_tool(
            "add_tag",
            {
                "name": "pair",
                "models": "openai/gpt-test@off|openai/gpt-test@low",
                "description": "two-effort suite",
                "registry": str(openai_registry),
            },
        )
    )
    assert added["ok"] is True
    assert added["cells_in_current_registry"] == 2

    listed = parse(await server.call_tool("list_tags", {}))
    assert listed["tags"]["pair"]["models"] == ["openai/gpt-test@off", "openai/gpt-test@low"]

    calls = fake_llm(text="hi")
    result = await server.call_tool(
        "run_evaluation",
        {
            "prompt": "hi",
            "providers": "openai",
            "registry": str(openai_registry),
            "tag": "pair",
            "out": str(tmp_path),
        },
    )
    payload = parse(result)
    assert payload["ok"] is True
    assert len(payload["results"]) == 2
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_tag_and_models_mutually_exclusive_in_mcp(tmp_path: Path):
    result = await build_server().call_tool(
        "run_evaluation",
        {"prompt": "hi", "tag": "x", "models": "openai/gpt-5.6-luna"},
    )
    payload = parse(result)
    assert payload["ok"] is False
    assert "not both" in payload["error"]
