import json
from pathlib import Path

import pytest

from fastevals.cli import build_parser, main


def test_parser_accepts_pipe_separated_providers():
    args = build_parser().parse_args(["--prompt", "x", "--providers", "openai|gemini"])
    assert args.providers == {"openai", "gemini"}


def test_parser_rejects_unknown_provider():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--prompt", "x", "--providers", "nope"])


def test_parser_defaults_to_all_providers():
    args = build_parser().parse_args(["--prompt", "x"])
    assert args.providers == {"all"}


def test_parser_accepts_model_selectors():
    args = build_parser().parse_args(["--prompt", "x", "--models", "openai/gpt-5.6-luna@none,high"])
    assert args.models == {"openai/gpt-5.6-luna@none,high"}


def test_parser_rejects_empty_models_value():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--prompt", "x", "--models", "|"])


def test_parser_no_longer_accepts_pr_short_flag():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["-pr", "openai"])


def test_version_flag_prints_version_and_exits(capsys):
    from fastevals import __version__

    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_list_models_prints_bundled_registry(capsys):
    exit_code = main(["--list-models"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    ids = [model["id"] for model in payload["models"]]
    assert "openai:gpt-5.6-luna" in ids
    assert all("reasoning_efforts" in model for model in payload["models"])


def test_list_models_with_custom_registry(tmp_path, capsys):
    registry = tmp_path / "custom.toml"
    registry.write_text('["openai:gpt-x"]\nprovider = "openai"\nmodel = "gpt-x"\n')
    exit_code = main(["--list-models", "--registry", str(registry)])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["models"][0]["model"] == "gpt-x"


def test_list_models_with_empty_registry_fails_cleanly(tmp_path, capsys):
    registry = tmp_path / "empty.toml"
    registry.write_text("")
    exit_code = main(["--list-models", "--registry", str(registry)])
    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False


def test_missing_prompt_and_dataset_returns_error(tmp_path, capsys):
    exit_code = main(["--out", str(tmp_path)])
    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "Prompt" in payload["error"]


@pytest.fixture
def offline_openai(fake_llm, api_key, openai_registry):
    """Wire the CLI to the fake LiteLLM with a temporary registry."""
    return ["--registry", str(openai_registry)]


def test_run_end_to_end(tmp_path, capsys, offline_openai, fake_llm):
    fake_llm(text="answer text")
    exit_code = main(["--prompt", "hi there", "--providers", "openai", *offline_openai, "--out", str(tmp_path)])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    run_dir = Path(payload["json_path"]).parent
    assert (run_dir / "run.json").exists()
    assert (run_dir / "report.html").exists()
    saved = json.loads((run_dir / "run.json").read_text())
    assert [row["output"] for row in saved["results"]] == ["answer text", "answer text"]
    assert all("ok" in row and "total_cost_usd" in row for row in saved["results"])


def test_structured_run_end_to_end(tmp_path, capsys, offline_openai, fake_llm):
    fake_llm(text=json.dumps({"name": "Ada"}))
    exit_code = main(
        [
            "--prompt",
            "extract",
            "--structured-output",
            'name:str("A name")',
            "--providers",
            "openai",
            *offline_openai,
            "--out",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    outputs = [row["output"] for row in payload["results"]]
    assert {"name": "Ada"} in outputs


def test_run_end_to_end_with_model_selector(tmp_path, capsys, offline_openai, fake_llm):
    calls = fake_llm(text="answer text")
    exit_code = main(
        [
            "--prompt",
            "hi there",
            "--providers",
            "openai",
            "--models",
            "openai/gpt-test@low",
            *offline_openai,
            "--out",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert len(payload["results"]) == 1
    assert payload["results"][0]["reasoning_effort"] == "low"
    assert len(calls) == 1


def test_failed_run_returns_exit_code_one(tmp_path, capsys):
    empty_registry = tmp_path / "empty.toml"
    empty_registry.write_text("")
    exit_code = main(
        ["--prompt", "hi", "--providers", "openai", "--registry", str(empty_registry), "--out", str(tmp_path)]
    )
    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "registry is empty" in payload["error"]


def test_dotenv_loaded_from_cwd(tmp_path, capsys, monkeypatch):
    (tmp_path / ".env").write_text('MY_TEST_API_KEY="abc-123"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MY_TEST_API_KEY", raising=False)
    empty_registry = tmp_path / "models.toml"
    empty_registry.write_text("")
    main(["--prompt", "hi", "--providers", "openai", "--registry", str(empty_registry), "--out", str(tmp_path)])
    capsys.readouterr()
    import os

    assert os.environ.get("MY_TEST_API_KEY") == "abc-123"
