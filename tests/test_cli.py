import json

import pytest

from fasteval.cli import build_parser, main


def test_parser_accepts_pipe_separated_providers():
    args = build_parser().parse_args(["--prompt", "x", "--providers", "openai|mock"])
    assert args.providers == {"openai", "mock"}


def test_parser_rejects_unknown_provider():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--prompt", "x", "--providers", "nope"])


def test_parser_defaults_to_all_providers():
    args = build_parser().parse_args(["--prompt", "x"])
    assert args.providers == {"all"}


def test_missing_prompt_exits_with_error():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_mock_run_end_to_end(tmp_path, capsys):
    exit_code = main(["--prompt", "hi there", "--providers", "mock", "--out", str(tmp_path)])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    run_dir = tmp_path / payload["json_path"].split("/")[-2]
    assert (run_dir / "run.json").exists()
    assert (run_dir / "report.html").exists()
    saved = json.loads((run_dir / "run.json").read_text())
    assert saved["results"][0]["output"] == "[demo] hi there"


def test_structured_mock_run_end_to_end(tmp_path, capsys):
    exit_code = main(
        [
            "--prompt",
            "extract",
            "--structured-output",
            'name:str("A name"),age:int',
            "--providers",
            "mock",
            "--out",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][0]["output"] == {"name": "A name", "age": 1}


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
