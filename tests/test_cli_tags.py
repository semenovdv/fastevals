import json

import pytest

from fastevals.cli import main


@pytest.fixture
def tags_file(tmp_path, monkeypatch):
    path = tmp_path / "tags.toml"
    monkeypatch.setenv("FASTEVAL_TAGS_FILE", str(path))
    return path


def _payload(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def test_tag_add_validates_against_registry(tags_file, capsys):
    exit_code = main(
        [
            "tag",
            "add",
            "cheap",
            "--models",
            "openai/gpt-5.6-luna@none|openai/gpt-5.6-luna@low",
            "-d",
            "Smoke suite",
        ]
    )
    assert exit_code == 0
    payload = _payload(capsys)
    assert payload["ok"] is True
    assert payload["cells_in_current_registry"] == 2
    assert tags_file.exists()


def test_tag_add_rejects_unresolvable_selector(tags_file, capsys):
    exit_code = main(["tag", "add", "broken", "--models", "openai/gpt-5.6-cyber@low"])
    assert exit_code == 1
    payload = _payload(capsys)
    assert payload["ok"] is False
    assert not tags_file.exists() or "broken" not in tags_file.read_text()


def test_tag_list_show_remove_lifecycle(tags_file, capsys):
    main(["tag", "add", "suite", "--models", "openai/gpt-5.6-luna"])
    capsys.readouterr()

    assert main(["tag", "list"]) == 0
    payload = _payload(capsys)
    assert "suite" in payload["tags"]

    assert main(["tag", "show", "suite"]) == 0
    payload = _payload(capsys)
    assert payload["models"] == ["openai/gpt-5.6-luna"]

    assert main(["tag", "remove", "suite"]) == 0
    payload = _payload(capsys)
    assert payload["removed"] is True
    capsys.readouterr()

    exit_code = main(["tag", "show", "suite"])
    assert exit_code == 1


def test_run_with_tag_end_to_end(tmp_path, capsys, fake_llm, api_key, openai_registry, tags_file):
    main(
        [
            "tag",
            "add",
            "pair",
            "--models",
            "openai/gpt-test@off|openai/gpt-test@low",
            "--registry",
            str(openai_registry),
        ]
    )
    capsys.readouterr()
    calls = fake_llm(text="answer")
    exit_code = main(
        [
            "--prompt",
            "hi",
            "--tag",
            "pair",
            "--registry",
            str(openai_registry),
            "--providers",
            "openai",
            "--out",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    payload = _payload(capsys)
    assert len(payload["results"]) == 2
    assert len(calls) == 2


def test_run_with_unknown_tag_fails(tmp_path, capsys, tags_file):
    exit_code = main(["--prompt", "hi", "--tag", "missing", "--out", str(tmp_path)])
    assert exit_code == 1
    payload = _payload(capsys)
    assert "built-in:" in payload["error"]
    assert payload["error"].startswith("Unknown tag 'missing'")


def test_tag_and_models_are_mutually_exclusive(tmp_path, capsys, tags_file):
    exit_code = main(["--prompt", "hi", "--tag", "x", "--models", "openai/gpt-5.6-luna", "--out", str(tmp_path)])
    assert exit_code == 1
    payload = _payload(capsys)
    assert "not both" in payload["error"]
