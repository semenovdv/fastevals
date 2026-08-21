import json
from pathlib import Path

import pytest

from fasteval.dataset import Case, load_dataset
from fasteval.evaluators import evaluate_output
from fasteval.exceptions import ConfigError


def test_load_jsonl_dataset(tmp_path: Path):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "cap",
                        "prompt": "Name the capital of France.",
                        "expected": "Paris",
                        "evaluator": "exact_match",
                    }
                ),
                json.dumps({"prompt": "Say something about JSON.", "evaluator": "json_valid"}),
            ]
        )
    )
    cases = load_dataset(dataset)
    assert [case.id for case in cases] == ["cap", "case-002"]
    assert cases[0].expected == "Paris"


def test_load_csv_dataset(tmp_path: Path):
    dataset = tmp_path / "cases.csv"
    dataset.write_text("prompt,expected,evaluator\nSay hi,hi,exact_match\n\n")
    cases = load_dataset(dataset)
    assert len(cases) == 1
    assert cases[0].prompt == "Say hi"


def test_load_dataset_rejects_missing_prompt(tmp_path: Path):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text('{"expected": "x"}\n')
    with pytest.raises(ConfigError, match="missing required field"):
        load_dataset(dataset)


def test_load_dataset_rejects_unknown_format(tmp_path: Path):
    dataset = tmp_path / "cases.yaml"
    dataset.write_text("prompt: hi\n")
    with pytest.raises(ConfigError, match="Unsupported dataset format"):
        load_dataset(dataset)


def test_load_dataset_rejects_empty_file(tmp_path: Path):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text("\n\n")
    with pytest.raises(ConfigError, match="Dataset is empty"):
        load_dataset(dataset)


def test_exact_match_evaluator():
    case = Case(id="c", prompt="p", expected="Paris", evaluator="exact_match")
    assert evaluate_output(case, "  Paris\n")["passed"] is True
    failed = evaluate_output(case, "Lyon")
    assert failed["passed"] is False
    assert "expected" in failed["detail"]


def test_contains_and_regex_evaluators():
    contains = Case(id="c", prompt="p", expected="42", evaluator="contains")
    assert evaluate_output(contains, "The answer is 42!")["passed"] is True
    regex = Case(id="r", prompt="p", evaluator="regex", pattern=r"\d{4}-\d{2}-\d{2}")
    assert evaluate_output(regex, "Date: 2026-08-21.")["passed"] is True
    assert evaluate_output(regex, "no date here")["passed"] is False


def test_json_valid_evaluator_accepts_structured_output():
    case = Case(id="c", prompt="p", evaluator="json_valid")
    assert evaluate_output(case, {"already": "parsed"})["passed"] is True
    assert evaluate_output(case, "not json")["passed"] is False


def test_no_evaluator_yields_neutral_result():
    case = Case(id="c", prompt="p")
    evaluation = evaluate_output(case, "anything")
    assert evaluation["passed"] is None
    assert evaluation["evaluator"] is None


def test_regex_without_pattern_raises():
    case = Case(id="c", prompt="p", evaluator="regex")
    with pytest.raises(ConfigError, match="no pattern"):
        evaluate_output(case, "text")


def test_unknown_evaluator_raises():
    case = Case(id="c", prompt="p", evaluator="llm_judge")
    with pytest.raises(ConfigError, match="Unknown evaluator"):
        evaluate_output(case, "text")
