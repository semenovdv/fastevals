"""Output evaluators: deterministic scoring for evaluation cases."""

import json
import re
from typing import Any

from .dataset import Case
from .exceptions import ConfigError

__all__ = ["evaluate_output"]

EVALUATORS = ("exact_match", "contains", "json_valid", "regex")


def _as_text(output: Any) -> str:
    if isinstance(output, (dict, list)):
        return json.dumps(output, ensure_ascii=False)
    return str(output or "")


def evaluate_output(case: Case, output: Any) -> dict[str, Any]:
    """Score one output against the case instructions.

    Returns ``{"evaluator", "passed", "detail"}``; ``passed`` is ``None``
    when the case defines no evaluator.
    """
    if not case.evaluator:
        return {"evaluator": None, "passed": None, "detail": None}

    name = case.evaluator.strip().lower()
    text = _as_text(output)
    if name == "exact_match":
        passed = text.strip() == (case.expected or "").strip()
        detail = None if passed else f"expected {case.expected!r}, got {text.strip()[:200]!r}"
    elif name == "contains":
        needle = case.expected or ""
        passed = bool(needle) and needle in text
        detail = None if passed else f"{needle!r} not found in output"
    elif name == "json_valid":
        try:
            json.loads(text)
            passed, detail = True, None
        except json.JSONDecodeError as exc:
            passed, detail = False, f"output is not valid JSON: {exc.msg}"
    elif name == "regex":
        if not case.pattern:
            raise ConfigError(f"Case '{case.id}' uses the regex evaluator but defines no pattern")
        match = re.search(case.pattern, text)
        passed, detail = match is not None, None if match else f"pattern {case.pattern!r} not found"
    else:
        raise ConfigError(
            f"Unknown evaluator '{case.evaluator}' for case '{case.id}'. Supported: {', '.join(EVALUATORS)}"
        )
    return {"evaluator": name, "passed": passed, "detail": detail}
