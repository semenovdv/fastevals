"""Dataset loading: JSONL and CSV evaluation cases."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .exceptions import ConfigError

__all__ = ["Case", "load_dataset"]

_REQUIRED_FIELDS = frozenset({"prompt"})


@dataclass(frozen=True)
class Case:
    """One evaluation input with optional scoring instructions."""

    id: str
    prompt: str
    expected: str | None = None
    evaluator: str | None = None
    pattern: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "expected": self.expected,
            "evaluator": self.evaluator,
            "pattern": self.pattern,
        }


def _make_case(index: int, row: dict[str, str]) -> Case:
    missing = _REQUIRED_FIELDS - {key for key, value in row.items() if value}
    if missing:
        raise ConfigError(f"Dataset row {index + 1} is missing required field(s): {', '.join(sorted(missing))}")
    case_id = str(row.get("id") or f"case-{index + 1:03d}")
    return Case(
        id=case_id,
        prompt=str(row["prompt"]).strip(),
        expected=(str(row["expected"]) if row.get("expected") else None),
        evaluator=(str(row["evaluator"]).strip() or None) if row.get("evaluator") else None,
        pattern=row.get("pattern") or None,
    )


def load_dataset(path: str | Path) -> list[Case]:
    """Load cases from a ``.jsonl`` or ``.csv`` file."""
    path = Path(path)
    suffix = path.suffix.lower()
    rows: list[dict[str, str]]
    if suffix == ".jsonl":
        rows = []
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"Dataset line {line_number} is not valid JSON: {exc.msg}") from exc
            if not isinstance(item, dict):
                raise ConfigError(f"Dataset line {line_number} must be a JSON object")
            rows.append({key: "" if value is None else str(value) for key, value in item.items()})
    elif suffix == ".csv":
        with path.open(newline="") as dataset_file:
            rows = [{key: (value or "") for key, value in row.items()} for row in csv.DictReader(dataset_file)]
    else:
        raise ConfigError(f"Unsupported dataset format '{suffix}'. Use .jsonl or .csv")
    if not rows:
        raise ConfigError(f"Dataset is empty: {path}")
    return [_make_case(index, row) for index, row in enumerate(rows)]
