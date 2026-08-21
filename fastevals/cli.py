"""Command-line interface for fastevals."""

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from . import __version__
from .config import DEFAULT_MAX_CONCURRENCY, SUPPORTED_PROVIDERS, RunConfig
from .exceptions import ConfigError, FastEvalError
from .registry import default_registry_path, describe_registry, load_registry, select_specs
from .report import save_report
from .runner import run
from .structured import shorthand_to_schema
from .tags import default_tags_path, load_tags, remove_tag, resolve_tag, save_tag

ALL_PROVIDERS = "all"


def _dotenv_candidates() -> list[Path]:
    return [Path.cwd() / ".env", Path(__file__).resolve().parents[1] / ".env"]


def _load_dotenv() -> None:
    """Load simple KEY=VALUE entries from a project .env if present."""
    for env_path in _dotenv_candidates():
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value


def _parse_providers(raw: str) -> frozenset[str]:
    providers = {item.strip().lower() for item in raw.split("|") if item.strip()}
    unknown = sorted(providers - set(SUPPORTED_PROVIDERS) - {ALL_PROVIDERS})
    if unknown:
        supported = ", ".join((*SUPPORTED_PROVIDERS, ALL_PROVIDERS))
        raise argparse.ArgumentTypeError(f"Unknown provider(s): {', '.join(unknown)}. Supported: {supported}")
    return frozenset(providers)


def _parse_models(raw: str) -> frozenset[str]:
    models = frozenset(item.strip() for item in raw.split("|") if item.strip())
    if not models:
        raise argparse.ArgumentTypeError("--models must contain at least one selector, e.g. luna@low")
    return models


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fastevals",
        description="Compare one task results across LLM providers and models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  fastevals --list-models
  fastevals tag add cheap --models "openai/gpt-5.6-luna@none|openai/gpt-5.6-luna@low" -d "Smoke suite"
  fastevals --tag cheap --prompt \"Summarize this\" --out runs
  fastevals --models \"openai/gpt-5.6-luna@high|openai/gpt-5.6-sol@low\" \\
    --prompt \"Find widget bboxes\" --providers openai
  fastevals --dataset cases.jsonl --nruns 3 --tag nightly --out runs/dataset

""",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-l",
        "--list-models",
        action="store_true",
        help="Print the resolved model registry as JSON and exit",
    )
    parser.add_argument("-p", "--prompt", help="Task prompt (omit when --dataset provides the prompts)")
    parser.add_argument("-s", "--structured-output", help="Structured output compact schema for the response")
    parser.add_argument("-f", "--file", type=Path, help="Input document (sent to the model as an attachment)")
    parser.add_argument("-i", "--image", type=Path, help="Input image")
    parser.add_argument(
        "-t",
        "--tag",
        help="Use a saved model-suite tag instead of typing --models (see: fastevals tag list)",
    )
    parser.add_argument(
        "--providers",
        type=_parse_providers,
        default=frozenset({ALL_PROVIDERS}),
        help=f"Pipe-separated providers: {'|'.join(SUPPORTED_PROVIDERS)}|all (default: all)",
    )
    parser.add_argument(
        "-m",
        "--models",
        type=_parse_models,
        help="Cherry-pick models, provider required: 'openai/gpt-5.6-luna@none,high'",
    )
    parser.add_argument(
        "-r", "--registry", type=Path, help="Path to the model registry TOML (default: config/models.toml)"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=Path,
        help="JSONL or CSV file with evaluation cases (columns: prompt, expected, evaluator, pattern)",
    )
    parser.add_argument(
        "-n", "--nruns", type=int, default=1, help="Repeat every case this many times for consistency checks"
    )
    parser.add_argument(
        "-c", "--concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY, help="Max parallel model calls"
    )
    parser.add_argument("-o", "--out", type=Path, default=Path("runs"), help="Output directory")

    sub = parser.add_subparsers(dest="command")
    tag_parser = sub.add_parser("tag", help="Manage saved model suites (presets)")
    tag_sub = tag_parser.add_subparsers(dest="tag_command", required=True)

    tag_add = tag_sub.add_parser("add", help="Create or overwrite a tag")
    tag_add.add_argument("name")
    tag_add.add_argument(
        "-m",
        "--models",
        required=True,
        help="Pipe-separated selectors, provider required: 'openai/gpt-5.6-luna@none|openai/gpt-5.6-sol@low'",
    )
    tag_add.add_argument("-d", "--description", help="What the suite is for")
    tag_add.add_argument("-r", "--registry", type=Path, help="Validate selectors against this registry")

    tag_sub.add_parser("list", help="List all tags")
    tag_show = tag_sub.add_parser("show", help="Print one tag as JSON")
    tag_show.add_argument("name")
    tag_remove = tag_sub.add_parser("remove", help="Delete a tag")
    tag_remove.add_argument("name")
    return parser


def _print_models(registry_override: Path | None) -> int:
    path = registry_override or default_registry_path()
    try:
        models = describe_registry(load_registry(path))
    except FastEvalError as exc:
        print(json.dumps({"ok": False, "registry": str(path), "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({"ok": True, "registry": str(path), "models": models}, ensure_ascii=False, indent=2))
    return 0


def _emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _handle_tag(args: argparse.Namespace) -> int:
    if args.tag_command == "add":
        selectors = _parse_models(args.models)
        registry_path = args.registry or default_registry_path()
        try:
            entries = load_registry(registry_path)
            cells = len(select_specs(entries, {"all"}, selectors=selectors))
            save_tag(args.name, sorted(selectors), args.description)
        except FastEvalError as exc:
            _emit({"ok": False, "name": args.name, "error": str(exc)})
            return 1
        _emit(
            {
                "ok": True,
                "name": args.name.strip(),
                "description": args.description,
                "models": sorted(selectors),
                "cells_in_current_registry": cells,
            }
        )
        return 0

    if args.tag_command == "list":
        tags = load_tags()
        _emit(
            {
                "ok": True,
                "tags_file": str(default_tags_path()),
                "tags": {
                    name: {"description": tag["description"], "models": tag["models"]}
                    for name, tag in sorted(tags.items())
                },
            }
        )
        return 0

    if args.tag_command == "show":
        try:
            models = resolve_tag(args.name)
        except FastEvalError as exc:
            _emit({"ok": False, "name": args.name, "error": str(exc)})
            return 1
        description = load_tags().get(args.name.strip(), {}).get("description")
        _emit({"ok": True, "name": args.name.strip(), "description": description, "models": models})
        return 0

    if args.tag_command == "remove":
        existed = remove_tag(args.name)
        _emit({"ok": True, "name": args.name.strip(), "removed": existed})
        return 0

    _emit({"ok": False, "error": f"Unknown tag command: {args.tag_command}"})
    return 1


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = build_parser().parse_args(argv)

    if getattr(args, "command", None) == "tag":
        return _handle_tag(args)

    if args.list_models:
        return _print_models(args.registry)

    try:
        models = args.models
        if args.tag:
            if args.models:
                raise ConfigError("Use --tag or --models, not both")
            models = frozenset(resolve_tag(args.tag))
        config = RunConfig(
            prompt=args.prompt or "",
            providers=args.providers,
            models=models,
            file=str(args.file) if args.file else None,
            image=str(args.image) if args.image else None,
            structured_output=shorthand_to_schema(args.structured_output) if args.structured_output else None,
            dataset=str(args.dataset) if args.dataset else None,
            nruns=max(1, args.nruns),
            registry=str(args.registry) if args.registry else None,
            max_concurrency=max(1, args.concurrency),
            out=str(args.out),
        )
        results = asyncio.run(run(config))
    except (FastEvalError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc), "results": []}, ensure_ascii=False))
        return 1

    json_path, html_path = save_report(config, results, args.out)
    payload = {
        "ok": all(row.ok for row in results),
        "json_path": str(json_path),
        "html_path": str(html_path) if html_path else None,
        "results": [row.as_dict() for row in results],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
