"""Command-line interface for fasteval."""

import argparse
import asyncio
import json
import os
from pathlib import Path

from .config import DEFAULT_MAX_CONCURRENCY, SUPPORTED_PROVIDERS, RunConfig
from .exceptions import FastEvalError
from .report import save_report
from .runner import run
from .structured import shorthand_to_schema

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fasteval",
        description="Compare one task results across LLM providers and models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  fasteval --prompt \"Summarize this\" --providers \"openai|gemini\" --out runs
  fasteval --image image.png --prompt \"Find widget bboxes\" \\
    --structured-output \"x:int(X coord),y:int(Y coord),width:int(Width),height:int(Height)\" \\
    --providers openai
  fasteval --prompt \"Hello\" --providers mock   # no API key needed

""",
    )
    parser.add_argument("-p", "--prompt", help="Task prompt (omit when --dataset provides the prompts)")
    parser.add_argument("-s", "--structured-output", help="Structured output compact schema for the response")
    parser.add_argument("-f", "--file", type=Path, help="Input document (sent to the model as an attachment)")
    parser.add_argument("-i", "--image", type=Path, help="Input image")
    parser.add_argument(
        "-pr",
        "--providers",
        type=_parse_providers,
        default=frozenset({ALL_PROVIDERS}),
        help=f"Pipe-separated providers: {'|'.join(SUPPORTED_PROVIDERS)}|all (default: all)",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = build_parser().parse_args(argv)

    try:
        config = RunConfig(
            prompt=args.prompt or "",
            providers=args.providers,
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
        "results": [vars(row) for row in results],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
