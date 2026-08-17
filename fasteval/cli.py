import argparse
import asyncio
import json
import sys
import os
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fasteval.runner import run, save_report
    from fasteval.structured import shorthand_to_schema
else:
    from .runner import run, save_report
    from .structured import shorthand_to_schema


def _load_dotenv() -> None:
    """Load simple KEY=VALUE entries from the project .env if present."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def main() -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(
        prog="fasteval",
        description="Compare one task results across LLM providers and models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  fasteval --prompt \"Summarize this\" --file report.pdf --providers \"openai|gemini\" --out runs
  fasteval --image image.png --prompt \"Find widget bboxes\" --structured-output \"x:int(X start coordinate),y:int(Y start coordinate),width:int(Width),height:int(Height)\" --providers openai

""",
    )
    parser.add_argument("-p", "--prompt", required=True, help="Task prompt")
    parser.add_argument("-s", "--structured-output", help="Structured output compact schema for the response")
    parser.add_argument("-f", "--file", type=Path, help="Input document")
    parser.add_argument("-i", "--image", type=Path, help="Input image")
    parser.add_argument("-pr", "--providers", default="all", choices=("all", "openai", "gemini", "anthropic"), help="Pipe-separated list of providers")
    parser.add_argument("-o", "--out", type=Path, default=Path("runs"), help="Output directory")
    #parser.add_argument("-n", "--nruns", type=int, default=1, help="Number of runs for consistency checks")

    args = parser.parse_args()
    providers = {item.strip().lower() for item in args.providers.split("|") if item.strip()}
    
    config = {
        "prompt": args.prompt, 
        "file": str(args.file) if args.file else None, 
        "image": str(args.image) if args.image else None,
        "structured_output": shorthand_to_schema(args.structured_output) if args.structured_output else None,
        "providers": providers, 
        #"nruns": args.nruns,
        "out": str(args.out),
        }
    results = asyncio.run(run(config))
    json_path, html_path = save_report(config, results, args.out)
    payload = {"ok": not any(row.error for row in results), "json_path": str(json_path), "html_path": str(html_path), "results": [row.__dict__ for row in results]}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
