# fastevals

**Evaluation tooling your AI agents can drive.**

fastevals is a small, provider-agnostic evaluation runner for LLM applications.
Run one prompt — or a whole dataset — across a matrix of models, reasoning
efforts and providers, save every response, and get a readable standalone
HTML comparison report with cost, latency and token metrics.

It ships as an **MCP server**, so Claude Desktop, Claude Code or any other
MCP client can run evaluations as a native tool.

<p align="center">
  <img src="assets/report.png" alt="fastevals HTML report" width="820">
</p>

## Highlights

- **Tags** — build a named model suite once; reuse it from the CLI, Python
  and AI agents. Four adaptive built-ins (`auto-fast`, `auto-deep`,
  `auto-cheap`, `auto-flagship`) work with any registry on day zero.
- **Model selectors** — compare exact cells without editing files:
  `openai/gpt-5.6-luna@high|openai/gpt-5.6-sol@low`.
- **Structured output that verifies** — compact schema → strict JSON Schema,
  every response validated locally.
- **Real metrics** — streamed completions give genuine time-to-first-token;
  disjoint token buckets keep costs honest.
- **Datasets & evaluators** — JSONL/CSV cases with `exact_match`, `contains`,
  `json_valid`, `regex`; repeated runs for stability.
- **Agent-ready** — non-interactive CLI returning JSON, plus an MCP server
  so assistants run evaluations themselves.

## Quick start

```bash
pip install fastevals
export OPENAI_API_KEY=...

fastevals --list-models                                # see what you can run
fastevals --prompt "Explain evaluation in 3 bullets" --providers openai --out runs
```

Every run produces `run.json` (machine-readable) and `report.html`
(a standalone dashboard). The bundled model registry works out of the box.

## Where to go next

- **[Usage guide](USAGE.md)** — the full manual: CLI reference, selector
  grammar, registry format, datasets, structured output, MCP tools,
  Python API, troubleshooting.
- [Changelog](changelog.md) · [Roadmap](ROADMAP.md) ·
  Source on [GitHub](https://github.com/semenovdv/fastevals)

MIT licensed.
