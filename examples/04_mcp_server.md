# Connecting fastevals to Claude via MCP

fastevals ships an MCP server, so AI assistants can run evaluations as a tool.

## Install

```bash
python3 -m pip install -e '.[mcp,native]'
```

The entry point is `fastevals-mcp` (stdio transport).

## Claude Code

```bash
claude mcp add fastevals -- fastevals-mcp
```

## Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "fastevals": {
      "command": "fastevals-mcp"
    }
  }
}
```

## Tools exposed

| Tool | Purpose |
|---|---|
| `run_evaluation` | Run one prompt (or a dataset) across providers and save a report |
| `list_models` | Show the model registry: providers, efforts, pricing |
| `get_run` | Summarize a saved `run.json`, including pass rate and cost |

Example assistant prompt:

> Use fastevals to compare gpt-5.6-luna at reasoning none and low on
> "Explain evaluation in three bullets", then tell me which run was cheaper.

The mock provider works without any API key, so you can explore the tools
before spending credits.
