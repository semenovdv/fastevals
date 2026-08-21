# Security Policy

## Supported versions

Only the latest `main` branch is supported. This project is pre-1.0; pin a
commit if you need stability.

## Reporting a vulnerability

Open a [private security advisory](https://github.com/semenovdv/fastevals/security/advisories/new)
or email the maintainer directly. Please do not open a public issue for
security reports. You can expect a response within a few days.

## Design notes relevant to security

- **Secrets**: API keys are read exclusively from environment variables named
  in `config/models.toml`. They are never written to reports, `run.json`,
  logs or error messages — error strings pass through a scrubber that masks
  any key value known to the process.
- **Reports**: `report.html` is a standalone file intended for sharing, but
  it contains full model outputs — review prompts and responses before
  sending reports to third parties.
- **Attachments** are limited to 20 MB and are base64-encoded into requests;
  nothing is uploaded anywhere except the configured provider.
- **MCP server** binds to stdio only and executes only the three documented
  tools; it performs no filesystem writes outside the `--out` directory you
  pass to `run_evaluation`.
- **Dependencies**: the core package has a single runtime dependency
  (`jsonschema`); provider SDKs are optional extras.
