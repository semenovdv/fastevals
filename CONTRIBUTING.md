# Contributing to fasteval

Thanks for considering a contribution! The bar for this project is
deliberately high: it is meant to stay small, honest and boring to maintain.

## Setup

```bash
git clone https://github.com/semenovdv/fasteval.git
cd fasteval
python3 -m venv .venv && source .venv/bin/activate
make dev          # installs '.[dev,native]'
make demo         # verify the mock pipeline works without API keys
```

## Quality gates

Every PR must pass the same gates as CI:

```bash
make check        # ruff lint + format check, mypy --strict, tests >= 85% coverage
```

Rules of thumb:

- **No live API calls in tests.** Everything runs on the mock provider or
  stubbed LiteLLM calls.
- **Type everything.** `mypy --strict` passes on the package and must keep
  passing.
- **Public behavior gets tests.** Bug fixes should include a regression test;
  features need both unit coverage and an example if user-facing.
- **Keep the dependency list short.** Core has one runtime dependency by
  design; anything heavier belongs in an extra.

## Commit style

Short imperative subject lines (`Add dataset dry-run flag`), body explaining
*why* when it is not obvious. No AI-generated commit spam.

## Reporting issues

Include: command you ran, full CLI output, your registry (with keys redacted),
Python version. For report bugs, attach the `report.html`.

## License

By contributing you agree that your contributions are licensed under the MIT
License.
