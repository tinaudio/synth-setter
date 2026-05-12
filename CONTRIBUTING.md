# Contributing to synth-setter

Thank you for your interest in contributing to synth-setter. This guide covers
everything you need to get started.

For detailed internal standards (aimed at AI-assisted development), see
[CLAUDE.md](CLAUDE.md). This document is the human-readable summary.

## Getting started

### Prerequisites

- **Python 3.10+**
- **[uv](https://github.com/astral-sh/uv)** (fast pip replacement) or plain pip
- **make** (GNU Make)
- **pre-commit** (`pip install pre-commit`)
- **Git**
- **[bats](https://github.com/bats-core/bats-core)** (optional, for shell tests:
  `brew install bats-core` on macOS or `apt-get install bats` on Debian/Ubuntu)

### Clone and install

```bash
git clone https://github.com/tinaudio/synth-setter.git
cd synth-setter
```

Install the project in editable mode with development dependencies:

```bash
make install        # runs: pip install uv && uv pip install -r requirements.txt -e .
```

Install pre-commit hooks:

```bash
pre-commit install
```

Verify your setup:

```bash
make test-fast      # quick tests (CPU-only; excludes slow, gpu, mps, requires_vst)
```

## Good first issues

If you're looking for a place to start, browse the [good first issue label](https://github.com/tinaudio/synth-setter/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22).
These are scoped, self-contained tasks that don't require deep audio/ML domain
knowledge or access to cloud resources (R2, RunPod, GPU runners).

To claim one, comment on the issue to let others know you're working on it, and
ask any questions in the thread. When you open your PR, reference the issue in
the PR body with `Closes #N` so it auto-closes on merge.

## Code standards

### Formatting and linting

All formatting and linting is enforced automatically by **pre-commit hooks** on
every commit. The key tools are:

- **[Ruff](https://docs.astral.sh/ruff/)** for linting (rules: E, F, I, S, T,
  UP, W, plus D102/D103/D107 for "must have a docstring" on public functions,
  methods, and `__init__` — closes pydoclint's missing-docstring blind spot)
  and formatting (line length 99)
- **[Pyright](https://microsoft.github.io/pyright/)** for static type checking
- **[interrogate](https://interrogate.readthedocs.io/)** for docstring coverage
  (minimum 80%)
- **[docformatter](https://github.com/PyCQA/docformatter)** for docstring
  normalization (Sphinx style)
- **[pydoclint](https://github.com/jsh9/pydoclint)** for signature ↔ docstring
  consistency (Sphinx style; checks args, returns/yields, raises, and class
  attributes — config in `pyproject.toml` under `[tool.pydoclint]`)
- **[shellcheck](https://www.shellcheck.net/)** for shell script linting
- **[mdformat](https://mdformat.readthedocs.io/)** for Markdown formatting
- **[codespell](https://github.com/codespell-project/codespell)** for typo
  detection
- **[prettier](https://prettier.io/)** for YAML formatting

Run all hooks manually with:

```bash
make format         # runs: pre-commit run -a
```

Always run `make format` before committing. This catches issues early and
auto-fixes what it can.

#### Editor integration

The repo also ships editor-side wiring so the same fast formatters run on save,
not just at commit time:

- `.editorconfig` — cross-editor indent, EOL, and trim-trailing-whitespace
  rules.
- `.vscode/extensions.json` — when you open the project in VS Code or Cursor,
  you'll be prompted to install the recommended extensions (ruff, prettier,
  editorconfig, runonsave, shellcheck).
- `.vscode/settings.json` — once those extensions are installed, save will
  format Python via ruff, YAML via prettier, and Markdown via
  `pre-commit run mdformat --files <path>`.
- `.claude/settings.json` — Claude Code's Edit/Write tool calls trigger the
  same dispatch.

All three editor surfaces go through `pre-commit run <hook> --files <path>`
for Markdown and YAML, so save-time output is byte-identical to `make format`
output — no version drift.

### Writing code

- **Type-annotate all function signatures.** Avoid `Any` -- use `Union`,
  `Optional`, or specific types.
- **No bare `except:`.** Always catch specific exceptions.
- **Pydantic `BaseModel` with `strict=True`** at trust boundaries (config
  parsing, external JSON). Use dataclasses for internal typed containers.
- **Keep functions short and single-purpose.** If you need a comment explaining
  what a block does, extract it into a function.
- **Logging:** use `structlog` for new pipeline code (see
  [design doc](docs/design/data-pipeline.md)); use Python's `logging` module
  elsewhere.

### Project layout

| Directory      | Purpose                                               |
| -------------- | ----------------------------------------------------- |
| `src/`         | ML code (models, data modules, training, evaluation)  |
| `pipeline/`    | Distributed data pipeline                             |
| `scripts/`     | Standalone utility scripts                            |
| `configs/`     | Hydra YAML configs and pipeline configs               |
| `tests/`       | Test suite (mirrors `src/` and `pipeline/` structure) |
| `docs/design/` | Design documents                                      |

## Testing

### Running tests

```bash
make test-fast      # quick tests — CPU-only; excludes slow, gpu, mps, requires_vst
make test-full-cpu  # all CPU tests (slow + requires_vst included; gpu/mps excluded)
make test-full-gpu  # GPU + CPU tests (mps excluded). Serial — exclusive GPU access
make test-full-mps  # MPS + CPU tests (gpu excluded). Serial — exclusive MPS access
make test-vst-cpu   # VST-only suite (requires_vst, slow included; gpu/mps excluded)
make test-bats      # BATS shell tests (requires bats — see Prerequisites)
make coverage       # tests with coverage report (HTML + terminal)
make benchmark      # performance benchmarks
```

### Test markers

Tests use `pytest` with strict markers. The registered markers are:

| Marker                      | Meaning                             |
| --------------------------- | ----------------------------------- |
| `@pytest.mark.slow`         | Long-running tests                  |
| `@pytest.mark.gpu`          | Requires a GPU                      |
| `@pytest.mark.requires_vst` | Requires Surge XT VST plugin binary |
| `@pytest.mark.r2`           | Requires R2/rclone access           |
| `@pytest.mark.hypothesis`   | Property-based tests (Hypothesis)   |
| `@pytest.mark.pipeline`     | Pipeline integration tests          |
| `@pytest.mark.benchmark`    | Performance benchmarks              |
| `@pytest.mark.docker_smoke` | Smoke tests inside Docker image     |

Tests with missing dependencies (GPU, VST plugin, R2 credentials) are skipped
automatically. This is expected and reported by pytest.

### Writing tests

- Place tests in `tests/`, mirroring the source structure.
- **Test behavior, not implementation details.**
- Use descriptive names: `test_<what>_<condition>_<expected>`.
- Mark slow tests with `@pytest.mark.slow`.

## Commit messages

Commit messages follow **[Conventional Commits](https://www.conventionalcommits.org/)**,
enforced by [gitlint](https://jorisroovers.com/gitlint/).

### Version-bumping prefixes

These prefixes trigger a release via semantic-release:

| Prefix                                | Version bump | When to use                                    |
| ------------------------------------- | ------------ | ---------------------------------------------- |
| `feat:`                               | **minor**    | New user-facing capability                     |
| `fix:`                                | **patch**    | Bug fix                                        |
| `perf:`                               | **patch**    | Performance improvement                        |
| `revert:`                             | **patch**    | Revert a previous commit                       |
| `feat!:` or `BREAKING CHANGE:` footer | **major**    | Breaking change (coordinate with a maintainer) |

### No-bump prefixes

These prefixes do not trigger a release:

| Prefix           | When to use                                                                       |
| ---------------- | --------------------------------------------------------------------------------- |
| `internal-feat:` | New code not yet exposed to users (building toward a feature across multiple PRs) |
| `internal-fix:`  | Fix to internal code not yet exposed to users                                     |
| `docs:`          | Documentation only                                                                |
| `chore:`         | Maintenance (deps, config, etc.)                                                  |
| `ci:`            | CI/CD changes                                                                     |
| `test:`          | Test-only changes                                                                 |
| `refactor:`      | Code restructuring without behavior change                                        |
| `style:`         | Formatting, whitespace                                                            |
| `build:`         | Build system or dependency changes                                                |
| `monitoring:`    | Observability and monitoring changes                                              |

### Guidelines

- Each PR should leave `main` in a valid state -- no dead code, no unhooked
  partial implementations.
- If a PR adds tested internal code that will be wired up later, use
  `internal-feat:`.
- The PR that makes a feature user-facing uses `feat:`.
- Don't contort prefixes to avoid version bumps. If it's user-facing, it's
  `feat:`.

## Pre-commit hooks

This project runs a comprehensive suite of pre-commit hooks. Common failure
modes and how to fix them:

| Hook                    | Failure reason                                      | Fix                                                                                                                                                                                |
| ----------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `interrogate`           | Docstring coverage below 80%                        | Add docstrings to new public functions/classes                                                                                                                                     |
| `pydoclint`             | Docstring args/returns/raises don't match signature | Update the docstring (Sphinx style) or the signature so they agree. To suppress a specific check, add `# noqa: DOCxxx` on the `def`/`class` line (matches flake8/ruff convention). |
| `ruff` (D102/D103/D107) | Public function/method/`__init__` has no docstring  | Add a Sphinx-style docstring. Pydoclint defers "must exist" to ruff's D rules, so this is the gate that catches missing docstrings.                                                |
| `pyright`               | Type errors in touched files                        | Fix type annotations                                                                                                                                                               |
| `gitlint`               | Commit message doesn't follow conventional commits  | Rewrite the commit message (see prefix table above)                                                                                                                                |
| `ruff`                  | Lint violations                                     | Ruff auto-fixes formatting; security/import issues need manual fix                                                                                                                 |
| `no-commit-to-branch`   | Attempted commit to `main`                          | Create a feature branch first                                                                                                                                                      |

If a hook auto-fixes files (ruff, trailing-whitespace, etc.), stage the fixes
and commit again.

## Pull requests

### Branch workflow

1. Create a feature branch from `main`.
2. Make your changes, run `make format`, and commit.
3. Push and open a PR.

### PR requirements

Every PR must:

- **Reference a GitHub issue** using `Closes #N`, `Fixes #N`, `Refs #N`, or
  `Part of #N` in the PR body. Use `Refs #N` (not `Fixes #N`) when the PR is a
  partial fix or workaround, since `Fixes` auto-closes the issue.
- **Pass all CI checks**, including the PR metadata gate which verifies:
  - Issue type label is present
  - Domain label is present
  - Milestone is assigned
  - The linked issue traces to an Epic via the sub-issue hierarchy

### PR title

Use the same conventional commit format as your commit message (e.g.,
`feat(search): add random preset parameter sweep`,
`fix(pipeline): correct shard checksum validation`). The title must stand
on its own — a reader who has not opened the linked issue should be able
to tell from the title alone what part of the system the PR touches and
what concrete change it makes. See `CLAUDE.md` § "PR Titles" for the full
rule and worked examples.

## Code of conduct

Please be respectful and constructive in all interactions. We expect
contributors to act professionally and collaboratively.
