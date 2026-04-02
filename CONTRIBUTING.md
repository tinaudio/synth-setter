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
- **Git** (with submodule support)
- **[bats](https://github.com/bats-core/bats-core)** (optional, for shell tests:
  `brew install bats-core` on macOS or `apt-get install bats` on Debian/Ubuntu)

### Clone and install

The repository uses a git submodule for shared skills (configured with an SSH
URL). Always clone with `--recurse-submodules`:

```bash
git clone --recurse-submodules https://github.com/tinaudio/synth-setter.git
cd synth-setter
```

If you already cloned without that flag:

```bash
git submodule update --init
```

> **Note:** The submodule in `.gitmodules` uses an SSH URL
> (`git@github.com:tinaudio/skills.git`). If you don't have SSH keys configured
> for GitHub, override the URL to use HTTPS:
>
> ```bash
> git config submodule..claude/skills.url https://github.com/tinaudio/skills.git
> git submodule update --init
> ```

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
make test           # quick tests (excludes slow and VST-dependent tests)
```

## Code standards

### Formatting and linting

All formatting and linting is enforced automatically by **pre-commit hooks** on
every commit. The key tools are:

- **[Ruff](https://docs.astral.sh/ruff/)** for linting (rules: E, F, I, S, T,
  UP, W) and formatting (line length 99)
- **[Pyright](https://microsoft.github.io/pyright/)** for static type checking
- **[interrogate](https://interrogate.readthedocs.io/)** for docstring coverage
  (minimum 80%)
- **[docformatter](https://github.com/PyCQA/docformatter)** for docstring
  normalization (Sphinx style)
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
make test           # quick tests -- excludes slow and requires_vst markers
make test-full      # all tests (some require GPU or VST plugins)
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

| Hook                  | Failure reason                                     | Fix                                                                |
| --------------------- | -------------------------------------------------- | ------------------------------------------------------------------ |
| `interrogate`         | Docstring coverage below 80%                       | Add docstrings to new public functions/classes                     |
| `pyright`             | Type errors in touched files                       | Fix type annotations                                               |
| `gitlint`             | Commit message doesn't follow conventional commits | Rewrite the commit message (see prefix table above)                |
| `ruff`                | Lint violations                                    | Ruff auto-fixes formatting; security/import issues need manual fix |
| `no-commit-to-branch` | Attempted commit to `main`                         | Create a feature branch first                                      |

If a hook auto-fixes files (ruff, trailing-whitespace, etc.), stage the fixes
and commit again.

## Plumb (spec/test/code sync)

This project uses [Plumb](https://github.com/tinaudio/plumb) to keep the spec,
tests, and source code in sync. If you're working on changes that touch
spec-tracked files, Plumb may intercept your commit and present decisions for
review.

### What happens on commit

When Plumb detects changes to tracked files, it pauses the commit and presents
pending decisions. Each decision can be:

- **Approved** -- the change is accepted and synced to the spec
- **Ignored** -- the change is noted but not synced
- **Rejected** -- the change is rolled back

### Useful commands

```bash
plumb status        # current alignment between spec, tests, and code
plumb diff          # preview what Plumb will capture on next commit
plumb sync          # update spec and generate tests after decisions are resolved
plumb coverage      # identify what needs to be implemented or tested
```

### Escape hatch

If Plumb intercepts a commit that doesn't touch spec-relevant files (e.g., a
docs-only change), you can skip it with the `PLUMB_SKIP` environment variable
(checked by the Plumb pre-commit hook in `.git/hooks/pre-commit`):

```bash
PLUMB_SKIP=1 git commit -m "docs: your message"
```

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
`feat: add parameter search`, `fix: correct shard validation`).

## Code of conduct

Please be respectful and constructive in all interactions. We expect
contributors to act professionally and collaboratively.
