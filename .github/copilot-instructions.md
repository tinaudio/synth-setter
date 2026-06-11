# Copilot instructions for synth-setter

This file is read by GitHub Copilot Coding Agent (the autonomous issue → PR
flow) and by Copilot Chat for repo-wide context. It is **not** the primary
contributor guide — [`CLAUDE.md`](../CLAUDE.md) is the source of truth for
project conventions, and every contributor (human or agent) is expected to
follow it. This file points to `CLAUDE.md` and adds Copilot-specific routing
for known agentic workflows.

## Read first

Before opening a PR, read [`CLAUDE.md`](../CLAUDE.md). It covers code
standards, commit conventions (gitlint enforces `feat:`/`fix:`/
`internal-feat:`/`chore:`/...), PR readiness rules, testing commands, and
architecture. A few rules that PRs most often violate:

- Run `make format` before committing.
- Every PR body must link a taxonomy-compliant issue via `Refs #N` /
  `Fixes #N` / `Closes #N` / `Part of #N`. The `pr-metadata-gate.yml` CI
  check enforces this; see `CLAUDE.md` for the authoritative list and the
  semantics of each keyword.
- PR titles and commit messages must follow conventional commits — gitlint
  will reject otherwise.
- Never add "Generated with Claude Code", "Generated with Copilot", or
  similar attribution footers to commits, PRs, or comments.

## Agent runbooks under `.github/agents/`

Specific recurring tasks have dedicated runbooks in `.github/agents/`. When an
assigned issue matches one of these workflows, follow the runbook step by
step:

- [`.github/agents/lint-cleanup.md`](agents/lint-cleanup.md) — clean up
  pre-existing lint violations in one legacy file. Two cleanup lanes:

  - **Pydoclint baseline drains** (tracked in
    [#938](https://github.com/tinaudio/synth-setter/issues/938)): pick a
    file with rows in `.pydoclint-baseline.txt`, fix the docstring
    violations, regenerate the baseline with
    `pydoclint --generate-baseline=1 src/ tests/ scripts/`, and commit
    the docstring fixes plus the baseline-row deletions together. PR
    body uses `Refs #938`. **Do not edit `[tool.pydoclint].exclude`** —
    it is infra-only after
    [#1044](https://github.com/tinaudio/synth-setter/pull/1044) and must
    not be touched by this workflow.
  - **Other exclusion lists** (tracked in
    [#25](https://github.com/tinaudio/synth-setter/issues/25)): files in
    `.pre-commit-config.yaml`'s `exclude:` blocks (pyright, interrogate,
    shellcheck, codespell) and `pyproject.toml`'s
    `[tool.ruff.lint.per-file-ignores]`. Graduate a file by clearing it
    from every non-pydoclint list it appears in. PR body uses `Refs #25`.

  **Trigger:** any issue that references #938 or #25 as its parent, or
  whose body says "apply `.github/agents/lint-cleanup.md` to `<file>`".

When in doubt, if an issue does not match a known runbook, follow the general
contributor flow from `CLAUDE.md` — branch, commit, run `make test-fast`,
open a PR with a linked issue, address review.
