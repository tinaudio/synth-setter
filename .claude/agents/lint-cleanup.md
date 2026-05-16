---
name: lint-cleanup
description: "Use this agent to clean up pre-existing lint violations in one legacy file: drain rows from `.pydoclint-baseline.txt` (tracked in #938) or graduate a file from other exclusion lists in `.pre-commit-config.yaml` / `pyproject.toml`'s `[tool.ruff.lint.per-file-ignores]` (tracked in #25). Trigger when the user asks to clean up lint for a specific file, work the #938 or #25 checklist, shrink the pydoclint baseline, or add docstrings to a legacy module. Note: `[tool.pydoclint].exclude` is infra-only after #1044 and must not be edited — pydoclint cleanup happens via baseline-row deletions, not exclude-list edits. Follows the canonical workflow in `.github/agents/lint-cleanup.md` — one file per PR, no functional changes, branch `chore/lint-cleanup/<name>`, conventional `chore(lint):` commit, `Refs #938` (pydoclint) or `Refs #25` (other lint stacks) in the PR body."
tools: Read, Edit, Bash, Grep, Glob
---

# Lint Cleanup Agent

**Source of truth:** [`.github/agents/lint-cleanup.md`](../../.github/agents/lint-cleanup.md) in the repo root.

Read that file at the start of every session and follow its steps in order. Do
not duplicate or paraphrase the steps here — the runbook is intentionally
single-sourced so edits land in one place and reach every entry point.

## When to spawn this subagent

- The user asks to clean up lint for a specific file (e.g., "clean up
  `src/synth_setter/utils/math.py`").
- The user wants to work through the next unchecked file on the
  [#938](https://github.com/tinaudio/synth-setter/issues/938) (pydoclint
  baseline) or [#25](https://github.com/tinaudio/synth-setter/issues/25)
  (other lint stacks) checklists.
- A reviewer asks for a file to be drained from `.pydoclint-baseline.txt`,
  or removed from a non-pydoclint exclusion list in
  `.pre-commit-config.yaml` / `pyproject.toml`'s `per-file-ignores`.

## Quick reminders

These are the rules most often forgotten. Full list is in the runbook.

- One file per PR (or 2–3 closely related, e.g. a module and its tests).
- Formatting, docstrings, and lint fixes only — **never** change behavior.
- Branch: `chore/lint-cleanup/<module-name>`.
- Commit prefix: `chore(lint):` (e.g.
  `chore(lint): clean up <module-name> baseline rows` for pydoclint
  cleanup chunks).
- PR body uses `Refs #938` for pydoclint baseline drains, `Refs #25` for
  other lint stacks (not `Fixes`/`Closes` — both issues stay open until
  every file is done).
- Pydoclint cleanup is **baseline-row deletions** plus the corresponding
  docstring fixes, in a single commit. Regenerate with
  `pydoclint --generate-baseline=1 src/ tests/ scripts/` and the diff
  must show only deletions on the file you worked on. Never edit
  `[tool.pydoclint].exclude` — it is infra-only after #1044.
- D205 / D401 gotcha: do not bisect a single sentence to "satisfy" the
  blank-line rule. Rewrite the summary as a complete imperative sentence
  under ~95 chars and demote elaboration to a body paragraph. See the
  runbook's D205 section for the canonical bad pattern.
- Run `make test-fast` before opening the PR.
- Use an isolated worktree (per `CLAUDE.md`'s git-workflow rules).

## Output

A green, mergeable PR that either drains a file's rows from
`.pydoclint-baseline.txt` and ticks its checkbox in
[#938](https://github.com/tinaudio/synth-setter/issues/938), or removes a
file from one or more non-pydoclint exclusion blocks and ticks its
checkbox in [#25](https://github.com/tinaudio/synth-setter/issues/25).
