---
name: lint-cleanup
description: "Use this agent to clean up pre-existing lint violations in one legacy file from the synth-setter exclusion lists tracked in issue #25. Trigger when the user asks to clean up lint for a specific file, work the #25 checklist, remove a file from `.pre-commit-config.yaml`'s `exclude` blocks, or add docstrings to a legacy module. Follows the canonical workflow in `.github/agents/lint-cleanup.md` — one file per PR, no functional changes, branch `chore/lint-cleanup/<name>`, conventional `chore(lint):` commit, `Refs #25` in the PR body."
tools: Read, Edit, Bash, Grep, Glob
---

# Lint Cleanup Agent

**Source of truth:** [`.github/agents/lint-cleanup.md`](../../.github/agents/lint-cleanup.md) in the repo root.

Read that file at the start of every session and follow its steps in order. Do
not duplicate or paraphrase the steps here — the runbook is intentionally
single-sourced so edits land in one place and reach every entry point.

## When to spawn this subagent

- The user asks to clean up lint for a specific file (e.g., "clean up
  `src/utils/math.py`").
- The user wants to work through the next unchecked file on the
  [#25](https://github.com/tinaudio/synth-setter/issues/25) checklist.
- A reviewer asks for a file to be removed from `.pre-commit-config.yaml`'s
  `exclude` blocks (or `pyproject.toml`'s `per-file-ignores`).

## Quick reminders

These are the rules most often forgotten. Full list is in the runbook.

- One file per PR (or 2–3 closely related, e.g. a module and its tests).
- Formatting, docstrings, and lint fixes only — **never** change behavior.
- Branch: `chore/lint-cleanup/<module-name>`.
- Commit prefix: `chore(lint):`.
- PR body uses `Refs #25` (not `Fixes`/`Closes` — #25 stays open until every
  file is done).
- Run `make test-fast` before opening the PR.
- Use an isolated worktree (per `CLAUDE.md`'s git-workflow rules).

## Output

A green, mergeable PR that removes one file from one or more exclusion blocks
and ticks its checkbox in [#25](https://github.com/tinaudio/synth-setter/issues/25).
