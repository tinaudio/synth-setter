---
description: "Run the lint-cleanup workflow on a legacy file from issue #25's exclusion list"
argument-hint: <path-to-file>
---

Run the lint-cleanup workflow from
[`.github/agents/lint-cleanup.md`](../../.github/agents/lint-cleanup.md) on
the file `$ARGUMENTS`.

If `$ARGUMENTS` is empty, follow the **"Picking the next file"** section in
the canonical runbook: work the exclusion lists in reverse order of when
each entry was added (most recently added first, LIFO). Use `git blame`
against `.pre-commit-config.yaml` and `pyproject.toml`'s exclusion lines to
rank candidates by introducing-commit date, and skip entries already in
flight on [#25](https://github.com/tinaudio/synth-setter/issues/25). Break
ties by smaller file size.

Spawn the `lint-cleanup` subagent (defined in
`.claude/agents/lint-cleanup.md`) in an isolated worktree to do the work.
Hand it the path and remind it of the hard rules: one file per PR, no
behavioral changes, branch `chore/lint-cleanup/<name>`, commit prefix
`chore(lint):`, PR body has `Refs #25` (not `Fixes`/`Closes`).
