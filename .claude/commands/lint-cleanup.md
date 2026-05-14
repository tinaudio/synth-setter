---
description: "Run the lint-cleanup workflow on a legacy file from issue #25's exclusion list"
argument-hint: <path-to-file>
---

Run the lint-cleanup workflow from
[`.github/agents/lint-cleanup.md`](../../.github/agents/lint-cleanup.md) on
the file `$ARGUMENTS`.

If `$ARGUMENTS` is empty, pick the target by following the canonical
runbook's [**Picking the next file**](../../.github/agents/lint-cleanup.md#picking-the-next-file)
section — do not reproduce its rules here.

Spawn the `lint-cleanup` subagent (defined in
`.claude/agents/lint-cleanup.md`) in an isolated worktree to do the work.
Hand it the path and remind it of the hard rules: one file per PR, no
behavioral changes, branch `chore/lint-cleanup/<name>`, commit prefix
`chore(lint):`, PR body has `Refs #25` (not `Fixes`/`Closes`).
