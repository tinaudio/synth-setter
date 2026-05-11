---
description: Run the lint-cleanup workflow on a legacy file from issue #25's exclusion list
argument-hint: <path-to-file>
---

Run the lint-cleanup workflow from
[`.github/agents/lint-cleanup.md`](../../.github/agents/lint-cleanup.md) on
the file `$ARGUMENTS`.

If `$ARGUMENTS` is empty, read the checklist in
[#25](https://github.com/tinaudio/synth-setter/issues/25) and pick the next
unchecked file (prefer the smallest by line count for a quick win).

Spawn the `lint-cleanup` subagent (defined in
`.claude/agents/lint-cleanup.md`) in an isolated worktree to do the work.
Hand it the path and remind it of the hard rules: one file per PR, no
behavioral changes, branch `chore/lint-cleanup/<name>`, commit prefix
`chore(lint):`, PR body has `Refs #25` (not `Fixes`/`Closes`).
