---
name: pr-readiness
description: >-
  Drive a pushed PR through the four readiness gates until it is genuinely
  ready — watch CI, fix red, confirm mergeable, reply inline to every open
  review comment, and wait out Copilot's re-review. Use this skill after every
  push to a PR branch, when the pr-readiness-stop hook blocks a turn, or when
  the user says "drive the PR to ready", "is this PR ready", "finish the
  readiness loop", "/pr-readiness", or anything about getting a PR past CI +
  review + Copilot. Implements docs/pr-readiness-loop.md as an invocable
  procedure.
---

# pr-readiness — drive the PR readiness loop

Run the loop in [`docs/pr-readiness-loop.md`](../../../docs/pr-readiness-loop.md)
until all four gates hold. That doc is canonical — read it for the exact
commands, endpoints, and traps. This skill is the invocable driver; do not
duplicate the doc's prose, follow it.

"I pushed the fix" is **not** "the PR is ready." Do not stop after the first
push.

## The four gates

A PR is ready only when **all four** hold (AND-ed):

1. **CI fully green** — every required and optional check passing. Pending or
   errored counts as not ready.
2. **`mergeable=MERGEABLE`** — `UNKNOWN` and `CONFLICTING` both fail.
3. **Every open review comment has an inline reply** — humans and Copilot.
4. **No fresh Copilot findings since the last push.**

## Procedure

Resolve the PR once up front:

```bash
PR=$(gh pr view "$(git branch --show-current)" --json number -q .number)
```

Then iterate until all four gates hold. Use `/loop` for the waiting steps
(e.g. `/loop 2m gh pr checks "$PR"`).

1. **Watch CI** — `gh pr checks "$PR" --watch`. On any failure, diagnose, fix,
   commit, push, and return to step 1. Never move on with red CI.

2. **Check mergeability** — `gh pr view "$PR" --json mergeable -q .mergeable`:

   - `CONFLICTING` → rebase or merge the base branch, resolve, push, back to 1.
   - `UNKNOWN` → GitHub is still computing; poll again.
   - `MERGEABLE` → continue.

3. **Reply inline to every open review comment.** Delegate to
   `/pr-review-resolver`, which triages each comment, fixes what's actionable,
   and replies inline with the fix-commit SHA or a justification. If any reply
   required a code change, push and return to step 1.

4. **Wait for Copilot's post-push review** (~60s, allow up to 15 min). Check
   both the inline-comments and top-level reviews endpoints
   (docs/pr-readiness-loop.md step 6). Filter to findings newer than your last
   push. Address new actionable findings as in step 3, then loop. A "no
   findings" review is silence, not a comment. If 15 min elapse with no
   activity, manually re-request Copilot per step 6a (at most once).

5. **Done** only when all four gates hold simultaneously.

## Relationship to the Stop hook

`agent/hooks/pr-readiness-stop.sh` blocks the turn from ending while gates 1-2
fail; it cannot decide gates 3-4 in bash, so it points here. Running this skill
to completion is what clears the block.
