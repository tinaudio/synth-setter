---
name: repo-review-full
description: |-
  Full multi-skill PR review. Spawns one orchestrator agent that fans out a
  parallel agent per applicable plugin checklist (selection rules in the shared
  analysis file) and posts every diff-anchored BLOCK/WARN as an individual
  unresolved inline PR review comment; non-diff findings (merge conflicts,
  failing checks) go in a `## PR health` section in the review body. Requires
  the tinaudio-synth-setter-skills plugin.
---

# repo-review-full — Multi-Skill Parallel PR Review

The entire pipeline runs inside **one** spawned orchestrator agent. As the main
agent you launch that agent and relay its result — you do NOT resolve the PR,
fan out the reviews, aggregate findings, or call `post_review.py` yourself.

## What you (the main agent) do

1. Capture the PR argument: if the command was invoked with an explicit `<N>`,
   keep it; otherwise the orchestrator resolves the PR from the current branch.
2. Spawn exactly **one** `general-purpose` agent. Its prompt is the entire
   "## Orchestrator agent brief" section below. Only substitute when an explicit
   `<N>` was passed — replace `<N>` with that number; otherwise pass the brief
   verbatim (it already tells the orchestrator to resolve the PR from the
   current branch). Do not otherwise edit the brief.
3. Relay the agent's returned `html_url` and one-line summary to the user
   verbatim. Do not re-run or second-guess the pipeline.

Spawn only this one orchestrator. It launches its own parallel per-skill review
sub-agents; you never launch those directly.

## Orchestrator agent brief

> You are the orchestrator for a full multi-skill PR review. Complete every step
> in order and do not stop early. "You" throughout the steps below and in the
> shared analysis file means you, this orchestrator agent.
>
> **Steps 1–6 — run the shared analysis pipeline.** Read and follow
> `agent/skills/_shared/repo-review-full-analysis.md` Steps 1 through 6. That
> file owns PR resolution, PR-health inspection, skill selection, the parallel
> fan-out, finding aggregation, and findings-JSON construction. When it says
> "the calling skill," that is `repo-review-full`:
>
> - PR number: `<N>`. If no explicit number was provided, first resolve it from
>   the current branch (`N=$(gh pr view --json number -q .number)`) and use that
>   value wherever `<N>` appears in the shared analysis commands (e.g.
>   `gh pr view <N> ...`) — never run a command with the literal `<N>`
>   placeholder.
> - Use `repo-review-full` as the calling-skill name in any
>   `[<calling-skill>:block]` prefixes inside the `## PR health` bullets.
> - Write the findings JSON to `/tmp/repo-review-full-findings.json`.
> - Phrase the `review_body` lead-in so every BLOCK/WARN reads as being posted
>   as an unresolved inline thread (sample wording is in the shared file).
> - Set the top-level `"event"` field: `REQUEST_CHANGES` if any finding is a
>   BLOCK (any `[*:block]`, including folded PR-health BLOCKs), else `COMMENT`
>   if any WARN exists, else `APPROVE`. The self-review COMMENT fallback (when
>   the bot is the PR author) is automatic in `post_review.py`.
>
> **Step 7 — submit the review.**
>
> ```bash
> python3 agent/skills/_shared/post_review.py < /tmp/repo-review-full-findings.json
> ```
>
> `post_review.py`:
>
> - Anchors each finding to its target line if that line falls inside a diff hunk.
> - Falls back to the nearest in-hunk line on the same file with a cross-ref note prepended to the body.
> - Rolls orphan findings (file outside the diff) into the review body under `## Findings on files outside the diff`.
> - Submits with the payload's `event` (REQUEST_CHANGES / COMMENT / APPROVE). On a self-review 422 (the bot is the PR author) it retries once as `event=COMMENT` with an event-aware intent banner prepended — `⛔ N BLOCKING finding(s) — changes required` for a REQUEST_CHANGES downgrade, `✅ No findings` for an APPROVE downgrade — so the original intent stays visible. Threads stay unresolved.
>
> **Return value.** Reply with ONLY the helper's `html_url` and a one-line
> summary: `Posted N findings: B BLOCK + W WARN across K skills; PR-health flags: <M merge-conflict / F failing-check>`. If PR-health found nothing,
> drop the trailing `; PR-health flags: ...` clause. This text is the main
> agent's deliverable — return data, not narration.

## When to use the no-comments sibling instead

`/repo-review-full-no-comments` runs the same analysis pipeline through its own
orchestrator agent (delegating Steps 3–6 to the shared file; it owns Steps 1–2
itself so it can also run against a local branch with no PR open) but prints the
aggregated report to the user instead of posting inline comments. Reach for it
when you want a local dry-run of the review (no GitHub side effects), as a
pre-PR gate before the branch is pushed, when you're iterating on a PR before
it's ready for reviewers, or when posting publicly is undesirable for any
reason.
