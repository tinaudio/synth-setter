---
name: repo-review-full
description: |-
  Full multi-skill PR review. Fans out one parallel agent per applicable plugin
  checklist (selection rules in the shared analysis file) and posts every
  diff-anchored BLOCK/WARN as an individual unresolved inline PR review comment;
  non-diff findings (merge conflicts, failing checks) go in a `## PR health`
  section in the review body. Requires the tinaudio-synth-setter-skills plugin.
---

# repo-review-full — Multi-Skill Parallel PR Review

You MUST complete every step in order.

## Steps 1–6: Run the shared analysis pipeline

Read and follow `agent/skills/_shared/repo-review-full-analysis.md` Steps 1
through 6. That file owns PR resolution, PR-health inspection, skill selection,
the parallel fan-out, finding aggregation, and the findings-JSON construction.

When the shared file says "the calling skill," that's this skill:

- Use `repo-review-full` as the calling-skill name in any `[<calling-skill>:block]`
  prefixes inside the `## PR health` bullets.
- Write the findings JSON to `/tmp/repo-review-full-findings.json`.
- Phrase the `review_body` lead-in to reflect that every BLOCK/WARN is being
  posted as an unresolved inline thread (sample wording is already in the
  shared file).
- Set the top-level `"event"` field in the JSON payload: `REQUEST_CHANGES` if
  any finding is a BLOCK (any `[*:block]`, including the folded PR-health
  BLOCKs), else `COMMENT` if any WARN exists, else `APPROVE`. The self-review
  COMMENT fallback (when the bot is the PR author) is automatic in
  `post_review.py`.

Return here once Step 6 has produced the JSON payload on disk.

## Step 7: Submit the review

```bash
python3 agent/skills/_shared/post_review.py < /tmp/repo-review-full-findings.json
```

`post_review.py`:

- Anchors each finding to its target line if that line falls inside a diff hunk.
- Falls back to the nearest in-hunk line on the same file with a cross-ref note prepended to the body.
- Rolls orphan findings (file outside the diff) into the review body under `## Findings on files outside the diff`.
- Submits with the payload's `event` (REQUEST_CHANGES / COMMENT / APPROVE). On a self-review 422 (the bot is the PR author) it retries once as `event=COMMENT` with a `⛔ N BLOCKING finding(s)` banner prepended, so the blocking intent stays visible. Threads stay unresolved.

Report the helper's `html_url` back to the user along with a one-line summary (`Posted N findings: B BLOCK + W WARN across K skills; PR-health flags: <M merge-conflict / F failing-check>`). If PR-health found nothing, drop the trailing `; PR-health flags: ...` clause.

## When to use the no-comments sibling instead

`/repo-review-full-no-comments` runs the same analysis pipeline (delegating
Steps 3–6 to the shared file; it owns Steps 1–2 itself so it can also run
against a local branch with no PR open) but prints the aggregated report to
the user instead of posting inline comments. Reach for it when you want a
local dry-run of the review (no GitHub side effects), as a pre-PR gate before
the branch is pushed, when you're iterating on a PR before it's ready for
reviewers, or when posting publicly is undesirable for any reason.
