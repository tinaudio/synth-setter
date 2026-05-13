---
name: repo-review-full-no-comments
description: |
  Multi-skill PR review (same fan-out as `/repo-review-full`) that prints the
  aggregated BLOCK/WARN report to the user instead of posting inline comments
  on GitHub. Use this for a dry run, for iterating on a PR before reviewers see
  it, or any time GitHub side effects are undesirable. Requires the
  tinaudio-synth-setter-skills plugin.
---

# repo-review-full-no-comments — Multi-Skill PR Review Without Posting

Same analysis as `/repo-review-full`. The only difference is the final delivery
step: this skill renders the findings inline in chat instead of submitting a
GitHub review.

You MUST complete every step in order.

## Steps 1–6: Run the shared analysis pipeline

Read and follow `.claude/skills/_shared/repo-review-full-analysis.md` Steps 1
through 6. That file owns PR resolution, PR-health inspection, skill selection,
the parallel fan-out, finding aggregation, and the findings-JSON construction.

When the shared file says "the calling skill," that's this skill:

- Use `repo-review-full-no-comments` as the calling-skill name in any
  `[<calling-skill>:block]` prefixes inside the `## PR health` bullets.
- Write the findings JSON to `/tmp/repo-review-full-no-comments-findings.json`.
- Phrase the `review_body` lead-in to reflect that nothing has been posted — the
  user is reading the findings directly. For example:
  `Multi-skill dry-run review of PR #<N> — <K> parallel passes (<list of skills>). Findings below; no inline comments posted.`

Return here once Step 6 has produced the JSON payload on disk.

## Step 7: Render the findings to the user

Do NOT invoke `post_review.py`. Do NOT call any `gh api .../reviews` or
`gh pr review` command. This skill has zero GitHub side effects.

Instead, transform the JSON payload at `/tmp/repo-review-full-no-comments-findings.json`
into a Markdown report and print it as your reply to the user. Use this layout:

````markdown
# repo-review-full-no-comments — PR #<N>

<review_body verbatim, including the `## PR health` section if present>

## Inline findings (would be posted by `/repo-review-full`)

### `<path>`
- **L<line>** — <body>
- **L<line>** — <body>

### `<other path>`
- **L<line>** — <body>

## Summary

- B BLOCK, W WARN across K skills
- PR-health flags: <M merge-conflict / F failing-check>  (omit if zero)
- Run `/repo-review-full <N>` to post these as inline review comments.
````

Rules for the rendering:

- Group inline findings by `path`, then list each finding as `**L<line>** — <body>`.
  Use the same `body` text you put into the JSON (`**[<short-tag>:<severity>]** <description>`).
- Preserve the PR-health bullets from `review_body` verbatim — they are
  important for human reviewers and easy to lose if you re-summarize.
- If the JSON has no findings AND no PR-health flags, print `PASS — no findings`
  and stop.
- Do not write the rendered report to a file unless the user asks. The chat
  response is the delivery surface.

## Notes

- This skill is intentionally side-effect-free on GitHub. If a future caller
  wants the comments posted after all, they can rerun with `/repo-review-full`
  — the analysis is deterministic enough that re-running is cheap relative to
  the value of an explicit "no, don't post" mode.
- Like `/repo-review-full`, this skill depends on the
  `tinaudio-synth-setter-skills` plugin being enabled. If a sub-skill
  invocation fails, surface the error — don't silently skip.
- Do not dedupe findings across skills. Keep each skill's signal independent,
  same as `/repo-review-full`.
