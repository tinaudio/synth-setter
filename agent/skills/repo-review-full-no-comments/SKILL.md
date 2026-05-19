---
name: repo-review-full-no-comments
description: |-
  Multi-skill review (same fan-out as `/repo-review-full`) that prints the
  aggregated BLOCK/WARN report to the user instead of posting inline comments
  on GitHub. Works against either an open PR or a local branch that has not
  been pushed yet — use it as a pre-PR gate or whenever GitHub side effects
  are undesirable. Requires the tinaudio-synth-setter-skills plugin.
---

# repo-review-full-no-comments — Multi-Skill Review Without Posting

Same analysis as `/repo-review-full`, with two differences:

1. The final delivery step renders findings inline in chat instead of
   submitting a GitHub review.
2. A PR is **not** required. If no PR exists for the current branch, this skill
   reviews the local branch vs. the default branch.

You MUST complete every step in order.

## Step 1: Resolve the target (PR or local branch)

Pick whichever mode applies. Steps 3–6 work the same once the metadata is in
hand.

**PR mode.** Use this when the caller passed an explicit `<N>`, or when
`gh pr view --json number` resolves a PR for the current branch:

```bash
gh pr view <N> --repo "$(gh repo view --json nameWithOwner -q .nameWithOwner)" \
  --json number,headRefOid,baseRefOid,files,title,headRefName,mergeable,mergeStateStatus,statusCheckRollup
```

This is the exact call from `agent/skills/_shared/repo-review-full-analysis.md`
Step 1 — use that file's guidance for parsing it.

**Local-branch mode.** Use this when no `<N>` was passed AND
`gh pr view --json number` fails / returns nothing for the current branch.
Derive the same fields from local git:

```bash
base_ref=$(gh repo view --json defaultBranchRef -q .defaultBranchRef.name)
base_sha=$(git merge-base HEAD "origin/${base_ref}")
head_sha=$(git rev-parse HEAD)
head_ref=$(git rev-parse --abbrev-ref HEAD)
# File list with status (for tdd-refactor R-detection in Step 3)
git diff --name-status "${base_sha}..${head_sha}"
```

Build a synthetic metadata object equivalent to the `gh pr view` JSON:

- `number`: `null` — no PR yet; use the branch name in any user-facing text.
- `headRefOid`: `head_sha`
- `baseRefOid`: `base_sha`
- `headRefName`: `head_ref`
- `title`: `git log -1 --pretty=%s` (informational only).
- `files`: parsed `git diff --name-status` output.
- `mergeable`, `mergeStateStatus`, `statusCheckRollup`: **not available** — Step
  2 handles this.

If there are zero changed files between `base_sha` and `head_sha`, stop and
reply `PASS — no diff vs ${base_ref}`.

## Step 2: Inspect PR health

**PR mode.** Follow `agent/skills/_shared/repo-review-full-analysis.md` Step 2
exactly as written (merge conflicts + failing checks).

**Local-branch mode.** Skip the GitHub-side `mergeable` and `statusCheckRollup`
checks — they don't exist pre-PR. Do not synthesize PR-health BLOCKs and do not
emit a `## PR health` section in Step 6's JSON. Instead, Step 6 prepends a
one-line caveat to the `review_body` so the user knows merge/CI health was not
verified.

## Steps 3–6: Run the shared analysis pipeline

Read and follow `agent/skills/_shared/repo-review-full-analysis.md` Steps 3
through 6 using the metadata from Step 1 above. The shared file owns skill
selection, the parallel fan-out, finding aggregation, and the findings-JSON
construction; it is mode-agnostic.

When the shared file says "the calling skill," that's this skill:

- Use `repo-review-full-no-comments` as the calling-skill name in any
  `[<calling-skill>:block]` prefixes inside the `## PR health` bullets (PR mode
  only — local-branch mode has no PR-health bullets).

- Write the findings JSON to `/tmp/repo-review-full-no-comments-findings.json`.

- For `pr_number` in the JSON: use the resolved PR number in PR mode, or
  `null` in local-branch mode. Do not place branch names in `pr_number`; keep
  branch/target details in `review_body` and, if the shared JSON builder
  supports it, a separate metadata field such as `target`.

- Phrase the `review_body` lead-in to reflect what was reviewed and that
  nothing was posted. In PR mode, write:

  > Multi-skill dry-run review of PR #\<N> — \<K> parallel passes (\<list of skills>). Findings below; no inline comments posted.

  In local-branch mode, write:

  > Multi-skill dry-run review of branch \<head_ref> vs \<base_ref> (no PR yet) — \<K> parallel passes (\<list of skills>). Findings below; no inline comments posted. Merge/CI health not checked — rerun after opening the PR for that.

Return here once Step 6 has produced the JSON payload on disk.

## Step 7: Render the findings to the user

Do NOT invoke `post_review.py`. Do NOT call any `gh api .../reviews` or
`gh pr review` command. This skill has zero GitHub side effects.

Instead, transform the JSON payload at `/tmp/repo-review-full-no-comments-findings.json`
into a Markdown report and print it as your reply to the user. Use this layout:

```markdown
# repo-review-full-no-comments — <target>

<review_body verbatim, including the `## PR health` section if present>

## Inline findings (would be posted by `/repo-review-full`)

### `<path>`
- **L<line>** — <body>
- **L<line>** — <body>

### `<other path>`
- **L<line>** — <body>

## Summary

- B BLOCK, W WARN across K skills
- PR-health flags: <M merge-conflict / F failing-check>  (omit if zero or in local-branch mode)
- <next-step tip>
```

For `<target>` in the header, use `PR #<N>` in PR mode or
`branch <head_ref>` in local-branch mode.

For `<next-step tip>` in the Summary section, use:

- PR mode: `Run /repo-review-full <N> to post these as inline review comments.`
- Local-branch mode: `Open a PR, then run /repo-review-full to post these as inline review comments.`

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
