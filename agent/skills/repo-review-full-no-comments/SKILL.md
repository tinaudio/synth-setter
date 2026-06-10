---
name: repo-review-full-no-comments
description: |-
  Multi-skill review (same fan-out as `/repo-review-full`) that prints the
  aggregated BLOCK/WARN report to the user instead of posting inline comments
  on GitHub. Spawns one orchestrator agent that runs the pipeline and returns
  the rendered report. Works against either an open PR or a local branch that
  has not been pushed yet — use it as a pre-PR gate or whenever GitHub side
  effects are undesirable. Requires the tinaudio-synth-setter-skills plugin.
---

# repo-review-full-no-comments — Multi-Skill Review Without Posting

Same analysis as `/repo-review-full`, with two differences:

1. The final delivery step renders findings in chat instead of submitting a
   GitHub review.
2. A PR is **not** required. If no PR exists for the current branch, the
   orchestrator reviews the local branch vs. the default branch.

The entire pipeline runs inside **one** spawned orchestrator agent. As the main
agent you launch that agent and relay its result.

## What you (the main agent) do

1. Capture the target argument: if the command was invoked with an explicit
   `<N>`, keep it; otherwise the orchestrator resolves PR-or-local-branch mode
   itself.
2. Spawn exactly **one** `general-purpose` agent. Its prompt is the entire
   "## Orchestrator agent brief" section below. Only substitute when an explicit
   `<N>` was passed — replace `<N>` with that number; otherwise pass the brief
   verbatim (Step 1 of the brief already resolves PR-or-local-branch mode). Do
   not otherwise edit the brief.
3. The agent returns the **full rendered Markdown report** ending in a final
   `Sentinel: <path>` line. Print exactly what the orchestrator returned,
   verbatim — that trailing line already surfaces the sentinel path, so do not
   append any narration of your own. Do not re-run the pipeline.

Spawn only this one orchestrator. It launches its own parallel per-skill review
sub-agents; you never launch those directly.

## Orchestrator agent brief

> You are the orchestrator for a multi-skill dry-run PR review that posts
> nothing to GitHub. Complete every step in order and do not stop early. "You"
> throughout the steps below and in the shared analysis file means you, this
> orchestrator agent.
>
> ### Step 1: Resolve the target (PR or local branch)
>
> Pick whichever mode applies. Steps 3–6 work the same once the metadata is in
> hand.
>
> **PR mode.** Use this when an explicit `<N>` was passed, or when
> `gh pr view --json number` resolves a PR for the current branch. If no `<N>`
> was passed, first resolve it from the current branch
> (`N=$(gh pr view --json number -q .number)`) and use that value wherever `<N>`
> appears below — never run the command with the literal `<N>` placeholder:
>
> ```bash
> gh pr view <N> --repo "$(gh repo view --json nameWithOwner -q .nameWithOwner)" \
>   --json number,headRefOid,baseRefOid,files,title,headRefName,mergeable,mergeStateStatus,statusCheckRollup
> ```
>
> This is the exact call from `agent/skills/_shared/repo-review-full-analysis.md`
> Step 1 — use that file's guidance for parsing it.
>
> **Local-branch mode.** Use this when no `<N>` was passed AND
> `gh pr view --json number` fails / returns nothing for the current branch.
> Derive the same fields from local git:
>
> ```bash
> base_ref=$(gh repo view --json defaultBranchRef -q .defaultBranchRef.name)
> base_sha=$(git merge-base HEAD "origin/${base_ref}")
> head_sha=$(git rev-parse HEAD)
> head_ref=$(git rev-parse --abbrev-ref HEAD)
> # File list with status (for tdd-refactor R-detection in Step 3)
> git diff --name-status "${base_sha}..${head_sha}"
> ```
>
> Build a synthetic metadata object equivalent to the `gh pr view` JSON:
>
> - `number`: `null` — no PR yet; use the branch name in any user-facing text.
> - `headRefOid`: `head_sha`
> - `baseRefOid`: `base_sha`
> - `headRefName`: `head_ref`
> - `title`: `git log -1 --pretty=%s` (informational only).
> - `files`: parsed `git diff --name-status` output.
> - `mergeable`, `mergeStateStatus`, `statusCheckRollup`: **not available** —
>   Step 2 handles this.
>
> If there are zero changed files between `base_sha` and `head_sha`, stop and
> return `PASS — no diff vs ${base_ref}`.
>
> ### Step 2: Inspect PR health
>
> **PR mode.** Follow `agent/skills/_shared/repo-review-full-analysis.md` Step 2
> exactly as written (merge conflicts + failing checks).
>
> **Local-branch mode.** Skip the GitHub-side `mergeable` and
> `statusCheckRollup` checks — they don't exist pre-PR. Do not synthesize
> PR-health BLOCKs and do not emit a `## PR health` section in Step 6's JSON.
> Instead, Step 6 prepends a one-line caveat to the `review_body` so the user
> knows merge/CI health was not verified.
>
> ### Steps 3–6: Run the shared analysis pipeline
>
> Read and follow `agent/skills/_shared/repo-review-full-analysis.md` Steps 3
> through 6 using the metadata from Step 1 above. The shared file owns skill
> selection, the parallel fan-out, finding aggregation, and the findings-JSON
> construction; it is mode-agnostic. When it says "the calling skill," that is
> `repo-review-full-no-comments`:
>
> - Use `repo-review-full-no-comments` as the calling-skill name in any
>   `[<calling-skill>:block]` prefixes inside the `## PR health` bullets (PR mode
>   only — local-branch mode has no PR-health bullets).
>
> - Write the findings JSON to `/tmp/repo-review-full-no-comments-findings.json`.
>
> - For `pr_number` in the JSON: use the resolved PR number in PR mode, or
>   `null` in local-branch mode. Do not place branch names in `pr_number`; keep
>   branch/target details in `review_body` and, if the shared JSON builder
>   supports it, a separate metadata field such as `target`.
>
> - Phrase the `review_body` lead-in to reflect what was reviewed and that
>   nothing was posted. In PR mode, write:
>
>   > Multi-skill dry-run review of PR #\<N> — \<K> parallel passes (\<list of skills>). Findings below; no inline comments posted.
>
>   In local-branch mode, write:
>
>   > Multi-skill dry-run review of branch \<head_ref> vs \<base_ref> (no PR yet) — \<K> parallel passes (\<list of skills>). Findings below; no inline comments posted. Merge/CI health not checked — rerun after opening the PR for that.
>
> ### Step 7: Render the findings — write the sentinel file **and** return the report
>
> Do NOT invoke `post_review.py`. Do NOT call any `gh api .../reviews` or
> `gh pr review` command. This step has zero GitHub side effects.
>
> Transform the JSON payload at
> `/tmp/repo-review-full-no-comments-findings.json` into a Markdown report. The
> report is **both** written to a sentinel file **and** returned as your final
> message (the main agent prints it for the user). The `pre-pr-review-gate.sh`
> PreToolUse hook validates the path supplied via `REVIEW_FULL=<path>` on
> `gh pr create` by parsing this filename.
>
> **Compute the sentinel path** — the format is owned by
> `agent/_shared/review_sentinel.py` (single source of truth shared with the
> gate hook):
>
> ```bash
> REVIEW_PATH=$(python3 agent/_shared/review_sentinel.py path "$(git rev-parse HEAD)")
> mkdir -p "$(dirname "$REVIEW_PATH")"
> ```
>
> The result is of the form
> `.agent-reviews/repo-review-full-no-comments.<40-char-sha>.md`. **Do not
> hand-write the filename** — always go through the helper.
>
> **Write the report to `$REVIEW_PATH`** using this layout:
>
> ```markdown
> # repo-review-full-no-comments — <target>
>
> <review_body verbatim, including the `## PR health` section if present>
>
> ## Inline findings (would be posted by `/repo-review-full`)
>
> ### `<path>`
> - **L<line>** — <body>
> - **L<line>** — <body>
>
> ### `<other path>`
> - **L<line>** — <body>
>
> ## Summary
>
> - B BLOCK, W WARN across K skills
> - PR-health flags: <M merge-conflict / F failing-check>  (omit if zero or in local-branch mode)
> - Reviewed at: <full-sha-from-git-rev-parse-HEAD>
> - <next-step tip>
> ```
>
> For `<target>` in the header, use `PR #<N>` in PR mode or
> `branch <head_ref>` in local-branch mode.
>
> For `<next-step tip>` in the Summary section, use (substitute
> `<REVIEW_PATH>` with the actual path computed above — do not emit the literal
> placeholder):
>
> - PR mode: `Run /repo-review-full <N> to post these as inline review comments.`
> - Local-branch mode: `Open a PR with REVIEW_FULL=<REVIEW_PATH> in the command. Then run /repo-review-full to post these as inline review comments if desired.`
>
> Rules for the rendering:
>
> - Group inline findings by `path`, then list each finding as `**L<line>** — <body>`.
>   Use the same `body` text you put into the JSON (`**[<short-tag>:<severity>]** <description>`).
>
> - Preserve the PR-health bullets from `review_body` verbatim — they are
>   important for human reviewers and easy to lose if you re-summarize.
>
> - **PASS short form.** If the JSON has no findings AND no PR-health flags,
>   still write the sentinel file. The gate's size guard rejects files under
>   200 bytes, and the header + `PASS` line + `Reviewed at:` line are
>   ~130 bytes — pad with a one-line context summary so the total is ≥200
>   bytes. Use this exact template (substitute `<target>` and `<sha>`):
>
>   ```markdown
>   # repo-review-full-no-comments — <target>
>
>   PASS — no findings across all skills (code-health, comment-hygiene,
>   python-style, shell-style, synth-setter, tdd-impl, ml-test).
>
>   ## Summary
>
>   - 0 BLOCK, 0 WARN
>   - Reviewed at: <sha>
>   ```
>
> - The sentinel file is the gate's contract; your returned report is the human
>   deliverable. Always produce both.
>
> **Return value.** Reply with the full Markdown report (the exact content you
> wrote to the sentinel) followed by a final line: `Sentinel: <REVIEW_PATH>`.
> The main agent prints the report and surfaces the path. Return the rendered
> report as data — do not summarize or re-narrate it.

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
