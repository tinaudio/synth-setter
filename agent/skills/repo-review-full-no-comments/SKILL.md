---
name: repo-review-full-no-comments
description: |-
  Multi-skill review (same fan-out as `/repo-review-full`) that prints the
  aggregated BLOCK/WARN report instead of posting inline comments. Routes every
  host harness through Pi and works against an open PR or local branch. Requires
  the tinaudio-synth-setter-skills plugin.
---

# repo-review-full-no-comments — Multi-Skill Review Without Posting

Same analysis as `/repo-review-full`, with two differences:

1. The final delivery step renders findings in chat instead of submitting a
   GitHub review.
2. A PR is **not** required. If no PR exists for the current branch, the
   orchestrator reviews the local branch vs. the default branch.

The foreground dry run posts nothing. On an existing PR, deferred second passes
may later post only new Codex-verified findings through detached aftercare; this
preserves the sub-ten-minute response without dropping slow independent review.

The review implementation is Pi-native. Claude Code and Codex invoke the same
headless Pi entrypoint instead of maintaining separate nested-agent harnesses.

## What you (the main agent) do

1. Capture the target argument: if the command was invoked with an explicit
   `<N>`, keep it; otherwise the orchestrator resolves PR-or-local-branch mode
   itself.

2. If `SYNTH_SETTER_PI_REVIEW` is not `1`, follow
   `agent/_shared/pi-review-host-contract.md` with
   `repo-review-full-no-comments` as the selected skill. Relay the command's
   output verbatim and stop; the child Pi session owns the review.

3. If `SYNTH_SETTER_PI_REVIEW=1`, do not invoke the launcher again. Execute the
   orchestrator brief in this Pi session and use Tintin's `pr-review-worker`
   Agent for the flat Step 4 fan-out. Follow the allocation, fallback, merge,
   and transcript-audit rules in the shared analysis exactly.

4. The agent returns the **full rendered Markdown report** ending in a final
   `Sentinel: <path>` line. Print exactly what the orchestrator returned,
   verbatim — that trailing line already surfaces the sentinel path, so do not
   append any narration of your own. Do not re-run the pipeline.

## Orchestrator agent brief

> You are the orchestrator for a multi-skill dry-run PR review that posts
> nothing to GitHub. Complete every step in order and do not stop early. "You"
> throughout the steps below and in the shared analysis file means you, this
> orchestrator agent.
>
> **Model policy.** Use the shared dynamic routing table and supply every
> worker's model and thinking level explicitly.
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
> If there are zero changed files between `base_sha` and `head_sha`, set
> `is_zero_diff=true`, skip Steps
> 2–6 and go straight to Step 7's **PASS short form**: write the sentinel file
> and return the rendered PASS report ending in `Sentinel: <path>` (note
> `PASS — no diff vs ${base_ref}` in the report body). Do not early-return a
> bare string — the pre-PR gate needs the sentinel on disk.
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
> - Before writing findings, create an invocation-isolated path:
>
>   ```bash
>   python3 agent/_shared/review_sentinel.py findings "${TMPDIR:-/tmp}"
>   ```
>
>   Capture the exact printed path and write this invocation's findings JSON only
>   there. Shell variables do not persist across tool calls, so substitute that
>   exact path in every later command; never use a shared fixed filename. Bash
>   tool calls do not share an `EXIT` trap, so remove the path on every
>   controlled success or failure; an interrupted run leaves only an isolated
>   file in the platform temporary directory and cannot contaminate another run.
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
> A terminal failure after target resolution bypasses this ordinary renderer.
> Follow the shared **Terminal failure delivery** section with
> `--mode no-comments`; the helper writes the canonical blocking HEAD sentinel
> and returns the exact failure report before exiting nonzero.
>
> Do NOT invoke `post_review.py`. Do NOT call any `gh api .../reviews` or
> `gh pr review` command. The foreground step has zero GitHub side effects;
> `repo-review-aftercare.md` owns the narrowly scoped late-finding exception.
>
> Transform the JSON payload at the exact printed findings path into a Markdown
> report. The
> report is **both** written to a sentinel file **and** returned as your final
> message (the main agent prints it for the user). The retained
> `pre-pr-review-gate.sh` parser validates this filename after its local
> PreToolUse registration is restored.
>
> **Render through the deterministic helper.** Do not hand-write the sentinel
> path, reconstruct Markdown, or embed the payload in a generated shell program.
> The helper validates the isolated payload, derives Git/progress state, writes
> the canonical sentinel atomically, removes only that payload, and prints the
> exact foreground deliverable:
>
> ```bash
> ./.venv/bin/python agent/_shared/pi_review_render.py \
>   --payload <exact-findings-json-path> \
>   --target <PR-or-branch-label> --reviewed-head <head-sha> --skill-count <K> \
>   --next-step <caller-specific-tip> --remove-payload
> ```
>
> For the Step 1 zero-diff path, which intentionally has no findings payload,
> invoke the same helper with `--zero-diff --target <target> --reviewed-head <head-sha>` and omit `--payload`, `--skill-count`, `--next-step`, and
> `--remove-payload`.
>
> Execute the applicable form once and return its stdout verbatim. The result ends with the
> absolute or repository-relative canonical `Sentinel: <path>` line. The layout
> below documents helper output; it is not an instruction to generate another
> report.
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
> - Local-branch mode: `Open a PR, then run /repo-review-full to post these as inline review comments if desired. Review sentinel: <REVIEW_PATH>.`
> - If there are any findings or PR-health flags, append: `After remediation and relevant checks, commit and push coherent progress before re-running this review or ending the session. If that is unsafe or impossible, state the blocker instead of retrying unchanged.`
>
> Rules for the rendering:
>
> - Group inline findings by `path`, then list each finding as `**L<line>** — <body>`.
>   Use the same `body` text you put into the JSON (`**[<short-tag>:<severity>]** <description>`).
>
> - Preserve the PR-health bullets from `review_body` verbatim — they are
>   important for human reviewers and easy to lose if you re-summarize.
>
> - **Pi PASS report.** When Pi has no findings or PR-health flags, preserve the
>   complete `## Pi review audit` section from `review_body` between the PASS
>   line and `## Summary`; a successful review must not discard its model,
>   attempt, agent-id, or transcript evidence. The fixed short form below is
>   only for zero-diff reviews, which skip worker allocation and audit rows.
>
> - **PASS short form.** If `is_zero_diff == true`, still write the sentinel
>   file. A non-zero diff with no findings keeps the complete Pi audit above
>   instead of using this short form. The gate's size guard rejects files under
>   200 bytes, and the header + `PASS` line + `Reviewed at:` line are
>   ~130 bytes — pad with a one-line context summary so the total is ≥200
>   bytes. Use this exact template (substitute `<target>` and `<sha>`):
>
>   ```markdown
>   # repo-review-full-no-comments — <target>
>
>   PASS — no findings across all skills (code-health, correctness,
>   comment-hygiene, python-style, shell-style, synth-setter, tdd-impl, ml-test).
>
>   ## Summary
>
>   - 0 BLOCK, 0 WARN
>   - Reviewed at: <sha>
>   - Progress: branch <head_ref>; HEAD <current_head>; upstream <current_upstream>; worktree <worktree_state>; unchanged review count 0.
>   ```
>
> - The sentinel file is the gate's contract; your returned report is the human
>   deliverable. Always produce both.
>
> On ordinary success, `--remove-payload` removes the exact isolated findings
> file. Terminal failure delivery remains responsible for its own cleanup.
>
> **Return value.** Reply with the full Markdown report (the exact content you
> wrote to the sentinel) followed by a final line: `Sentinel: <REVIEW_PATH>`.
> The main agent prints the report and surfaces the path. Return the rendered
> report as data — do not summarize or re-narrate it.

## Notes

- This skill's foreground result is side-effect-free on GitHub. For an existing
  PR, detached aftercare may post one review containing only new Codex-verified
  findings from passes deferred at the response deadline. It rechecks the exact
  head immediately before posting. Local-branch mode remains fully side-effect-free.
- A non-PASS report starts a remediation loop, not a license to retry the same
  review. Follow the non-PASS Summary instruction before another review or a
  handoff. This is advisory so investigation and deliberately uncommitted
  experiments remain possible.
- Like `/repo-review-full`, this skill depends on the
  `tinaudio-synth-setter-skills` plugin being enabled. If a sub-skill
  invocation fails, surface the error — don't silently skip.
- Do not dedupe findings across skills. Keep each skill's signal independent,
  same as `/repo-review-full`.
