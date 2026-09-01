# Deferred Pi review follow-up

Process only the deferred passes in the runtime manifest named by the launch
prompt. The Python supervisor, not this model, owns the canonical
`<foreground-manifest>.result.json`. It has already validated any adopted
foreground outputs and named those rows in the launch prompt. Rows with an
incomplete known foreground owner fail before this session starts; remaining
rows come from legacy manifests without ownership handles. Never repeat a pass
not present in the runtime manifest.

1. Validate the runtime manifest before using it:

   ```bash
   ./.venv/bin/python agent/_shared/run_pi_review_follow_up.py <manifest> --dry-run
   ```

2. Fetch the PR's current `headRefOid`, state, and author login. If the PR is not
   open or its head differs from `head_sha`, record `stale` in the runtime result
   and exit without posting.

3. Re-fetch the complete review history so foreground findings and author replies
   posted after the original manifest are visible to deferred workers. Use the
   runtime assignment directory and substitute the manifest's repo, PR number,
   and the author login from Step 2:

   ```bash
   manifest=<runtime-manifest-path>
   assignment_dir="${manifest%.json}.assignments"
   mkdir -p "$assignment_dir"
   review_comments="$assignment_dir/pr-review-comments.json"
   review_history="$assignment_dir/pr-review-history.md"
   pr_author=<PR-author-login-from-Step-2>
   set -o pipefail
   history_fetched=false
   for attempt in 1 2 3; do
     if gh api --paginate "repos/${repo}/pulls/${pr_number}/comments?per_page=100" \
       --jq '.[]' | jq -s '.' > "$review_comments"; then
       history_fetched=true
       break
     fi
     sleep "$((attempt * 2))"
   done
   if [[ $history_fetched != true ]]; then
     printf 'Failed to fetch complete PR review history after 3 attempts.\n' >&2
     exit 1
   fi
   "${PI_REVIEW_PYTHON}" agent/_shared/pi_review_routing.py review-history \
     --input "$review_comments" --author "$pr_author" --output "$review_history"
   ```

   For each adopted row named in the launch prompt, extract and validate its
   existing `output_path`, use that report, and do not launch its pass again.
   For every other `deferred_passes` row, generate the assignment with
   `pi_review_routing.py worker-prompt --review-history "$review_history"`,
   launch one `pr-review-worker` using the row's exact pinned model and thinking,
   and validate its output with `extract-report` and `validate-report`. Exactly
   one model call owns a pass; never launch a second owner for the same row.
   A history-fetch or parse failure fails the follow-up instead of reverting to a
   history-blind review.

4. If strict validation fails after envelope extraction, generate the
   diagnostic with `pi_review_routing.py repair-prompt` and resume the same
   worker once. The correction prompt says `Do not repeat the review`; do not
   launch a fresh model merely to remove prose, a fence, or another formatting
   defect. If the resumed result remains invalid, record it and stop that pass.

5. Codex-origin findings need no extra verification. Send every free-pool-only
   candidate to one Codex verification worker using that row's exact
   `verification_model`. Keep only findings it reproduces from the diff.

6. Before fingerprinting, compare every retained finding, including adopted
   foreground reports, against the refreshed review history. Remove findings
   that are semantically equivalent despite skill, severity, wording, or line
   drift. Keep one only when new diff evidence invalidates the prior disposition,
   and include that delta in its description. Then fingerprint each retained
   finding with `pi_review_routing.py finding-fingerprint`; remove fingerprints
   listed in `foreground_fingerprints` and duplicates from another deferred pass.

7. Re-fetch `headRefOid` immediately before delivery. On any head or PR-state
   drift, record `stale` and post nothing. For `mode: "no-comments"`, retain the
   late findings in the runtime result without GitHub writes. For `mode: "full"`,
   submit one `COMMENT` review through `agent/skills/_shared/post_review.py`.
   Its body must identify late Codex-verified follow-up findings and include the
   originating skill/model audit rows. Never approve or request changes from
   follow-up; each BLOCK and WARN remains an unresolved inline thread. Late NITs
   go under a `## Nits` body section, never inline — the same advisory contract
   the foreground uses.

8. Write exactly one strict JSON object atomically to `<manifest>.result.json`.
   The supervisor validates it with `FollowUpResult`, merges its ownership
   audit, captures the Pi exit code and bounded log tail, and atomically publishes
   the canonical result. Use this shape with no additional fields:

   ```json
   {
     "status": "complete",
     "attempts": [
       {
         "skill": "correctness-review",
         "pass_name": "free-pool",
         "model": "kimi-coding/k3",
         "status": "success",
         "agent_id": "<Tintin agent id or null>",
         "output_path": "<Tintin transcript path or null>",
         "detail": "<exact audit detail>"
       }
     ],
     "diagnostics": [],
     "late_findings": [
       {
         "severity": "warn",
         "path": "agent/example.py",
         "line": 42,
         "description": "<validated late finding>"
       }
     ],
     "posted_review_url": null,
     "child_exit_code": null,
     "log_tail": "",
     "completed_at": "2026-07-24T00:00:00Z"
   }
   ```

   Overall `status` is exactly `complete`, `stale`, or `failed`. Attempt status
   is exactly `success`, `failed`, `stale`, `verified`, `rejected`, or
   `malformed-report`; supervisor-only rows add `adopted-foreground-result`.
   Diagnostic category is exactly `capacity`,
   `child-exit`, `invalid-result`, `missing-result`, `ownership`, or
   `supervisor-error`. Set `child_exit_code` to `null` and `log_tail` to an empty
   string; the supervisor replaces both with observed process evidence.

Do not modify the foreground manifest, source checkout, or unrelated GitHub
metadata. The supervisor always persists child stdout and stderr in the bounded
`<foreground-manifest>.follow-up.log`; no follow-up output is discarded.
