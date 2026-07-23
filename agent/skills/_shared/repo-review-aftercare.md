# Deferred Pi review aftercare

Process only the deferred passes in the runtime manifest named by the launch
prompt. The Python supervisor, not this model, owns the canonical
`<foreground-manifest>.result.json`. It has already validated any adopted
foreground outputs, named those rows in the launch prompt, and proved every
other foreground owner stopped before launching this session. Never repeat a
pass not present in the runtime manifest.

1. Validate the runtime manifest before using it:

   ```bash
   ./.venv/bin/python agent/_shared/run_pi_review_aftercare.py <manifest> --dry-run
   ```

2. Fetch the PR's current `headRefOid` and state. If the PR is not open or its
   head differs from `head_sha`, record `stale` in the runtime result and exit
   without posting.

3. For each adopted row named in the launch prompt, extract and validate its
   existing `output_path`, use that report, and do not launch its pass again.
   For every other `deferred_passes` row, generate the assignment with
   `pi_review_routing.py worker-prompt`, launch one `pr-review-worker` using the
   row's exact pinned model and thinking, and validate its output with
   `extract-report` and `validate-report`. Exactly one model call owns a pass;
   never launch a second owner for the same row.

4. If strict validation fails after envelope extraction, generate the
   diagnostic with `pi_review_routing.py repair-prompt` and resume the same
   worker once. The correction prompt says `Do not repeat the review`; do not
   launch a fresh model merely to remove prose, a fence, or another formatting
   defect. If the resumed result remains invalid, record it and stop that pass.

5. Codex-origin findings need no extra verification. Send every free-pool-only
   candidate to one Codex verification worker using that row's exact
   `verification_model`. Keep only findings it reproduces from the diff.

6. Fingerprint each retained finding with
   `pi_review_routing.py finding-fingerprint`. Remove fingerprints listed in
   `foreground_fingerprints` and duplicates produced by another deferred pass.

7. Re-fetch `headRefOid` immediately before delivery. On any head or PR-state
   drift, record `stale` and post nothing. For `mode: "no-comments"`, retain the
   late findings in the runtime result without GitHub writes. For `mode: "full"`,
   submit one `COMMENT` review through `agent/skills/_shared/post_review.py`.
   Its body must identify late Codex-verified aftercare findings and include the
   originating skill/model audit rows. Never approve or request changes from
   aftercare; each finding remains an unresolved inline thread.

8. Write exactly one strict JSON object atomically to `<manifest>.result.json`.
   The supervisor validates it with `AftercareResult`, merges its ownership
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
`<foreground-manifest>.aftercare.log`; no aftercare output is discarded.
