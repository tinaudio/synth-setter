# Deferred Pi review aftercare

Process only the deferred passes in the manifest named by the launch prompt. This
session is an asynchronous supplement to a foreground review that already met
its one-valid-report-per-skill quality floor. It may post only genuinely new,
Codex-verified findings against the exact reviewed PR head.

1. Validate the manifest before using it:

   ```bash
   ./.venv/bin/python agent/_shared/run_pi_review_aftercare.py <manifest> --dry-run
   ```

2. Fetch the PR's current `headRefOid` and state. If the PR is not open or its
   head differs from `head_sha`, record `stale` beside the manifest and exit
   without posting.

3. For each `deferred_passes` row, generate the assignment with
   `pi_review_routing.py worker-prompt`, launch one `pr-review-worker` using the
   row's exact pinned model and thinking, and validate its output with
   `extract-report` and `validate-report`. Never repeat a pass already completed
   in the foreground.

4. If strict validation fails after envelope extraction, generate the diagnostic
   with `pi_review_routing.py repair-prompt` and resume the same worker once.
   The correction prompt says `Do not repeat the review`; do not launch a fresh
   model merely to remove prose, a fence, or another formatting defect. If the
   resumed result remains invalid, record it and stop that pass.

5. Codex-origin findings need no extra verification. Send every free-pool-only
   candidate to one Codex verification worker using that row's exact
   `verification_model`, which records the effective foreground Codex model for
   the skill. Keep only findings it reproduces from the diff.

6. Fingerprint each retained finding with
   `pi_review_routing.py finding-fingerprint`. Remove fingerprints listed in
   `foreground_fingerprints` and duplicates produced by another deferred pass.
   If no findings remain, record `complete` and exit without GitHub writes.

7. Re-fetch `headRefOid` immediately before delivery. On any head or PR-state
   drift, record `stale` and post nothing. Otherwise submit one `COMMENT` review
   through `agent/skills/_shared/post_review.py`. Its body must say that these
   are late Codex-verified findings from deferred review aftercare and include
   the originating skill/model audit rows. Never approve or request changes
   from aftercare; each finding remains an unresolved inline thread.

8. Write the final state (`complete`, `stale`, or `failed`), audit rows, posted
   review URL, and completion timestamp atomically to `<manifest>.result.json`.
   Do not modify the manifest, source checkout, or any other GitHub metadata.
