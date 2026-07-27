# PR readiness loop

After every push to a PR branch, drive the readiness loop until the PR clears
the four gates below. **Do not stop after the first push.** "I pushed the fix"
is not the same as "the PR is ready." The summary line in the project's
`AGENTS.md` (under `## PRs`) points here for the full procedure.

## The four gates

A PR is **not** ready — for review, merge, or hand-off — until **all four**
hold. They are AND-ed; failing any one means not ready.

1. **CI is fully green** — every required AND optional check passing. Pending,
   errored, or failing all count as not ready.
2. **`mergeable=MERGEABLE`** — `gh pr view <N> --json mergeable -q .mergeable`
   reports `MERGEABLE`. `UNKNOWN` (GitHub still computing) and `CONFLICTING`
   both fail this gate; keep polling `UNKNOWN` until it resolves.
3. **Every open review comment has an inline reply** — every unresolved review
   thread (human reviewers, Copilot, and the Claude CI review) has either a
   code change linked by commit SHA or an inline reply with justification.
   Drive this with `/pr-review-resolver`.
4. **Copilot has produced no new comments since the last push** — Copilot
   re-reviews after every push, usually within ~60s. Both the inline-comments
   endpoint and the top-level reviews endpoint must be clear.

## The probe

`agent/_shared/pr_readiness_probe.sh <N>` reports all four gates in one shot —
one line per gate, with terminal check failures including their result and URL
and gate-3 failures listing each unresolved thread. Its final state is:

- `READY` (exit 0): gates 1–3 hold.
- `ACTION_REQUIRED` (exit 1): failed CI, a merge conflict, or a review thread
  needs agent work.
- `ERROR` (exit 2): the probe could not evaluate readiness.
- `WAIT` (exit 8): only transient work remains, such as pending CI or
  `mergeable=UNKNOWN`.

Action takes precedence over waiting when both exist. Merged/closed PRs
short-circuit to `READY`. `--gates-only` skips gate 4. `--loop` is exclusively
a polling adapter: it retries `WAIT` and stops on `READY`, `ACTION_REQUIRED`,
or `ERROR`, preserving the printed state. The Stop hook runs the normal probe,
so hook and loop classifications cannot disagree.

**Poll only `WAIT`.** Polling every non-ready result traps the agent on failed
checks and conflicts that cannot heal themselves. Polling `gh pr checks` alone
also misses review comments that land mid-wait.

## The loop

After every push, run the normal probe once. If and only if it reports `WAIT`,
use the action-aware adapter:

```bash
/loop 2m bash agent/_shared/pr_readiness_probe.sh --loop <N>
```

The loop stops when a terminal decision appears. Run the normal probe again,
then follow its state; never continue passive polling after `ACTION_REQUIRED`
or `ERROR`.

1. **Push the change.**

2. **Run the normal probe.** Poll with `--loop` only when it reports `WAIT`.

3. **If any check fails:** use the reported check name and URL to diagnose,
   fix, push, and return to step 2. Do not wait for terminal CI to change.

4. **Check mergeability** with `gh pr view <N> --json mergeable -q .mergeable`:

   - `CONFLICTING` → rebase or merge the base branch, resolve, push, back to 2.
   - `UNKNOWN` → GitHub hasn't finished computing; poll again.
   - `MERGEABLE` → continue.

5. **Reply inline to every open review comment.** The probe's gate-3 output
   lists every unresolved thread awaiting a reply; for full bodies use
   `gh api repos/<OWNER>/<REPO>/pulls/<N>/comments --paginate`. If a reply
   required a code change, push and return to step 2. Drive this with
   `/pr-review-resolver`. The reply endpoint that works is
   `repos/<OWNER>/<REPO>/pulls/<N>/comments/<id>/replies` — the shorter
   `pulls/comments/<id>/replies` returns 404.

6. **Wait for Copilot's post-push review** (~60s, allow up to 15 minutes).
   Check both endpoints:

   ```bash
   gh api repos/<OWNER>/<REPO>/pulls/<N>/comments --paginate \
     --jq '[.[] | select(.user.login | test("[Cc]opilot")) | {id, path, line, created_at, commit_id, body}]'
   gh api repos/<OWNER>/<REPO>/pulls/<N>/reviews --paginate \
     --jq '[.[] | select(.user.login | test("[Cc]opilot")) | {id, state, submitted_at, commit_id, body}]'
   ```

   `created_at` (inline) and `submitted_at` (review) let you filter to
   comments newer than your last push timestamp; `commit_id` lets you filter
   to comments anchored to the just-pushed SHA (`git rev-parse HEAD`). Use
   either to distinguish fresh findings from already-addressed ones.

   If Copilot left new unaddressed inline comments **or** a new top-level
   review with actionable content (`state=COMMENTED` / `CHANGES_REQUESTED`
   with a body that isn't just a "no findings" note), return to step 5 and
   address them the same way as human comments. If 15 minutes elapse with no
   Copilot activity at all, the auto-review didn't fire — go to 6a.

   **6a. Manually re-request Copilot** when the 15-minute window elapses with
   no activity. Try in this order, stopping at the first that succeeds:

   - Re-request via the reviewers API:

     ```bash
     gh api --method POST \
       /repos/<OWNER>/<REPO>/pulls/<N>/requested_reviewers \
       -f 'reviewers[]=copilot-pull-request-reviewer[bot]'
     ```

     If that errors, confirm the bot slug your org uses with
     `gh pr view <N> --json reviewRequests,reviews` and retry.

   - If the reviewers API still won't take Copilot, force a re-trigger with
     an empty commit (works when Copilot runs on push, not as a requested
     reviewer):

     ```bash
     git commit --allow-empty -m "chore: trigger copilot review"
     git push
     ```

     Pushing restarts the readiness loop — return to step 2.

   After re-requesting, wait another ~60s (allow up to 15 minutes) and re-check
   Copilot's comments. Repeat at most once; if Copilot still produces nothing
   after a manual re-request, record that in the PR thread and move on.

7. **Done** only when all four gates hold: CI green ∧ `mergeable=MERGEABLE`
   ∧ every review comment has an inline reply ∧ Copilot has produced no new
   comments since the last push (or has been confirmed silent via 6a).

This applies whether the PR is yours or one you were asked to drive across
the finish line.

## Traps

- **`gh pr review` HTTP 502 ≠ post failed.** The POST often succeeded
  server-side. Re-list reviews before re-posting or you'll create a duplicate
  review.
- **`mergeable=UNKNOWN` is `WAIT`, not a pass.** `CONFLICTING` is
  `ACTION_REQUIRED`; only `MERGEABLE` clears gate 2.
- **A "no findings" Copilot review counts as silence, not a comment.** Only
  `state=COMMENTED` / `CHANGES_REQUESTED` with substantive body content
  blocks gate 4.
