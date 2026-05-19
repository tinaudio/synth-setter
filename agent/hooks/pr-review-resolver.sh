#!/usr/bin/env bash
# pr-review-resolver.sh — PostToolUse hook on `git push`. Waits ~6min for CI
# and reviewers to settle, then runs the pr-review-resolver skill (or an
# inline fallback) headlessly in a detached worktree pinned to the captured
# HEAD. Writes to .agent-reviews/pr-review-resolver-<uuid>.md in the main repo
# and exits 2 with a metadata-stamped pointer; never blocks.
#
# Worktree isolation: the headless agent's cwd is a temporary worktree under
# $WORKTREES_DIR. This prevents the agent from `git checkout`-ing over the
# user's main worktree if the prompt asks it to switch branches, and keeps the
# resolver's PR/branch resolution stable while the user is free to branch-hop.
#
# Bails (all silent early-exit, before sleep and before worktree creation):
# cmd does not contain "git push", current branch is main/master, no PR for
# the captured branch.
#
# Lockfile dedupe: per-branch token under $REVIEWS_DIR; after the sleep we
# re-read. If the token changed, a newer push is in flight and this run exits
# without creating its worktree.
#
# Env overrides: RESOLVER_SLEEP_SECS overrides the 360s wait (tests use 1s);
# RESOLVER_DRY_RUN=1 skips the agent call; AGENT_HEADLESS=claude|codex pins
# the CLI; AGENT_TIMEOUT_SECS bounds the headless run (default 900s).
set -euo pipefail

export HOOK_NAME="pr-review-resolver"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=agent/hooks/_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null || true)

if ! echo "$COMMAND" | grep -qE '(^|[;|&`(][[:space:]]*)git[[:space:]]+push([[:space:]]|$)'; then
  exit 0
fi

BRANCH=$(git branch --show-current 2>/dev/null || true)
case "$BRANCH" in
  main|master|"")
    exit 0
    ;;
esac

ensure_reviews_dir
sweep_stale_worktrees "resolver"
log "matched: ${COMMAND} (branch=${BRANCH})"

HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || true)
PR=$(gh pr view "$BRANCH" --json number -q .number 2>/dev/null || true)
if [[ -z "$PR" ]]; then
  log "no PR for branch ${BRANCH}, exiting"
  exit 0
fi

LOCKFILE="${REVIEWS_DIR}/.resolver-${BRANCH//\//_}.lock"
TOKEN=$(gen_id)
echo "$TOKEN" > "$LOCKFILE"
log "wrote lock token ${TOKEN}"

# Defensive: RESOLVER_SLEEP_SECS comes from the environment; an operator typo
# (`RESOLVER_SLEEP_SECS=abc`) would otherwise abort the hook at `sleep` under
# `set -e`. Fall back to the default if the value isn't a small positive
# integer (1-99999s; 0 would defeat the lockfile-dedupe purpose entirely).
SLEEP_SECS="${RESOLVER_SLEEP_SECS:-360}"
[[ "$SLEEP_SECS" =~ ^[1-9][0-9]{0,4}$ ]] || { log "invalid RESOLVER_SLEEP_SECS=${SLEEP_SECS}, using 360"; SLEEP_SECS=360; }
log "sleeping ${SLEEP_SECS}s"
sleep "$SLEEP_SECS"

CURRENT_TOKEN=$(cat "$LOCKFILE" 2>/dev/null || true)
if [[ "$CURRENT_TOKEN" != "$TOKEN" ]]; then
  log "superseded by newer push (lock now ${CURRENT_TOKEN}), exiting"
  exit 0
fi

WT=$(make_isolated_worktree "resolver" "$HEAD_SHA") || {
  log "worktree creation failed for ${HEAD_SHA}, exiting"
  exit 0
}
trap 'remove_worktree "$WT"' EXIT
cd "$WT"

if has_skill pr-review-resolver; then
  log "using pr-review-resolver skill"
  PROMPT="Use the pr-review-resolver skill for PR #${PR} on branch ${BRANCH} (HEAD ${HEAD_SHA:0:7})."
else
  log "pr-review-resolver skill not found, using fallback prompt"
  PROMPT="PR #${PR} on branch ${BRANCH} (HEAD ${HEAD_SHA:0:7}). Fetch review comments with 'gh pr view ${PR} --json reviews,comments' and 'gh api repos/{owner}/{repo}/pulls/${PR}/comments'. Address each actionable comment; reply inline to each comment you addressed with the fix commit SHA. Ignore nits unless trivial."
fi

META=$(printf 'PR: #%s\nBranch: %s\nHEAD: %s\n' "$PR" "$BRANCH" "$HEAD_SHA")
REVIEW_FILE=$(run_review "pr-review-resolver" "$META" "$PROMPT" "RESOLVER_DRY_RUN")

log "wrote ${REVIEW_FILE}"

printf 'pr-review-resolver report for PR #%s (branch %s, origin HEAD %s) at %s.\n' \
  "$PR" "$BRANCH" "${HEAD_SHA:0:7}" "$REVIEW_FILE" >&2
printf 'If your current HEAD is not %s, this advisory crossed sessions — verify before acting.\n' \
  "${HEAD_SHA:0:7}" >&2
exit 2
