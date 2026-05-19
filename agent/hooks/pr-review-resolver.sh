#!/usr/bin/env bash
# pr-review-resolver.sh — PostToolUse hook on `git push`. Waits ~6min for CI
# and reviewers to settle, then runs the pr-review-resolver skill (or an
# inline fallback) headlessly. Writes to .agent-reviews/pr-review-resolver-<uuid>.md
# and exits 2 with a pointer; never blocks.
#
# Gates (all silent early-exit, no wait wasted): cmd does not contain "git push",
# current branch is main/master, a newer push overwrote our lock, no PR for the branch.
# Lockfile dedupe: write a per-branch token under .agent-reviews; after the sleep,
# re-read — if the token changed, a newer push is in flight and this run exits.
#
# Env overrides: RESOLVER_SLEEP_SECS overrides the 360s wait (tests use 1s);
# RESOLVER_DRY_RUN=1 skips the agent call; AGENT_HEADLESS=claude|codex pins the CLI.
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
log "matched: ${COMMAND} (branch=${BRANCH})"

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

PR=$(gh pr view --json number -q .number 2>/dev/null || true)
if [[ -z "$PR" ]]; then
  log "no PR for branch ${BRANCH}, exiting"
  exit 0
fi

if has_skill pr-review-resolver; then
  log "using pr-review-resolver skill"
  PROMPT="Use the pr-review-resolver skill for PR #${PR} on branch ${BRANCH}."
else
  log "pr-review-resolver skill not found, using fallback prompt"
  PROMPT="PR #${PR} on branch ${BRANCH}. Fetch review comments with 'gh pr view ${PR} --json reviews,comments' and 'gh api repos/{owner}/{repo}/pulls/${PR}/comments'. Address each actionable comment; reply inline to each comment you addressed with the fix commit SHA. Ignore nits unless trivial."
fi

REVIEW_ID=$(gen_id)
REVIEW_FILE="${REVIEWS_DIR}/pr-review-resolver-${REVIEW_ID}.md"
STDERR_FILE="${REVIEWS_DIR}/pr-review-resolver-${REVIEW_ID}.stderr"
META=$(printf 'PR: #%s\nBranch: %s\n' "$PR" "$BRANCH")

run_review "pr-review-resolver" "$META" "$PROMPT" "$REVIEW_FILE" "$STDERR_FILE" "${RESOLVER_DRY_RUN:-0}"

log "wrote ${REVIEW_FILE}"

printf 'pr-review-resolver report for PR #%s at %s. Advisory — read it and act on unresolved items.\n' \
  "$PR" "$REVIEW_FILE" >&2
exit 2
