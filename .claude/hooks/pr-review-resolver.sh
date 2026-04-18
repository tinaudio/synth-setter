#!/usr/bin/env bash
# =============================================================================
# pr-review-resolver.sh — PostToolUse hook, runs on `git push`
# =============================================================================
#
# PURPOSE
# -------
# After Claude pushes a feature branch, wait for reviewers/CI to settle
# (default 360s), then run the pr-review-resolver skill (or an inline
# fallback) in a headless `claude -p` session to triage and respond to
# review comments. Report written to .agent-reviews/pr-review-resolver-<uuid>.md.
#
# Returns exit 2 with a pointer so the active Claude session reads the report
# and acts on unresolved items. Advisory — does not block.
#
# GATES (all early-exit silently, no 6-min wait wasted)
#   - cmd does not contain "git push"
#   - current branch is main/master
#   - after the wait: a newer push has overwritten our lock (dedupe)
#   - after the wait: no PR exists for this branch
#
# LOCKFILE DEDUPE
#   On entry we write a per-branch token to .agent-reviews/.resolver-<branch>.lock.
#   After the sleep we re-read; if the token changed a newer push is in
#   flight and this run exits silently. Last push wins.
#
# ENV OVERRIDES
#   RESOLVER_SLEEP_SECS       — override the 360s settle-wait (used by tests)
#   RESOLVER_DRY_RUN=1        — skip `claude -p`; write a stub report
#
# INVOCATION
#   Registered in .claude/settings.json as a PostToolUse hook on Bash with
#   asyncRewake: true and timeout: 1200 (20 min: ~6 min wait + resolver runtime).
# =============================================================================
set -euo pipefail

export HOOK_NAME="pr-review-resolver"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.claude/hooks/_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null || true)

case "$COMMAND" in
  *"git push"*) ;;
  *) exit 0 ;;
esac

BRANCH=$(git branch --show-current 2>/dev/null || true)
case "$BRANCH" in
  main|master|"")
    exit 0
    ;;
esac

ensure_reviews_dir
log "matched: ${COMMAND} (branch=${BRANCH})"

LOCKFILE="${REVIEWS_DIR}/.resolver-${BRANCH//\//_}.lock"
TOKEN="$$-$(date +%s%N)"
echo "$TOKEN" > "$LOCKFILE"
log "wrote lock token ${TOKEN}"

SLEEP_SECS="${RESOLVER_SLEEP_SECS:-360}"
log "sleeping ${SLEEP_SECS}s"
sleep "$SLEEP_SECS"

CURRENT_TOKEN=$(cat "$LOCKFILE" 2>/dev/null || true)
if [ "$CURRENT_TOKEN" != "$TOKEN" ]; then
  log "superseded by newer push (lock now ${CURRENT_TOKEN}), exiting"
  exit 0
fi

PR=$(gh pr view --json number -q .number 2>/dev/null || true)
if [ -z "$PR" ]; then
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

REVIEW_FILE="${REVIEWS_DIR}/pr-review-resolver-$(gen_id).md"

if [ "${RESOLVER_DRY_RUN:-0}" = "1" ]; then
  log "DRY_RUN: writing stub report"
  printf '# pr-review-resolver (dry-run)\nPR: #%s\nBranch: %s\n\n## Prompt\n%s\n' \
    "$PR" "$BRANCH" "$PROMPT" > "$REVIEW_FILE"
else
  log "invoking claude -p (headless)"
  claude -p "$PROMPT" > "$REVIEW_FILE" 2>/dev/null || {
    log "claude -p failed"
    exit 0
  }
fi

log "wrote ${REVIEW_FILE}"

printf 'pr-review-resolver report for PR #%s at %s. Advisory — read it and act on unresolved items.\n' \
  "$PR" "$REVIEW_FILE" >&2
exit 2
