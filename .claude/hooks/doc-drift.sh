#!/usr/bin/env bash
# =============================================================================
# doc-drift.sh — PostToolUse hook, runs on `gh pr create`
# =============================================================================
#
# PURPOSE
# -------
# After Claude creates a PR, run the doc-drift skill (or an inline fallback
# if the skill is not installed) in a headless `claude -p` session to produce
# an advisory documentation-drift report. Written to .agent-reviews/doc-drift-<uuid>.md.
#
# Returns exit 2 with a pointer so the active Claude session reads the report
# and applies doc updates as appropriate. Advisory — does not block.
#
# INPUT (stdin)
#   JSON with .tool_input.command — the Bash command Claude just ran.
#
# INVOCATION
#   Registered in .claude/settings.json as a PostToolUse hook on Bash with
#   asyncRewake: true and timeout: 900 (15 min).
#
# ENV OVERRIDES
#   DOC_DRIFT_DRY_RUN=1        — skip the actual `claude -p` call; write a
#                                stub report. Used by the unit harness.
# =============================================================================
set -euo pipefail

export HOOK_NAME="doc-drift"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.claude/hooks/_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

case "$COMMAND" in
  *"gh pr create"*) ;;
  *) exit 0 ;;
esac

ensure_reviews_dir
log "matched: ${COMMAND}"

BRANCH=$(git branch --show-current 2>/dev/null || true)
PR=$(gh pr view --json number -q .number 2>/dev/null || true)

if [ -z "$PR" ]; then
  log "no PR found for branch ${BRANCH:-?}, skipping"
  exit 0
fi

DIFF_FILES=$(git diff main...HEAD --name-only 2>/dev/null | head -50 || true)

if has_skill doc-drift; then
  log "using doc-drift skill"
  PROMPT="Use the doc-drift skill to review PR #${PR} on branch ${BRANCH}. Cross-reference docs/doc-map.yaml if present. Changed files:
${DIFF_FILES}

Report findings as \"file:line — issue — suggested fix\". If no drift is found, state that explicitly."
else
  log "doc-drift skill not found, using fallback prompt"
  PROMPT="Review the diff 'git diff main...HEAD' for documentation drift on PR #${PR}, branch ${BRANCH}. Cross-reference docs/doc-map.yaml if present. Check for: references to renamed/removed files, stale signatures, missing docs for new public APIs, out-of-date examples. Changed files:
${DIFF_FILES}

Report findings as \"file:line — issue — suggested fix\". If no drift is found, state that explicitly."
fi

REVIEW_FILE="${REVIEWS_DIR}/doc-drift-$(gen_id).md"

if [ "${DOC_DRIFT_DRY_RUN:-0}" = "1" ]; then
  log "DRY_RUN: writing stub report"
  printf '# doc-drift (dry-run)\nPR: #%s\nBranch: %s\n\n## Prompt\n%s\n' \
    "$PR" "$BRANCH" "$PROMPT" > "$REVIEW_FILE"
else
  log "invoking claude -p (headless)"
  claude -p "$PROMPT" > "$REVIEW_FILE" 2>/dev/null || {
    log "claude -p failed"
    exit 0
  }
fi

log "wrote ${REVIEW_FILE}"

printf 'doc-drift report for PR #%s at %s. Advisory — read it and apply documentation updates as appropriate before continuing.\n' \
  "$PR" "$REVIEW_FILE" >&2
exit 2
