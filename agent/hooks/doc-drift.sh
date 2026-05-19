#!/usr/bin/env bash
# doc-drift.sh — PostToolUse hook on `gh pr create`. Runs the doc-drift skill
# (or an inline fallback) headlessly and writes an advisory report to
# .agent-reviews/doc-drift-<uuid>.md. Exits 2 with a pointer so the active
# agent session reads the report; never blocks. Env overrides: DOC_DRIFT_DRY_RUN=1
# skips the agent call; AGENT_HEADLESS=claude|codex pins the CLI.
set -euo pipefail

export HOOK_NAME="doc-drift"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=agent/hooks/_lib.sh
source "${SCRIPT_DIR}/_lib.sh"

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null || true)

if ! echo "$COMMAND" | grep -qE '(^|[;|&`(][[:space:]]*)gh[[:space:]]+pr[[:space:]]+create([[:space:]]|$)'; then
  exit 0
fi

ensure_reviews_dir
log "matched: ${COMMAND}"

BRANCH=$(git branch --show-current 2>/dev/null || true)
PR=$(gh pr view --json number -q .number 2>/dev/null || true)

if [[ -z "$PR" ]]; then
  log "no PR found for branch ${BRANCH:-?}, skipping"
  exit 0
fi

BASE_BRANCH=$(default_branch)
log "base branch resolved to: ${BASE_BRANCH}"
# `head -50` closes its read end after 50 lines, sending SIGPIPE to `git diff`;
# `pipefail` would otherwise propagate git's non-zero exit. `|| true` keeps the
# substitution non-fatal under `set -e`.
DIFF_FILES=$(git diff "origin/${BASE_BRANCH}...HEAD" --name-only 2>/dev/null | head -50 || true)

if has_skill doc-drift; then
  log "using doc-drift skill"
  PROMPT="Use the doc-drift skill to review PR #${PR} on branch ${BRANCH} against base ${BASE_BRANCH}. Cross-reference docs/doc-map.yaml if present. Changed files:
${DIFF_FILES}

Report findings as \"file:line — issue — suggested fix\". If no drift is found, state that explicitly."
else
  log "doc-drift skill not found, using fallback prompt"
  PROMPT="Review the diff 'git diff origin/${BASE_BRANCH}...HEAD' for documentation drift on PR #${PR}, branch ${BRANCH}. Cross-reference docs/doc-map.yaml if present. Check for: references to renamed/removed files, stale signatures, missing docs for new public APIs, out-of-date examples. Changed files:
${DIFF_FILES}

Report findings as \"file:line — issue — suggested fix\". If no drift is found, state that explicitly."
fi

REVIEW_ID=$(gen_id)
REVIEW_FILE="${REVIEWS_DIR}/doc-drift-${REVIEW_ID}.md"
STDERR_FILE="${REVIEWS_DIR}/doc-drift-${REVIEW_ID}.stderr"
META=$(printf 'PR: #%s\nBranch: %s\nBase: %s\n' "$PR" "$BRANCH" "$BASE_BRANCH")

run_review "doc-drift" "$META" "$PROMPT" "$REVIEW_FILE" "$STDERR_FILE" "${DOC_DRIFT_DRY_RUN:-0}"

log "wrote ${REVIEW_FILE}"

printf 'doc-drift report for PR #%s at %s. Advisory — read it and apply documentation updates as appropriate before continuing.\n' \
  "$PR" "$REVIEW_FILE" >&2
exit 2
