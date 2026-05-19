#!/usr/bin/env bash
# doc-drift.sh — PostToolUse hook on `gh pr create`. Runs the doc-drift skill
# (or an inline fallback) headlessly in a detached worktree pinned to the
# captured HEAD and writes an advisory report to
# .agent-reviews/doc-drift-<uuid>.md in the main repo. Exits 2 with a metadata-
# stamped pointer so the active agent session reads the report; never blocks.
#
# Worktree isolation: prevents the headless agent's commands from touching the
# user's primary checkout if a prompt asks it to switch branches mid-task.
#
# Env overrides: DOC_DRIFT_DRY_RUN=1 skips the agent call;
# AGENT_HEADLESS=claude|codex pins the CLI; AGENT_TIMEOUT_SECS bounds the
# headless run (default 900s).
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

BRANCH=$(git branch --show-current 2>/dev/null || true)
HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || true)
PR=$(gh pr view "$BRANCH" --json number -q .number 2>/dev/null || true)

if [[ -z "$PR" ]]; then
  exit 0
fi

ensure_reviews_dir
sweep_stale_worktrees "doc-drift"
log "matched: ${COMMAND}"

BASE_BRANCH=$(default_branch)
log "base branch resolved to: ${BASE_BRANCH}"
# `head -50` closes its read end after 50 lines, sending SIGPIPE to `git diff`;
# `pipefail` would otherwise propagate git's non-zero exit. `|| true` keeps the
# substitution non-fatal under `set -e`.
DIFF_FILES=$(git diff "origin/${BASE_BRANCH}...HEAD" --name-only 2>/dev/null | head -50 || true)

WT=$(make_isolated_worktree "doc-drift" "$HEAD_SHA") || {
  log "worktree creation failed for ${HEAD_SHA}, exiting"
  exit 0
}
trap 'remove_worktree "$WT"' EXIT
cd "$WT"

if has_skill doc-drift; then
  log "using doc-drift skill"
  PROMPT="Use the doc-drift skill to review PR #${PR} on branch ${BRANCH} (HEAD ${HEAD_SHA:0:7}) against base ${BASE_BRANCH}. Cross-reference docs/doc-map.yaml if present. Changed files:
${DIFF_FILES}

Report findings as \"file:line — issue — suggested fix\". If no drift is found, state that explicitly."
else
  log "doc-drift skill not found, using fallback prompt"
  PROMPT="Review the diff 'git diff origin/${BASE_BRANCH}...HEAD' for documentation drift on PR #${PR}, branch ${BRANCH} (HEAD ${HEAD_SHA:0:7}). Cross-reference docs/doc-map.yaml if present. Check for: references to renamed/removed files, stale signatures, missing docs for new public APIs, out-of-date examples. Changed files:
${DIFF_FILES}

Report findings as \"file:line — issue — suggested fix\". If no drift is found, state that explicitly."
fi

META=$(printf 'PR: #%s\nBranch: %s\nBase: %s\nHEAD: %s\n' "$PR" "$BRANCH" "$BASE_BRANCH" "$HEAD_SHA")
REVIEW_FILE=$(run_review "doc-drift" "$META" "$PROMPT" "DOC_DRIFT_DRY_RUN")

log "wrote ${REVIEW_FILE}"

emit_rewake_stamp "doc-drift" "$PR" "$BRANCH" "$HEAD_SHA" "$REVIEW_FILE" \
  "Advisory — read it and apply documentation updates as appropriate."
exit 2
