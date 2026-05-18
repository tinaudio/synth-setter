#!/usr/bin/env bash
# =============================================================================
# pre-pr-review-gate.sh — PreToolUse hook, gates `gh pr create`
# =============================================================================
#
# PURPOSE
# -------
# Block `gh pr create` unless the command contains the literal token
# REVIEW_FULL_DONE=1 (anywhere on the line, typically as a trailing comment).
# Forces Claude to invoke /repo-review-full-no-comments and iterate on findings
# before opening a PR. The token is honor-system — the gate is a circuit-breaker
# that forces a pause, not unbypassable security.
#
# INPUT (stdin)
#   JSON with .tool_input.command — the Bash command Claude is about to run.
#
# INVOCATION
#   Registered in .claude/settings.json as a PreToolUse hook on Bash.
#   The handler-level `if: "Bash(gh pr create *)"` guard is unreliable for
#   PreToolUse hooks (see CLAUDE.md history), so this script re-validates the
#   command itself — same defensive pattern doc-drift.sh / pr-review-resolver.sh
#   use. Non-matching commands exit 0 silently.
# =============================================================================
set -euo pipefail

export HOOK_NAME="pre-pr-review-gate"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.claude/hooks/_lib.sh
# shellcheck disable=SC1091  # _lib.sh resolved at runtime; not staged together
source "${SCRIPT_DIR}/_lib.sh"

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null || true)

if ! echo "$COMMAND" | grep -qE '(^|[;|&`(][[:space:]]*)gh[[:space:]]+pr[[:space:]]+create([[:space:]]|$)'; then
  exit 0
fi

if echo "$COMMAND" | grep -q 'REVIEW_FULL_DONE=1'; then
  log "token present, allowing"
  exit 0
fi

log "blocking gh pr create (token missing)"
cat >&2 <<'EOF'
BLOCKED: gh pr create requires running /repo-review-full-no-comments first.
Run that skill, address every BLOCK/WARN finding (or document why each is
intentional), then re-run with REVIEW_FULL_DONE=1 included in the command —
recommended as a trailing comment so other hooks still fire:

  gh pr create --title "..." --body "..."  # REVIEW_FULL_DONE=1
EOF
exit 2
