#!/usr/bin/env bash
# PreToolUse branch-safety gate.
# Echoes the current branch name before any git commit.
set -euo pipefail

main() {
  # stdin is JSON payload containing tool input command
  local input cmd branch
  input=$(cat)
  cmd=$(jq -r '.tool_input.command // empty' <<<"$input" 2>/dev/null)

  # Only run for git commit commands
  if [[ "$cmd" =~ git[[:space:]]+commit ]]; then
    branch=$(git branch --show-current 2>/dev/null || true)
    if [[ -z "$branch" ]]; then
      branch="DETACHED HEAD"
    fi
    echo "Committing to branch: $branch" >&2
  fi
  exit 0
}

main "$@"
