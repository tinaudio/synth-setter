#!/usr/bin/env bash
# session-start-cwd-banner.sh — SessionStart hook (matchers: startup, resume,
# clear, compact). Prints a short banner to stdout so the agent sees its cwd
# and whether it's in the primary checkout before issuing the first tool call.
# Exits 0 outside a git repo (no banner; silent).
set -euo pipefail

main() {
  # Drain stdin to avoid SIGPIPE when the harness pipes event JSON.
  cat >/dev/null 2>&1 || true

  local git_dir common_dir
  git_dir=$(git rev-parse --git-dir 2>/dev/null) || exit 0
  common_dir=$(git rev-parse --git-common-dir 2>/dev/null) || exit 0

  local abs_git_dir abs_common_dir primary_root branch
  abs_git_dir=$(cd "$git_dir" 2>/dev/null && pwd) || exit 0
  abs_common_dir=$(cd "$common_dir" 2>/dev/null && pwd) || exit 0
  primary_root=$(dirname "$abs_common_dir")
  branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "<detached>")

  local in_primary=0
  [[ "$abs_git_dir" == "$abs_common_dir" ]] && in_primary=1

  local worktree_count
  worktree_count=$(git worktree list --porcelain 2>/dev/null | awk '/^worktree /' | wc -l | tr -d ' ')

  printf '[worktree status]\n'
  printf '  cwd      : %s\n' "$PWD"
  printf '  branch   : %s\n' "$branch"
  if (( in_primary == 1 )); then
    local slug=${branch//\//-}
    [[ -z "$slug" || "$slug" == "<detached>" ]] && slug="scratch"
    printf '  status   : PRIMARY CHECKOUT — read-only per AGENTS.md\n'
    # shellcheck disable=SC2016  # backticks are literal markdown-style hint to the agent, not command substitution
    printf '  worktrees: %s active (run `git worktree list` for paths)\n' "$worktree_count"
    printf '\n'
    printf 'Spawn a worktree before editing:\n'
    printf '  git worktree add .claude/worktrees/%s %s && cd .claude/worktrees/%s\n' \
      "$slug" "$branch" "$slug"
  else
    printf '  status   : isolated worktree (OK)\n'
    printf '  primary  : %s\n' "$primary_root"
    # shellcheck disable=SC2016  # backticks are literal markdown-style hint to the agent, not command substitution
    printf '  worktrees: %s active (run `git worktree list` for paths)\n' "$worktree_count"
  fi
}

main "$@"
