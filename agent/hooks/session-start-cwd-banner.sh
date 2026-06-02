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

  local abs_git_dir abs_common_dir primary_root branch_label short_sha
  abs_git_dir=$(cd "$git_dir" 2>/dev/null && pwd) || exit 0
  abs_common_dir=$(cd "$common_dir" 2>/dev/null && pwd) || exit 0
  primary_root=$(dirname "$abs_common_dir")
  # `git branch --show-current` is the only reliable detached-HEAD probe:
  # --abbrev-ref returns the literal "HEAD" instead of erroring out, so a
  # `|| <fallback>` chain on it never fires.
  branch_label=$(git branch --show-current 2>/dev/null || true)
  local slug
  if [[ -n "$branch_label" ]]; then
    slug=${branch_label//\//-}
  else
    short_sha=$(git rev-parse --short HEAD 2>/dev/null || true)
    branch_label="(detached HEAD${short_sha:+ ${short_sha}})"
    slug="detached-${short_sha:-scratch}"
  fi

  local in_primary=0
  [[ "$abs_git_dir" == "$abs_common_dir" ]] && in_primary=1

  local worktree_count
  worktree_count=$(git worktree list --porcelain 2>/dev/null | awk '/^worktree /' | wc -l | tr -d ' ')

  printf '[worktree status]\n'
  printf '  cwd      : %s\n' "$PWD"
  printf '  branch   : %s\n' "$branch_label"
  if (( in_primary == 1 )); then
    printf '  status   : PRIMARY CHECKOUT — read-only per AGENTS.md\n'
    # shellcheck disable=SC2016  # backticks are literal markdown-style hint to the agent, not command substitution
    printf '  worktrees: %s active (run `git worktree list` for paths)\n' "$worktree_count"
    printf '\n'
    # `--detach` always works regardless of where the branch is checked out,
    # matching the primary-edit guard's remediation and `_lib.sh` convention.
    printf 'Spawn a worktree before editing:\n'
    # Anchor to $primary_root so the command works even when the session started in a subdir.
    # `uv sync` builds the worktree's own .venv so it stops sharing the image's /venv/main.
    # Single-quote the emitted paths so the command survives a $primary_root with spaces.
    printf "  git worktree add --detach '%s/.claude/worktrees/%s' && cd '%s/.claude/worktrees/%s' && uv sync\n" \
      "$primary_root" "$slug" "$primary_root" "$slug"
  else
    printf '  status   : isolated worktree (OK)\n'
    printf '  primary  : %s\n' "$primary_root"
    # shellcheck disable=SC2016  # backticks are literal markdown-style hint to the agent, not command substitution
    printf '  worktrees: %s active (run `git worktree list` for paths)\n' "$worktree_count"
  fi
}

main "$@"
