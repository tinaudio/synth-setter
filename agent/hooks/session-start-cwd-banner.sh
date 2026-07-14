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
    # `uv sync` builds the worktree's own .venv; `make link-plugins` backfills the
    # gitignored plugins/ symlink; `make link-thoughts` points thoughts/ at the
    # central primary copy; `make link-skills` projects marketplace skills.
    # Single-quote the paths so the command survives spaces.
    printf "  git worktree add --detach '%s/.claude/worktrees/%s' && cd '%s/.claude/worktrees/%s' && uv sync && make link-plugins && make link-thoughts && make link-skills\n" \
      "$primary_root" "$slug" "$primary_root" "$slug"
  else
    printf '  status   : isolated worktree (OK)\n'
    printf '  primary  : %s\n' "$primary_root"
    # shellcheck disable=SC2016  # backticks are literal markdown-style hint to the agent, not command substitution
    printf '  worktrees: %s active (run `git worktree list` for paths)\n' "$worktree_count"
  fi

  # Tool discovery paths are committed as symlinks into the canonical agent/
  # tree; unmaterialized or dangling links mean project assets disappear.
  local repo_top asset_path
  repo_top=$(git rev-parse --show-toplevel 2>/dev/null || true)
  [[ -n "$repo_top" ]] || return 0
  for asset_path in \
    "$repo_top/.agents/skills" \
    "$repo_top/.claude/hooks" \
    "$repo_top/.claude/skills"
  do
    [[ ( -e "$asset_path" || -L "$asset_path" ) && ! -d "$asset_path" ]] || continue
    printf '\n  WARNING: %s did not materialize as a directory — agent asset discovery is BROKEN.\n' "${asset_path#"$repo_top/"}"
    printf "    Fix: git -C '%s' config core.symlinks true && git -C '%s' checkout -- '%s'\n" "$repo_top" "$repo_top" "${asset_path#"$repo_top/"}"
  done
}

main "$@"
