#!/usr/bin/env bash
# =============================================================================
# _lib.sh — shared helpers for agent hook scripts
# =============================================================================
#
# Sourced by doc-drift.sh and pr-review-resolver.sh. Provides:
#   - has_skill <name>         : returns 0 if the named skill is installed
#   - log <msg>                : append a timestamped line to .agent-reviews/.hook.log
#   - ensure_reviews_dir       : create .agent-reviews/ if missing
#   - gen_id                   : portable unique ID for report filenames and lock tokens
#   - default_branch           : resolve the repo's default branch (origin/HEAD to main to master)
#   - run_agent_prompt <prompt>: run a prompt through the available headless agent CLI
#
# Skill detection searches project-local, Claude, and Codex skill locations.
# =============================================================================

REVIEWS_DIR=".agent-reviews"
HOOK_LOG="${REVIEWS_DIR}/.hook.log"

ensure_reviews_dir() {
  mkdir -p "$REVIEWS_DIR"
}

log() {
  # Usage: log "message"
  # Appends "<iso8601> [<hook_name>] <msg>" to .agent-reviews/.hook.log.
  # Silent on failure — log path may not exist early in a run.
  local ts
  ts=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
  printf '%s [%s] %s\n' "$ts" "${HOOK_NAME:-unknown}" "$*" >> "$HOOK_LOG" 2>/dev/null || true
}

gen_id() {
  # Portable unique ID for report filenames and lock tokens.
  if command -v uuidgen >/dev/null 2>&1; then
    uuidgen
  elif [[ -r /proc/sys/kernel/random/uuid ]]; then
    cat /proc/sys/kernel/random/uuid
  else
    printf '%s-%s' "$(date +%s)" "$$"
  fi
}

default_branch() {
  # Resolve the repo's default branch name. Order:
  #   1. origin/HEAD symbolic-ref (best: tracks the real remote default)
  #   2. local 'main' if it exists
  #   3. local 'master' if it exists
  #   4. 'main' (final fallback)
  local ref
  ref=$(git symbolic-ref --quiet refs/remotes/origin/HEAD 2>/dev/null || true)
  if [[ -n "$ref" ]]; then
    echo "${ref##*/}"
    return 0
  fi
  for b in main master; do
    git show-ref --verify --quiet "refs/heads/$b" 2>/dev/null && { echo "$b"; return 0; }
    git show-ref --verify --quiet "refs/remotes/origin/$b" 2>/dev/null && { echo "$b"; return 0; }
  done
  echo "main"
}

has_skill() {
  # Usage: has_skill <skill-name>
  # Returns 0 if <name>/SKILL.md exists in any known skill location.
  # Worktree-aware: also checks the main repo via the git common dir, since
  # agent tool directories may live only in the primary checkout.
  local name="$1"
  local home="${HOME:-}"
  local paths=(
    "agent/skills/${name}/SKILL.md"
    ".claude/skills/${name}/SKILL.md"
  )
  if [[ -n "$home" ]]; then
    paths+=(
      "${home}/.claude/skills/${name}/SKILL.md"
      "${home}/.codex/skills/${name}/SKILL.md"
    )
  fi
  local common_dir repo_root
  if common_dir=$(git rev-parse --git-common-dir 2>/dev/null) && [[ -n "$common_dir" ]]; then
    repo_root=$(cd "$common_dir/.." 2>/dev/null && pwd)
    if [[ -n "$repo_root" ]]; then
      paths+=(
        "${repo_root}/agent/skills/${name}/SKILL.md"
        "${repo_root}/.claude/skills/${name}/SKILL.md"
      )
    fi
  fi
  for p in "${paths[@]}"; do
    [[ -f "$p" ]] && return 0
  done
  if [[ -n "$home" ]]; then
    for p in "${home}"/.claude/plugins/*/skills/"${name}"/SKILL.md; do
      [[ -f "$p" ]] && return 0
    done
    for p in "${home}"/.codex/plugins/*/skills/"${name}"/SKILL.md; do
      [[ -f "$p" ]] && return 0
    done
  fi
  return 1
}

run_agent_prompt() {
  local prompt="$1"
  if [[ "${AGENT_HEADLESS:-}" == "codex" ]]; then
    codex exec "$prompt"
  elif [[ "${AGENT_HEADLESS:-}" == "claude" ]]; then
    claude -p "$prompt"
  elif command -v claude >/dev/null 2>&1; then
    claude -p "$prompt"
  elif command -v codex >/dev/null 2>&1; then
    codex exec "$prompt"
  else
    printf 'No supported headless agent CLI found; install claude or codex.\n' >&2
    return 127
  fi
}
