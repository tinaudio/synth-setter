#!/usr/bin/env bash
# =============================================================================
# _lib.sh — shared helpers for .claude/hooks/ scripts
# =============================================================================
#
# Sourced by doc-drift.sh and pr-review-resolver.sh. Provides:
#   - has_skill <name>         : returns 0 if the named skill is installed
#   - log <msg>                : append a timestamped line to .agent-reviews/.hook.log
#   - ensure_reviews_dir       : create .agent-reviews/ if missing
#
# Skill detection searches, in order:
#   1. .claude/skills/<name>/SKILL.md          (project-local, plugin-managed)
#   2. ~/.claude/skills/<name>/SKILL.md        (user-global)
#   3. ~/.claude/plugins/*/skills/<name>/SKILL.md  (plugin-managed user-global)
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
  # Portable unique ID for report filenames.
  if command -v uuidgen >/dev/null 2>&1; then
    uuidgen
  elif [ -r /proc/sys/kernel/random/uuid ]; then
    cat /proc/sys/kernel/random/uuid
  else
    printf '%s-%s' "$(date +%s)" "$$"
  fi
}

has_skill() {
  # Usage: has_skill <skill-name>
  # Returns 0 if <name>/SKILL.md exists in any known skill location.
  local name="$1"
  local paths=(
    ".claude/skills/${name}/SKILL.md"
    "${HOME}/.claude/skills/${name}/SKILL.md"
  )
  for p in "${paths[@]}"; do
    [ -f "$p" ] && return 0
  done
  for p in "${HOME}"/.claude/plugins/*/skills/"${name}"/SKILL.md; do
    [ -f "$p" ] && return 0
  done
  return 1
}
