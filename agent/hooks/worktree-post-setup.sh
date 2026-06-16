#!/usr/bin/env bash
# PostToolUse hook: runs make link-plugins && make link-thoughts in every
# new worktree after `git worktree add`. Fail-safe: exits 0 on any error — see #1343.
set -euo pipefail

# shellcheck disable=SC2034  # read by log() in _lib.sh via ${HOOK_NAME:-unknown}
readonly HOOK_NAME="worktree-post-setup"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly SCRIPT_DIR

[[ -f "${SCRIPT_DIR}/_lib.sh" ]] && {
  # shellcheck source=agent/hooks/_lib.sh
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/_lib.sh"
}
# No-op fallback keeps fail-safe `|| { log ...; exit 0; }` branches safe
# when _lib.sh is absent — without it, `command -v log` on macOS resolves
# to /usr/bin/log and set -e would turn an undefined-function call fatal.
declare -F log >/dev/null 2>&1 || log() { :; }

# Parse worktree path from `git worktree add [flags] <path> [<commit-ish>]`.
# Prints the path and exits 0; exits 1 when no path is found.
_parse_worktree_path() {
  WT_CMD="$1" python3 - <<'PYEOF'
import os
import shlex
import sys

try:
    tokens = shlex.split(os.environ["WT_CMD"])
except Exception:
    sys.exit(1)

i = 0
while i < len(tokens) - 2:
    if tokens[i] == "git" and tokens[i + 1] == "worktree" and tokens[i + 2] == "add":
        i += 3
        break
    i += 1
else:
    sys.exit(1)

# --orphan takes <new-branch> in git 2.42+ (same positional role as -b).
flags_with_args = {"-b", "-B", "--orphan", "--reason"}
path = None
while i < len(tokens):
    tok = tokens[i]
    if tok == "--":
        # End-of-options: next token is the path even if it looks like a flag.
        if i + 1 < len(tokens):
            path = tokens[i + 1]
        break
    elif tok.startswith("-"):
        i += 2 if tok in flags_with_args else 1
    else:
        path = tok
        break

if not path:
    sys.exit(1)
print(path)
PYEOF
}

main() {
  local input cmd wt_path

  input=$(cat)

  # PostToolUse for Bash delivers tool_input.command in the JSON payload.
  cmd=$(jq -r '.tool_input.command // empty' <<< "$input" 2>/dev/null) || { log "could not parse tool_input.command; skipping"; exit 0; }

  # Boundary-aware check: matches invocations at start of command or after a
  # separator (&&, ;, |, backtick) — avoids false positives from echo/quoted strings.
  if ! echo "$cmd" | grep -qE '(^|[;|&`(][[:space:]]*)git[[:space:]]+worktree[[:space:]]+add([[:space:]]|$)'; then
    exit 0
  fi

  wt_path=$(_parse_worktree_path "$cmd" 2>/dev/null) || {
    log "could not parse worktree path from: $cmd"; exit 0
  }

  if [[ ! -d "$wt_path" ]]; then
    log "worktree path $wt_path does not exist; skipping"
    exit 0
  fi

  log "running make link-plugins && make link-thoughts && make link-skills in $wt_path"
  (
    cd "$wt_path"
    make link-plugins && make link-thoughts && make link-skills
  ) || {
    log "make link-plugins/link-thoughts/link-skills failed in $wt_path (non-fatal)"
  }
}

main "$@"
