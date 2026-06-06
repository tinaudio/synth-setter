#!/usr/bin/env bash
# worktree-post-setup.sh — PostToolUse hook that auto-runs
# `make link-plugins && make link-thoughts` whenever the agent calls
# `git worktree add`. Parses the worktree path from the captured command,
# then runs both targets from that directory so the new worktree is
# immediately usable without a manual follow-up.
#
# Fail-safe: any parse/execution failure logs and exits 0 so the hook
# never blocks after a completed worktree-add.
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

# Parse the worktree path from a `git worktree add [flags] <path> [<commit-ish>]`
# command string. Flags consuming an extra token: -b, -B, --orphan, --reason.
# Prints the path and returns 0, or returns 1 when no path is found.
_parse_worktree_path() {
  WT_CMD="$1" python3 - <<'PYEOF'
import os, sys, shlex

try:
    tokens = shlex.split(os.environ["WT_CMD"])
except Exception:
    sys.exit(1)

i = 0
while i < len(tokens) and tokens[i] in ("git", "worktree", "add"):
    i += 1

flags_with_args = {"-b", "-B", "--orphan", "--reason"}
path = None
while i < len(tokens):
    tok = tokens[i]
    if tok.startswith("-"):
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
  cmd=$(python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
print(d.get('tool_input', {}).get('command', ''))
" <<< "$input" 2>/dev/null) || { log "could not parse tool_input.command; skipping"; exit 0; }

  # The settings.json `if` matcher already scopes this to `git worktree add *`,
  # but re-validate for safety.
  case "$cmd" in
    *"git worktree add"*) ;;
    *) exit 0 ;;
  esac

  wt_path=$(_parse_worktree_path "$cmd" 2>/dev/null) || {
    log "could not parse worktree path from: $cmd"; exit 0
  }

  if [[ ! -d "$wt_path" ]]; then
    log "worktree path $wt_path does not exist; skipping"
    exit 0
  fi

  log "running make link-plugins && make link-thoughts in $wt_path"
  (
    cd "$wt_path"
    make link-plugins && make link-thoughts
  ) || {
    log "make link-plugins/link-thoughts failed in $wt_path (non-fatal)"
  }
}

main "$@"
