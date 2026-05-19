#!/usr/bin/env bash
# no-baseline-additions.sh — PreToolUse gate keeping .pydoclint-baseline.txt
# append-frozen (AGENTS.md "Lint Exception Lists Are Closed", #938).
# Reads tool-call JSON on stdin; exits 0 (out of scope or row count <=) or 2.
set -euo pipefail

# shellcheck disable=SC2034  # read by log() in _lib.sh via ${HOOK_NAME:-unknown}
readonly HOOK_NAME="no-baseline-additions"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly SCRIPT_DIR

[[ -f "${SCRIPT_DIR}/_lib.sh" ]] || { echo "BLOCKED: missing _lib.sh" >&2; exit 2; }
# shellcheck source=agent/hooks/_lib.sh
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_lib.sh"

main() {
  # Any unexpected failure (Python crash, jq parse, etc.) must block — never
  # leak a non-2 exit that bypasses the contract documented in the header.
  trap 'log "internal failure on line $LINENO; blocking"; echo "BLOCKED: no-baseline-additions hit an internal error (line $LINENO); fix the hook or report it." >&2; exit 2' ERR

  local input file_path counts old_count new_count
  input=$(cat)
  # Fail closed: a silent fail-open is the bug class this gate exists to prevent.
  if ! file_path=$(jq -r '.tool_input.file_path // empty' <<<"$input" 2>/dev/null); then
    log "jq parse failed; blocking conservatively"
    echo "BLOCKED: no-baseline-additions could not parse tool-call JSON." >&2
    exit 2
  fi

  case "$file_path" in
    .pydoclint-baseline.txt|*/.pydoclint-baseline.txt) ;;
    *) exit 0 ;;
  esac

  # Missing baseline → OLD_COUNT=0 so delete+Write recreate cannot bypass the gate.
  counts=$(HOOK_INPUT="$input" SRC_PATH="$file_path" python3 - <<'PY'
import json, os, pathlib, sys

payload = json.loads(os.environ["HOOK_INPUT"])
tool_name = payload.get("tool_name", "")
tool = payload.get("tool_input", {})
src = pathlib.Path(os.environ["SRC_PATH"])
current = src.read_text() if src.exists() else ""

if tool_name == "Write":
    proposed = tool.get("content", "")
else:
    old = tool.get("old_string", "")
    new = tool.get("new_string", "")
    if old and old in current:
        proposed = current.replace(old, new, 1)
    else:
        print(
            "note: Edit old_string not found in current content; row count unchanged",
            file=sys.stderr,
        )
        proposed = current


def count_rows(text):
    """Count logical rows independent of trailing-newline convention."""
    return len(text.splitlines())


print(count_rows(current))
print(count_rows(proposed))
PY
)

  readarray -t _counts <<<"$counts"
  old_count="${_counts[0]}"
  new_count="${_counts[1]}"

  # Arithmetic context is safe here: both operands are integer row counts emitted
  # by the Python block, and set -e tolerates the `(( ))` exit because BLOCK is
  # the intended behaviour when the comparison is true.
  if (( new_count > old_count )); then
    log "blocking baseline addition: ${old_count} -> ${new_count}"
    cat >&2 <<EOF
BLOCKED: .pydoclint-baseline.txt is append-frozen (#938).
  Current rows: ${old_count}
  Proposed rows: ${new_count}

The remediation for a new pydoclint violation is to fix the underlying
docstring, not to register the file as exempt. See AGENTS.md for the
escalation path (rare, maintainer pre-approval on #938 required).
EOF
    exit 2
  fi

  exit 0
}

main "$@"
