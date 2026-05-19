#!/usr/bin/env bash
# no-baseline-additions.sh — PreToolUse Edit/Write gate keeping
# .pydoclint-baseline.txt append-frozen (AGENTS.md "Lint Exception Lists Are
# Closed", tracked in #938). Other append-frozen lists deferred to a follow-up.
# Reads tool-call JSON on stdin; exits 0 (out of scope or row count <= current)
# or 2 (proposed row count exceeds current).
set -euo pipefail

export HOOK_NAME="no-baseline-additions"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=agent/hooks/_lib.sh
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_lib.sh"

INPUT=$(cat)
# Fail closed: a silent fail-open is the bug class this gate exists to prevent.
if ! FILE_PATH=$(jq -r '.tool_input.file_path // empty' <<<"$INPUT" 2>/dev/null); then
  log "jq parse failed; blocking conservatively"
  echo "BLOCKED: no-baseline-additions could not parse tool-call JSON." >&2
  exit 2
fi

case "$FILE_PATH" in
  .pydoclint-baseline.txt|*/.pydoclint-baseline.txt) ;;
  *) exit 0 ;;
esac

if [[ ! -f "$FILE_PATH" ]]; then
  exit 0
fi

COUNTS=$(HOOK_INPUT="$INPUT" SRC_PATH="$FILE_PATH" python3 - <<'PY'
import json, os, pathlib

payload = json.loads(os.environ["HOOK_INPUT"])
tool_name = payload.get("tool_name", "")
tool = payload.get("tool_input", {})
src = pathlib.Path(os.environ["SRC_PATH"])
current = src.read_text()

if tool_name == "Write":
    proposed = tool.get("content", "")
else:
    old = tool.get("old_string", "")
    new = tool.get("new_string", "")
    proposed = current.replace(old, new, 1) if old and old in current else current


def count_rows(text):
    """Count logical rows independent of trailing-newline convention."""
    return len(text.splitlines())


print(count_rows(current))
print(count_rows(proposed))
PY
)

OLD_COUNT=$(printf '%s\n' "$COUNTS" | sed -n '1p')
NEW_COUNT=$(printf '%s\n' "$COUNTS" | sed -n '2p')

if (( NEW_COUNT > OLD_COUNT )); then
  log "blocking baseline addition: ${OLD_COUNT} -> ${NEW_COUNT}"
  cat >&2 <<EOF
BLOCKED: .pydoclint-baseline.txt is append-frozen (#938).
  Current rows: ${OLD_COUNT}
  Proposed rows: ${NEW_COUNT}

The remediation for a new pydoclint violation is to fix the underlying
docstring, not to register the file as exempt. See AGENTS.md for the
escalation path (rare, maintainer pre-approval on #938 required).
EOF
  exit 2
fi

exit 0
