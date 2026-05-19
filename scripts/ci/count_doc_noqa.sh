#!/usr/bin/env bash
# Count inline `# noqa: DOC*` suppressions under src/ and tests/.
# Drives the gate metric for the inline-DOC-noqa cleanup. Refs #1106.
set -euo pipefail

count_in() {
  local path="$1"
  { git grep -c 'noqa: DOC' -- "$path" 2>/dev/null || true; } \
    | awk -F: '{s+=$NF} END{print s+0}'
}

SRC_COUNT=$(count_in src/)
TESTS_COUNT=$(count_in tests/)
echo "src=${SRC_COUNT} tests=${TESTS_COUNT}"

if [[ "${1:-}" == "--gate" ]] && (( SRC_COUNT > 0 || TESTS_COUNT > 0 )); then
  exit 1
fi
