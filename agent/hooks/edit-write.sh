#!/usr/bin/env bash
# Shared Edit/Write hook handlers.
set -euo pipefail

MODE="${1:-}"
INPUT=$(cat)
# `format` and `test` are best-effort PostToolUse hooks — a missing jq, malformed
# stdin, or a formatter/test failure must not block the edit. Guard the parse and
# every external invocation with `|| true`. `credential-protect` stays strict.
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty' 2>/dev/null || true)

case "$MODE" in
  credential-protect)
    case "$FILE_PATH" in
      *.env|*.env.*|*.pem|*.key)
        echo 'BLOCKED: Cannot edit credential/secret files (.env, .pem, .key)' >&2
        exit 1
        ;;
    esac
    ;;
  format)
    case "$FILE_PATH" in
      *.py)
        ruff check --fix --quiet "$FILE_PATH" 2>/dev/null || true
        ruff format --quiet "$FILE_PATH" 2>/dev/null || true
        ;;
      *.md)
        pre-commit run mdformat --files "$FILE_PATH" >/dev/null 2>&1 || true
        ;;
      *.yaml|*.yml)
        pre-commit run prettier --files "$FILE_PATH" >/dev/null 2>&1 || true
        ;;
    esac
    exit 0
    ;;
  test)
    case "$FILE_PATH" in
      */src/*|*/scripts/*)
        base=$(basename "$FILE_PATH" .py)
        test_file="tests/test_${base}.py"
        if [[ -f "$test_file" ]]; then
          echo "Running $test_file" >&2
          pytest "$test_file" -x -q --no-header --tb=short 2>&1 | tail -5 >&2 || true
        fi
        ;;
      */tests/*)
        echo "Running $FILE_PATH" >&2
        pytest "$FILE_PATH" -x -q --no-header --tb=short 2>&1 | tail -5 >&2 || true
        ;;
    esac
    exit 0
    ;;
  *)
    echo "Unknown edit-write hook mode: ${MODE}" >&2
    exit 2
    ;;
esac
