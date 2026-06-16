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
        # Guard mid-edit imports: an import often lands one tool call before its
        # use, so the save-time fix must not delete it. CLI/pre-commit strip dead ones.
        ruff check --fix --unfixable F401 --quiet "$FILE_PATH" 2>/dev/null || true
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
    # Match `src/`/`scripts/`/`tests/` at either the start of a relative path
    # or after a `/` in an absolute path. The `*/X/*` form alone was a latent
    # bug: it only matched absolute paths like `/foo/bar/src/baz.py`, never
    # a relative `src/foo/bar.py` from the agent's typical tool_input.file_path.
    case "$FILE_PATH" in
      src/*|scripts/*|*/src/*|*/scripts/*)
        # Try the package-mirrored layout first (tests/<package>/test_<base>.py),
        # then fall back to flat tests/test_<base>.py. Project uses the mirrored
        # form (e.g. src/synth_setter/pipeline/data/stats.py →
        # tests/pipeline/data/test_stats.py); the flat form is the legacy default.
        test_file=""
        if [[ "$FILE_PATH" =~ ^src/[^/]+/(.+)/([^/]+)\.py$ ]]; then
          mirrored="tests/${BASH_REMATCH[1]}/test_${BASH_REMATCH[2]}.py"
          [[ -f "$mirrored" ]] && test_file="$mirrored"
        fi
        if [[ -z "$test_file" ]]; then
          base=$(basename "$FILE_PATH" .py)
          flat="tests/test_${base}.py"
          [[ -f "$flat" ]] && test_file="$flat"
        fi
        if [[ -n "$test_file" ]]; then
          echo "Running $test_file" >&2
          pytest "$test_file" -x -q --no-header --tb=short 2>&1 | tail -5 >&2 || true
        fi
        ;;
      tests/*|*/tests/*)
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
