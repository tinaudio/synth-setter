#!/usr/bin/env bash
# no-yaml-run-comments.sh — PreToolUse Edit/Write gate blocking `#`-comments
# inside a `run: |` / `setup: |` block scalar in .github/workflows/*.yml or
# configs/compute/*.yaml. The block body is bash once it reaches the runner,
# so a stray `'`/`` ` ``/`$`/`\` inside a comment can trigger shell expansion;
# put the comment ABOVE the step. Reads tool-call JSON on stdin; exits 0
# (out of scope / clean) or 2 (offending comment line found).
set -euo pipefail

export HOOK_NAME="no-yaml-run-comments"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=agent/hooks/_lib.sh
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_lib.sh"

INPUT=$(cat)
# Fail closed on jq parse error.
if ! FILE_PATH=$(jq -r '.tool_input.file_path // empty' <<<"$INPUT" 2>/dev/null); then
  log "jq parse failed; blocking conservatively"
  echo "BLOCKED: no-yaml-run-comments could not parse tool-call JSON." >&2
  exit 2
fi

case "$FILE_PATH" in
  .github/workflows/*.yml|.github/workflows/*.yaml) ;;
  */.github/workflows/*.yml|*/.github/workflows/*.yaml) ;;
  configs/compute/*.yaml|configs/compute/*.yml) ;;
  */configs/compute/*.yaml|*/configs/compute/*.yml) ;;
  *) exit 0 ;;
esac

TOOL_NAME=$(jq -r '.tool_name // empty' <<<"$INPUT" 2>/dev/null || echo "")

# Synthesize the post-edit content in a tmpfile for the Python scanner.
# Edit applies old_string -> new_string against the current file; Write uses
# the new content verbatim.
TMP_OUT=$(mktemp)
trap 'rm -f "$TMP_OUT"' EXIT

if [[ "$TOOL_NAME" == "Write" ]]; then
  jq -r '.tool_input.content // ""' <<<"$INPUT" > "$TMP_OUT"
else
  if [[ ! -f "$FILE_PATH" ]]; then
    exit 0
  fi
  HOOK_INPUT="$INPUT" SRC_PATH="$FILE_PATH" OUT_PATH="$TMP_OUT" \
    python3 - <<'PY'
import json, os, pathlib
payload = json.loads(os.environ["HOOK_INPUT"])
tool = payload.get("tool_input", {})
old = tool.get("old_string", "")
new = tool.get("new_string", "")
content = pathlib.Path(os.environ["SRC_PATH"]).read_text()
if old and old in content:
    content = content.replace(old, new, 1)
pathlib.Path(os.environ["OUT_PATH"]).write_text(content)
PY
fi

VIOLATIONS=$(SCAN_PATH="$TMP_OUT" python3 - <<'PY'
import os, re, pathlib

lines = pathlib.Path(os.environ["SCAN_PATH"]).read_text().splitlines()
# GitHub Actions and SkyPilot both accept `|`, `|-`, `|+`, `>`, `>-`, `>+`
# (literal and folded scalars with optional chomping indicators). An inline
# trailing comment on the header line is also valid YAML. The optional
# `(?:-\s+)?` allows the list-marker form `- run: |` where the key is on the
# same line as the list dash.
block_re = re.compile(r'^(\s*)(?:-\s+)?(run|setup)\s*:\s*[|>][-+]?\s*(?:#.*)?$')
violations = []
i = 0
while i < len(lines):
    m = block_re.match(lines[i])
    if not m:
        i += 1
        continue
    block_key = m.group(2)
    # When the line has a list-marker (`- run: |`), the body block scalar
    # must indent deeper than the `run:` column, not deeper than the dash.
    # `line.index(block_key)` resolves the actual column of `run`/`setup`.
    header_indent = lines[i].index(block_key)
    header_lineno = i + 1
    j = i + 1
    body_indent = None
    while j < len(lines):
        line = lines[j]
        if line.rstrip() == "":
            j += 1
            continue
        leading = len(line) - len(line.lstrip(" "))
        if body_indent is None:
            if leading <= header_indent:
                break
            body_indent = leading
        if leading <= header_indent:
            break
        bare = line.lstrip(" ")
        if bare.startswith("#") and not bare.startswith("#!"):
            violations.append((j + 1, block_key, header_lineno, bare.rstrip()))
        j += 1
    i = j

for lineno, key, header_lineno, text in violations:
    print(f"{lineno}\t{key}\t{header_lineno}\t{text}")
PY
)

if [[ -n "$VIOLATIONS" ]]; then
  log "blocking yaml-run-comment in $FILE_PATH"
  {
    echo "BLOCKED: comments inside a YAML \`run: |\` / \`setup: |\` block scalar."
    echo "File: ${FILE_PATH}"
    echo
    echo "Block-scalar bodies are bash. Stray ', \`, \$, or \\ inside a comment"
    echo "has caused unintended shell expansion. Move the comment ABOVE the step:"
    echo
    echo "  # Pin the template's image_id from its default to the dispatched tag."
    echo "  - name: Pin image tag"
    echo "    run: |"
    echo "      sed -i \"s|...|...|\" configs/compute/runpod-template.yaml"
    echo
    echo "Offenders (tab-delimited: line<TAB>block<TAB>header_line<TAB>text):"
    printf '  %s\n' "${VIOLATIONS//$'\n'/$'\n  '}"
  } >&2
  exit 2
fi

exit 0
