"""Scanner for YAML block-scalar `#`-comment violations.

Companion to `no-yaml-run-comments.sh`. Reads the full PreToolUse JSON payload
on stdin and emits one tab-delimited violation per line on stdout:
``<line>\\t<block_key>\\t<header_lineno>\\t<text>``. The shell wrapper formats
the BLOCKED message; this module only locates offenders so the Edit/Write tool
call is gated before it lands content where a stray ``'``/`` ` ``/``$``/``\\``
inside a block-scalar comment would trigger shell expansion on the runner.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

# Matches a YAML block-scalar header such as ``run: |``, ``- setup: >-``,
# ``run: |+  # comment``. The optional ``(?:-\s+)?`` is load-bearing: it allows
# the list-marker form (``- run: |``) where the dash sits to the left of the
# key — chomping/folding indicators and trailing comments are absorbed by the
# regex itself, so the rest of the scanner only handles indentation.
_BLOCK_HEADER_RE = re.compile(r"^(\s*)(?:-\s+)?(run|setup)\s*:\s*[|>][-+]?\s*(?:#.*)?$")


def _scan(lines: list[str]) -> list[tuple[int, str, int, str]]:
    """Return one entry per `#`-comment found inside a YAML block-scalar body.

    :param lines: File contents already split on newlines (no trailing ``\\n``).
    :returns: Tuples of ``(line_number, block_key, header_line_number, text)`` in document order;
        line numbers are 1-indexed.
    """
    violations: list[tuple[int, str, int, str]] = []
    i = 0
    while i < len(lines):
        match = _BLOCK_HEADER_RE.match(lines[i])
        if not match:
            i += 1
            continue
        block_key = match.group(2)
        # Anchor at match.start(2) so a stray ``run``/``setup`` substring
        # inside an inline trailing comment can't be mis-resolved as the key.
        header_indent = lines[i].index(block_key, match.start(2))
        header_lineno = i + 1
        j = i + 1
        while j < len(lines):
            line = lines[j]
            if line.rstrip() == "":
                j += 1
                continue
            leading = len(line) - len(line.lstrip(" "))
            if leading <= header_indent:
                break
            bare = line.lstrip(" ")
            if bare.startswith("#") and not bare.startswith("#!"):
                violations.append((j + 1, block_key, header_lineno, bare.rstrip()))
            j += 1
        i = j
    return violations


def _post_edit_content(payload: dict) -> str | None:
    """Synthesise the content the tool call would commit.

    :param payload: Parsed PreToolUse JSON envelope.
    :returns: Post-edit text, or ``None`` when the Edit target file is missing
        (the Edit tool itself will then reject the call — we just skip scanning).
    """
    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", {})
    if tool_name == "Write":
        return tool_input.get("content", "")
    file_path = tool_input.get("file_path", "")
    path = pathlib.Path(file_path)
    if not path.is_file():
        sys.stderr.write(f"LOG:Edit target {file_path} not found; skipping scan\n")
        return None
    content = path.read_text()
    old = tool_input.get("old_string", "")
    new = tool_input.get("new_string", "")
    if old and old in content:
        content = content.replace(old, new, 1)
    return content


def main() -> int:
    """Read JSON on stdin, scan synthesised content, write violations to stdout.

    :returns: Always 0 — the wrapper decides the hook's own exit code from whether stdout has any
        lines.
    """
    payload = json.loads(sys.stdin.read())
    content = _post_edit_content(payload)
    if content is None:
        return 0
    out = sys.stdout
    for lineno, key, header_lineno, text in _scan(content.splitlines()):
        out.write(f"{lineno}\t{key}\t{header_lineno}\t{text}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
