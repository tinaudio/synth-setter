"""Scanner for YAML block-scalar `#`-comment violations.

Companion to `no-yaml-run-comments.sh`. Reads the full PreToolUse JSON payload
on stdin and emits one tab-delimited violation per line on stdout:
``<line>\\t<block_key>\\t<header_lineno>\\t<text>``. The shell wrapper formats
the BLOCKED message; this module only locates offenders so the Edit/Write tool
call is gated before it lands content where a stray ``'``/`` ` ``/``$``/``\\``
inside a block-scalar comment would trigger shell expansion on the runner.

Only violations newly introduced by the edit are reported: the scanner
diffs the pre-edit and post-edit violation multisets so unrelated edits to
files with existing block-scalar comments pass through.
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


def _safe_read(path: pathlib.Path) -> str:
    """Return ``path``'s text, or ``""`` on any OSError.

    Permission denied / EISDIR / ENOTDIR turn into "empty pre-edit" rather
    than a Python traceback that the shell trap would re-format as an
    opaque internal-error block.

    :param path: File to read.
    :returns: File contents, or ``""`` if the file is unreadable or missing.
    """
    try:
        return path.read_text()
    except OSError:
        return ""


def _edit_contents(payload: dict) -> tuple[str, str] | None:
    """Return ``(pre_edit, post_edit)`` text for the tool call.

    :param payload: Parsed PreToolUse JSON envelope.
    :returns: ``(pre, post)`` strings, or ``None`` when the Edit target file is
        missing (the Edit tool itself will then reject the call — we just skip
        scanning). For Write, ``pre`` is the existing file content if any.
    """
    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", {})
    file_path = tool_input.get("file_path", "")
    path = pathlib.Path(file_path)
    pre = _safe_read(path) if path.is_file() else ""
    if tool_name == "Write":
        return pre, tool_input.get("content", "")
    if not path.is_file():
        sys.stderr.write(f"LOG:Edit target {file_path} not found; skipping scan\n")
        return None
    old = tool_input.get("old_string", "")
    new = tool_input.get("new_string", "")
    if old and old in pre:
        count = -1 if tool_input.get("replace_all") else 1
        post = pre.replace(old, new, count)
    else:
        post = pre
    return pre, post


def _new_violations(
    pre: list[tuple[int, str, int, str]], post: list[tuple[int, str, int, str]]
) -> list[tuple[int, str, int, str]]:
    """Return post-edit violations not accounted for by a matching pre-edit one.

    Multiset diff by ``(block_key, text)`` so line-number shifts don't cause the
    same comment to look "new". A pre-existing violation with matching key and
    text consumes one instance from the post-edit set.

    :param pre: Violations found in the file before the edit.
    :param post: Violations found after the synthesised edit.
    :returns: Violations attributable to this edit, in document order.
    """
    pre_counts: dict[tuple[str, str], int] = {}
    for _, block_key, _, text in pre:
        key = (block_key, text)
        pre_counts[key] = pre_counts.get(key, 0) + 1
    new: list[tuple[int, str, int, str]] = []
    for entry in post:
        _, block_key, _, text = entry
        key = (block_key, text)
        if pre_counts.get(key, 0) > 0:
            pre_counts[key] -= 1
            continue
        new.append(entry)
    return new


def main() -> int:
    """Read JSON on stdin, scan synthesised content, write violations to stdout.

    :returns: Always 0 — the wrapper decides the hook's own exit code from whether stdout has any
        lines.
    """
    payload = json.loads(sys.stdin.read())
    contents = _edit_contents(payload)
    if contents is None:
        return 0
    pre_text, post_text = contents
    pre_violations = _scan(pre_text.splitlines())
    post_violations = _scan(post_text.splitlines())
    out = sys.stdout
    for lineno, key, header_lineno, text in _new_violations(pre_violations, post_violations):
        out.write(f"{lineno}\t{key}\t{header_lineno}\t{text}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
