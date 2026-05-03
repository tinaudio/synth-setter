"""PR-comment-posting helper for the repo-review and repo-review-full skills.

Reads a JSON review payload on stdin (list of findings), resolves each finding's
anchor against the PR's diff hunks, falls back to the nearest in-hunk line with a
cross-ref note when the natural line isn't anchorable, then submits one
GitHub Pull Request review with every finding as an unresolved inline thread.

Input format on stdin:

    {
      "pr_number": 777,
      "repo": "tinaudio/synth-setter",
      "review_body": "Top-level review summary (markdown).",
      "findings": [
        {
          "path": "configs/compute/oci-cpu-template.yaml",
          "line": 147,
          "body": "**[code-health:block]** ..."
        },
        ...
      ]
    }

The submitted review uses event=COMMENT so threads stay open (not approved or
rejected). Verified end-to-end on PR #777 — see that PR's description for the
trace.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Hunk:
    """One unified-diff hunk's right-side line range, inclusive on both ends."""

    new_start: int
    new_end: int

    def contains(self, line: int) -> bool:
        """Return True if `line` is within this hunk's right-side range."""
        return self.new_start <= line <= self.new_end


@dataclass(frozen=True)
class Finding:
    """One review finding — a target file/line and a comment body."""

    path: str
    line: int
    body: str


@dataclass(frozen=True)
class AnchoredComment:
    """A finding resolved to a valid in-diff anchor; `rewritten` flags fallbacks."""

    path: str
    line: int
    body: str
    original_line: int
    rewritten: bool


HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def fetch_diff(repo: str, pr_number: int) -> str:
    """Fetch the PR diff via gh — raw unified diff."""
    result = subprocess.run(
        ["gh", "pr", "diff", str(pr_number), "--repo", repo],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        sys.exit(result.returncode)
    return result.stdout


def parse_diff_hunks(diff_text: str) -> dict[str, list[Hunk]]:
    """Parse a unified diff into {path: [Hunk, ...]} on the right (new) side.

    The right-side range covers added (`+`) and unchanged (` `) lines — both are valid GitHub PR-
    review-comment anchors. Only deleted (`-`) lines are skipped because they don't exist on the
    new side.
    """
    hunks: dict[str, list[Hunk]] = {}
    current_path: str | None = None
    new_line: int | None = None
    new_start: int | None = None

    def flush() -> None:
        if current_path is not None and new_start is not None and new_line is not None:
            new_end = new_line - 1
            if new_end >= new_start:
                hunks[current_path].append(Hunk(new_start, new_end))

    for raw_line in diff_text.splitlines():
        if raw_line.startswith("+++ b/"):
            flush()
            current_path = raw_line[len("+++ b/") :]
            hunks.setdefault(current_path, [])
            new_line = None
            new_start = None
            continue

        match = HUNK_HEADER_RE.match(raw_line)
        if match:
            flush()
            new_start = int(match.group(1))
            new_line = new_start
            continue

        if current_path is None or new_line is None:
            continue

        if raw_line.startswith("+") and not raw_line.startswith("+++"):
            new_line += 1
        elif raw_line.startswith(" "):
            new_line += 1
        # `-` lines: skip; only present in the old side.

    flush()

    return hunks


def nearest_in_hunk_line(line: int, hunks: list[Hunk]) -> int | None:
    """Return the line in `hunks` closest to `line`, or None if `hunks` is empty."""
    if not hunks:
        return None
    best: tuple[int, int] | None = None
    for hunk in hunks:
        if hunk.contains(line):
            return line
        candidates = (hunk.new_start, hunk.new_end)
        for candidate in candidates:
            distance = abs(candidate - line)
            if best is None or distance < best[0]:
                best = (distance, candidate)
    return None if best is None else best[1]


def anchor_finding(
    finding: Finding, hunks_by_path: dict[str, list[Hunk]]
) -> AnchoredComment | None:
    """Pin a finding to a valid review-comment anchor; rewrite body on fallback.

    Returns None if the path has no hunks at all (file not in diff) — caller can roll those into
    the top-level review body.
    """
    hunks = hunks_by_path.get(finding.path, [])
    target = nearest_in_hunk_line(finding.line, hunks)
    if target is None:
        return None
    rewritten = target != finding.line
    if rewritten:
        prefix = (
            f"*(anchored at line {target}; finding is on line {finding.line}, "
            "outside the diff hunks)*\n\n"
        )
        body = prefix + finding.body
    else:
        body = finding.body
    return AnchoredComment(
        path=finding.path,
        line=target,
        body=body,
        original_line=finding.line,
        rewritten=rewritten,
    )


def build_review_payload(
    review_body: str, anchored: list[AnchoredComment], orphaned: list[Finding]
) -> dict[str, object]:
    """Compose the JSON body for POST /repos/.../pulls/N/reviews."""
    body = review_body
    if orphaned:
        body += "\n\n## Findings on files outside the diff\n\n"
        for finding in orphaned:
            indented = finding.body.replace("\n", "\n  ")
            body += f"- `{finding.path}:{finding.line}` — {indented}\n"
    return {
        "body": body,
        "event": "COMMENT",
        "comments": [
            {"path": comment.path, "line": comment.line, "body": comment.body}
            for comment in anchored
        ],
    }


def submit_review(repo: str, pr_number: int, payload: dict[str, object]) -> dict:
    """POST the review and return the parsed response."""
    payload_json = json.dumps(payload)
    result = subprocess.run(
        [
            "gh",
            "api",
            "-X",
            "POST",
            f"repos/{repo}/pulls/{pr_number}/reviews",
            "--input",
            "-",
        ],
        input=payload_json,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        sys.exit(result.returncode)
    return json.loads(result.stdout)


def main() -> int:
    """CLI entry point — read JSON request on stdin, post or dry-run the review."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved payload as JSON instead of submitting.",
    )
    args = parser.parse_args()

    request = json.load(sys.stdin)
    repo: str = request["repo"]
    pr_number: int = int(request["pr_number"])
    review_body: str = request.get("review_body", "")
    findings = [Finding(**raw) for raw in request["findings"]]

    diff_text = fetch_diff(repo, pr_number)
    hunks_by_path = parse_diff_hunks(diff_text)

    anchored: list[AnchoredComment] = []
    orphaned: list[Finding] = []
    for finding in findings:
        result = anchor_finding(finding, hunks_by_path)
        if result is None:
            orphaned.append(finding)
        else:
            anchored.append(result)

    payload = build_review_payload(review_body, anchored, orphaned)

    sys.stderr.write(
        f"Resolved {len(anchored)} inline anchor(s) "
        f"({sum(1 for c in anchored if c.rewritten)} fell back to nearest-in-hunk); "
        f"{len(orphaned)} finding(s) on files outside the diff rolled into review body.\n"
    )

    if args.dry_run:
        print(json.dumps(payload, indent=2))
        return 0

    response = submit_review(repo, pr_number, payload)
    print(response.get("html_url", json.dumps(response)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
