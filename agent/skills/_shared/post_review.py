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

The review event comes from the request's optional top-level ``event`` field
(default ``COMMENT``; one of ``COMMENT`` / ``REQUEST_CHANGES`` / ``APPROVE``).
GitHub rejects ``REQUEST_CHANGES`` and ``APPROVE`` on the bot's own PR with an
HTTP 422, so ``submit_review`` retries once as ``COMMENT`` with a banner
prepended to the body — preserving the finding threads even when the API won't
let the bot formally request changes.

Typical invocation::

    python3 agent/skills/_shared/post_review.py < payload.json
    python3 agent/skills/_shared/post_review.py --dry-run < payload.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Hunk:
    """One unified-diff hunk's right-side line range.

    .. attribute :: new_start
        :type: int

        First line in the right-side hunk range.

    .. attribute :: new_end
        :type: int

        Last line in the right-side hunk range.
    """

    new_start: int
    new_end: int

    def contains(self, line: int) -> bool:
        """Return whether `line` is within this hunk's right-side range.

        :param line: Right-side line number to check.
        :returns: True when the line is inside the hunk range.
        :rtype: bool
        """
        return self.new_start <= line <= self.new_end


@dataclass(frozen=True)
class Finding:
    """One review finding.

    .. attribute :: path
        :type: str

        Repository-relative file path for the finding.

    .. attribute :: line
        :type: int

        Right-side line number for the finding.

    .. attribute :: body
        :type: str

        Markdown review comment body.
    """

    path: str
    line: int
    body: str


@dataclass(frozen=True)
class AnchoredComment:
    """A finding resolved to a valid in-diff anchor.

    .. attribute :: path
        :type: str

        Repository-relative file path for the review comment.

    .. attribute :: line
        :type: int

        Right-side line number used as the GitHub review anchor.

    .. attribute :: body
        :type: str

        Markdown review comment body.

    .. attribute :: original_line
        :type: int

        Original finding line before any fallback anchoring.

    .. attribute :: rewritten
        :type: bool

        Whether the body was prefixed with fallback-anchor context.
    """

    path: str
    line: int
    body: str
    original_line: int
    rewritten: bool


HUNK_HEADER_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def gh_executable() -> str:
    """Return the absolute gh executable path or exit with an error.

    :returns: Absolute path to the `gh` executable.
    :rtype: str
    """
    path = shutil.which("gh")
    if path is None:
        sys.stderr.write("gh executable not found on PATH\n")
        sys.exit(127)
    return path


def fetch_diff(repo: str, pr_number: int) -> str:
    """Fetch the PR diff via gh.

    :param repo: GitHub repository in `owner/name` form.
    :param pr_number: Pull request number.
    :returns: Raw unified diff text.
    :rtype: str
    """
    result = subprocess.run(  # noqa: S603
        [gh_executable(), "pr", "diff", str(pr_number), "--repo", repo],
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

    :param diff_text: Raw unified diff text.
    :returns: Mapping from repository-relative file path to right-side hunk ranges.
    :rtype: dict[str, list[Hunk]]
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
    """Return the line in `hunks` closest to `line`.

    :param line: Preferred right-side line number.
    :param hunks: Candidate right-side hunk ranges.
    :returns: The preferred line when anchorable, the nearest hunk boundary, or None when no hunks
        exist.
    :rtype: int | None
    """
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

    :param finding: Review finding to anchor.
    :param hunks_by_path: Mapping from file path to right-side hunk ranges.
    :returns: Anchored comment, or None when the finding's file has no diff hunks.
    :rtype: AnchoredComment | None
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


_VALID_EVENTS = frozenset({"COMMENT", "REQUEST_CHANGES", "APPROVE"})


def build_review_payload(
    review_body: str,
    anchored: list[AnchoredComment],
    orphaned: list[Finding],
    event: str = "COMMENT",
) -> dict[str, object]:
    """Compose the JSON body for POST /repos/.../pulls/N/reviews.

    :param review_body: Top-level review body markdown.
    :param anchored: Findings that can be posted as inline comments.
    :param orphaned: Findings whose files are outside the PR diff.
    :param event: Review event; one of COMMENT, REQUEST_CHANGES, APPROVE.
    :returns: GitHub review API payload. The ``comments`` key is omitted for
        APPROVE with no findings, which GitHub requires.
    :rtype: dict[str, object]
    :raises ValueError: If ``event`` is not a recognized GitHub review event.
    """
    if event not in _VALID_EVENTS:
        raise ValueError(
            f"unsupported review event: {event!r}; expected one of {sorted(_VALID_EVENTS)}"
        )
    body = review_body
    if orphaned:
        body += "\n\n## Findings on files outside the diff\n\n"
        for finding in orphaned:
            indented = finding.body.replace("\n", "\n  ")
            body += f"- `{finding.path}:{finding.line}` — {indented}\n"
    comments = [
        {"path": comment.path, "line": comment.line, "body": comment.body} for comment in anchored
    ]
    payload: dict[str, object] = {"body": body, "event": event}
    # GitHub rejects APPROVE carrying an empty `comments`; omit the key in that case.
    if comments or event != "APPROVE":
        payload["comments"] = comments
    return payload


_SELF_REVIEW_422_RE = re.compile(
    r"can not (?:request changes|approve) (?:on )?your own", re.IGNORECASE
)


def _post_review(
    repo: str, pr_number: int, payload: dict[str, object]
) -> subprocess.CompletedProcess[str]:
    """POST one review payload via `gh api` without inspecting the outcome.

    :param repo: GitHub repository in `owner/name` form.
    :param pr_number: Pull request number.
    :param payload: GitHub review API payload.
    :returns: The completed `gh api` process.
    :rtype: subprocess.CompletedProcess[str]
    """
    return subprocess.run(  # noqa: S603
        [
            gh_executable(),
            "api",
            "-X",
            "POST",
            f"repos/{repo}/pulls/{pr_number}/reviews",
            "--input",
            "-",
        ],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )


def submit_review(
    repo: str, pr_number: int, payload: dict[str, object], fallback_banner: str
) -> dict[str, object]:
    """POST the review, falling back to COMMENT on a self-review 422.

    GitHub rejects REQUEST_CHANGES/APPROVE on the bot's own PR with an HTTP 422.
    On that specific failure, retry once as COMMENT with ``fallback_banner``
    prepended to the body so the blocking intent stays visible.

    :param repo: GitHub repository in `owner/name` form.
    :param pr_number: Pull request number.
    :param payload: GitHub review API payload.
    :param fallback_banner: Banner prepended to the body on the COMMENT retry.
    :returns: Parsed JSON response from GitHub.
    :rtype: dict[str, object]
    """
    result = _post_review(repo, pr_number, payload)
    if result.returncode == 0:
        return json.loads(result.stdout)
    if _SELF_REVIEW_422_RE.search(result.stderr):
        sys.stderr.write(
            "Self-review 422: falling back to event=COMMENT with the fallback banner.\n"
        )
        retry = dict(payload)
        retry["event"] = "COMMENT"
        retry["body"] = f"{fallback_banner}\n\n{payload.get('body', '')}"
        result = _post_review(repo, pr_number, retry)
        if result.returncode == 0:
            return json.loads(result.stdout)
    sys.stderr.write(result.stderr)
    sys.exit(result.returncode)


def main() -> int:
    """Read JSON request on stdin, then post or dry-run the review.

    :returns: Process exit code.
    :rtype: int
    """
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
    event: str = request.get("event", "COMMENT")
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

    payload = build_review_payload(review_body, anchored, orphaned, event=event)

    sys.stderr.write(
        f"Resolved {len(anchored)} inline anchor(s) "
        f"({sum(1 for c in anchored if c.rewritten)} fell back to nearest-in-hunk); "
        f"{len(orphaned)} finding(s) on files outside the diff rolled into review body.\n"
    )

    if args.dry_run:
        sys.stdout.write(json.dumps(payload, indent=2))
        sys.stdout.write("\n")
        return 0

    # Only REQUEST_CHANGES/APPROVE can 422 on a self-review; phrase the banner to match
    # the intent GitHub refused (an APPROVE downgrade carries no "changes required").
    if event == "APPROVE":
        banner = "✅ No findings (self-review: APPROVE not allowed on own PR — posted as COMMENT)"
    else:
        block_count = review_body.count(":block]") + sum(f.body.count(":block]") for f in findings)
        banner = (
            f"⛔ {block_count} BLOCKING finding(s) — changes required "
            "(self-review: posted as COMMENT)"
        )
    response = submit_review(repo, pr_number, payload, fallback_banner=banner)
    html_url = response.get("html_url")
    sys.stdout.write(html_url if isinstance(html_url, str) else json.dumps(response))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
