"""State derivation for the /handoff skill.

Reads chain.yaml + live `gh` queries + `git worktree list` and returns a
DerivedState the templates can render. Pure parsing/aggregation lives in
free functions so unit tests can exercise the logic without mocking the
network — subprocess wrappers are kept thin and isolated at module scope.

Public entry point: derive_state(...).
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import yaml

HANDOFF_HEADER_PREFIX = "## Handoff update"
TASK_TITLE_RE_TEMPLATE = r"^{prefix}\.(\d+)\b"
COPILOT_LOGIN_RE = re.compile(r"copilot", re.IGNORECASE)

# Repo root anchor for `git -C` so worktree/branch queries succeed regardless
# of the caller's CWD. Layout: <repo>/.claude/skills/handoff/helpers/<this file>
REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class LinkedIssue:
    """How a chain PR's tracking issue is sourced — existing or to-be-created."""

    strategy: str = "create_under_phase"
    existing: int | None = None
    parent_issue: int | None = None


@dataclass
class ChainPR:
    """One plan-PR row in `chain.yaml`."""

    id: str
    title: str
    source_commits: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    linked_issue: LinkedIssue = field(default_factory=LinkedIssue)
    status: str = "pending"
    pr_number: int | None = None
    depends_on: list[str] = field(default_factory=list)


@dataclass
class Chain:
    """The full chain manifest — header metadata plus a list of plan PRs."""

    tracking_issue: int
    repo: str
    parent_phase: int
    task_prefix: str
    plan_prs: list[ChainPR] = field(default_factory=list)


@dataclass
class PriorHandoff:
    """One prior handoff comment on the tracking issue.

    `created_at` is an ISO8601 UTC string straight from the GitHub API — kept
    as a string because we only ever compare it lexicographically (ISO8601 is
    lexicographically ordered) and pass it through to the next gh query.
    """

    comment_id: int
    url: str
    created_at: str
    body: str


@dataclass
class ClosedPR:
    """One merged PR that referenced the tracking issue since the prior handoff."""

    number: int
    title: str
    merged_at: str
    closes_issues: list[int]


@dataclass
class InFlightPR:
    """One open PR's health snapshot — feeds the comment's In-flight section."""

    number: int
    title: str
    branch: str
    head_oid: str
    head_committer_date: str
    mergeable: str
    review_decision: str
    checks_state: str
    unresolved_threads: int
    new_copilot_comments_since_push: int


@dataclass
class PhaseTaskNumbering:
    """Phase N's used `Task N.M` numbers and the next free integer."""

    phase_number: int
    task_prefix: str
    used_numbers: list[int]

    @property
    def next_free(self) -> int:
        """Return one past the maximum used number (or 1 if empty)."""
        return (max(self.used_numbers) + 1) if self.used_numbers else 1


@dataclass
class WorktreeEntry:
    """One `git worktree list` row, cross-referenced with its PR state."""

    path: str
    branch: str | None
    head: str
    associated_pr_number: int | None
    associated_pr_state: str | None
    safe_to_remove: bool


@dataclass
class DerivedState:
    """The full bundle of state the templates render against."""

    repo: str
    tracking_issue: int
    chain: Chain
    prior_handoffs: list[PriorHandoff]
    done_since: list[ClosedPR]
    in_flight: list[InFlightPR]
    phase_numbering: PhaseTaskNumbering
    worktrees: list[WorktreeEntry]
    now_utc: str


def run_gh(args: list[str], *, input_text: str | None = None) -> str:
    """Invoke the gh CLI and return stdout.

    Raises CalledProcessError on failure.
    """
    result = subprocess.run(  # noqa: S603 — argv list, no shell
        ["gh", *args],
        capture_output=True,
        text=True,
        check=True,
        input=input_text,
    )
    return result.stdout


def run_git(args: list[str]) -> str:
    """Invoke git anchored at REPO_ROOT and return stdout.

    Raises CalledProcessError on failure.
    """
    result = subprocess.run(  # noqa: S603 — argv list, no shell
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


CHAIN_HEADER = """\
# Structured manifest of the remaining PR chain rooted at the tracking issue.
#
# The /handoff skill reads this on every invocation, updates `status` and
# `pr_number` from live `gh` queries, and writes the file back only when one
# of those columns changes. Merged PRs automatically surface in the "Done
# since prior handoff" table; pending and in-flight PRs surface in the
# "Remaining chain" table.
#
# To start a new chain, copy this file (e.g. chain.wds.yaml) and edit. One
# tracking issue per chain — multi-issue chains are out of scope.
"""


def load_chain(path: Path) -> Chain:
    """Read chain.yaml into a Chain dataclass."""
    raw = yaml.safe_load(path.read_text())
    return _chain_from_dict(raw)


def save_chain(path: Path, chain: Chain) -> None:
    """Write a Chain dataclass back to chain.yaml with the standard header."""
    body = yaml.safe_dump(_chain_to_dict(chain), sort_keys=False, width=1000, allow_unicode=True)
    path.write_text(CHAIN_HEADER + body)


def chains_equivalent(a: Chain, b: Chain) -> bool:
    """Return True if two Chains carry the same logical content."""
    return _chain_to_dict(a) == _chain_to_dict(b)


def _chain_from_dict(raw: dict) -> Chain:
    """Build a Chain from its YAML-loaded dict form."""
    plan_prs = [_chain_pr_from_dict(row) for row in raw.get("plan_prs", [])]
    return Chain(
        tracking_issue=int(raw["tracking_issue"]),
        repo=str(raw["repo"]),
        parent_phase=int(raw["parent_phase"]),
        task_prefix=str(raw["task_prefix"]),
        plan_prs=plan_prs,
    )


def _chain_pr_from_dict(raw: dict) -> ChainPR:
    """Build a ChainPR row from its YAML-loaded dict form."""
    linked = raw.get("linked_issue") or {}
    return ChainPR(
        id=str(raw["id"]),
        title=str(raw["title"]),
        source_commits=list(raw.get("source_commits") or []),
        notes=list(raw.get("notes") or []),
        linked_issue=LinkedIssue(
            strategy=str(linked.get("strategy", "create_under_phase")),
            existing=linked.get("existing"),
            parent_issue=linked.get("parent_issue"),
        ),
        status=str(raw.get("status", "pending")),
        pr_number=raw.get("pr_number"),
        depends_on=list(raw.get("depends_on") or []),
    )


def _chain_to_dict(chain: Chain) -> dict:
    """Render a Chain to its YAML-serializable dict form."""
    return {
        "tracking_issue": chain.tracking_issue,
        "repo": chain.repo,
        "parent_phase": chain.parent_phase,
        "task_prefix": chain.task_prefix,
        "plan_prs": [_chain_pr_to_dict(pr) for pr in chain.plan_prs],
    }


def _chain_pr_to_dict(pr: ChainPR) -> dict:
    """Render a ChainPR row to its YAML-serializable dict form."""
    row: dict = {
        "id": pr.id,
        "title": pr.title,
        "source_commits": pr.source_commits,
    }
    if pr.notes:
        row["notes"] = pr.notes
    row["linked_issue"] = {
        "strategy": pr.linked_issue.strategy,
        "existing": pr.linked_issue.existing,
    }
    if pr.linked_issue.parent_issue is not None:
        row["linked_issue"]["parent_issue"] = pr.linked_issue.parent_issue
    row["status"] = pr.status
    row["pr_number"] = pr.pr_number
    row["depends_on"] = pr.depends_on
    return row


def filter_handoff_comments(comments: Iterable[dict]) -> list[PriorHandoff]:
    """Pick handoff-update comments out of the tracking issue's full comment list."""
    handoffs: list[PriorHandoff] = []
    for comment in comments:
        body = comment.get("body") or ""
        if not body.lstrip().startswith(HANDOFF_HEADER_PREFIX):
            continue
        handoffs.append(
            PriorHandoff(
                comment_id=int(comment.get("databaseId") or comment.get("id") or 0),
                url=str(comment.get("url") or ""),
                created_at=str(comment.get("createdAt") or ""),
                body=body,
            )
        )
    handoffs.sort(key=lambda h: h.created_at, reverse=True)
    return handoffs


def fetch_issue_comments(repo: str, issue_number: int) -> list[dict]:
    """Page through every comment on the tracking issue via GraphQL."""
    owner, name = repo.split("/", 1)
    nodes: list[dict] = []
    after: str | None = None
    query = """
      query($owner: String!, $name: String!, $number: Int!, $after: String) {
        repository(owner: $owner, name: $name) {
          issue(number: $number) {
            comments(first: 100, after: $after) {
              nodes { databaseId url body createdAt }
              pageInfo { hasNextPage endCursor }
            }
          }
        }
      }
    """
    while True:
        args = [
            "api",
            "graphql",
            "-f",
            f"query={query}",
            "-f",
            f"owner={owner}",
            "-f",
            f"name={name}",
            "-F",
            f"number={issue_number}",
        ]
        if after is not None:
            args += ["-f", f"after={after}"]
        payload = json.loads(run_gh(args))
        page = payload["data"]["repository"]["issue"]["comments"]
        nodes.extend(page["nodes"])
        if not page["pageInfo"]["hasNextPage"]:
            return nodes
        after = page["pageInfo"]["endCursor"]


def fetch_done_since(repo: str, tracking_issue: int, since_iso: str | None) -> list[ClosedPR]:
    """Return merged PRs referencing the tracking issue since `since_iso`.

    `since_iso=None` means "no lower bound" (first handoff). We let GitHub's PR search filter by
    body-mention of the tracking issue — picks up both `Closes #<N>` and `Refs #<N>` flavors.
    """
    search_terms = [f"#{tracking_issue}"]
    if since_iso:
        search_terms.append(f"merged:>{since_iso}")
    args = [
        "pr",
        "list",
        "--repo",
        repo,
        "--state",
        "merged",
        "--search",
        " ".join(search_terms),
        "--json",
        "number,title,mergedAt,closingIssuesReferences",
        "--limit",
        "100",
    ]
    raw = json.loads(run_gh(args))
    return [_closed_pr_from_dict(row) for row in raw]


def _closed_pr_from_dict(raw: dict) -> ClosedPR:
    """Build a ClosedPR from gh's `pr list --json` row."""
    closes = []
    for ref in raw.get("closingIssuesReferences") or []:
        if ref.get("number") is not None:
            closes.append(int(ref["number"]))
    return ClosedPR(
        number=int(raw["number"]),
        title=str(raw.get("title", "")),
        merged_at=str(raw.get("mergedAt", "")),
        closes_issues=closes,
    )


def fetch_in_flight(repo: str, tracking_issue: int) -> list[InFlightPR]:
    """Return open PRs that reference the tracking issue, with health fields."""
    args = [
        "pr",
        "list",
        "--repo",
        repo,
        "--state",
        "open",
        "--search",
        f"#{tracking_issue}",
        "--json",
        "number",
        "--limit",
        "50",
    ]
    raw = json.loads(run_gh(args))
    numbers = [int(row["number"]) for row in raw]
    return [fetch_one_in_flight(repo, n) for n in numbers]


def fetch_one_in_flight(repo: str, pr_number: int) -> InFlightPR:
    """Fetch the full health snapshot for one in-flight PR."""
    fields = ",".join(
        [
            "number",
            "title",
            "headRefName",
            "headRefOid",
            "mergeable",
            "reviewDecision",
            "statusCheckRollup",
        ]
    )
    raw = json.loads(
        run_gh(
            ["pr", "view", str(pr_number), "--repo", repo, "--json", fields],
        )
    )
    head_committer_date = _fetch_commit_committer_date(repo, raw["headRefOid"])
    new_copilot = _count_new_copilot_comments(repo, pr_number, head_committer_date)
    unresolved = _fetch_unresolved_threads(repo, pr_number)
    return InFlightPR(
        number=int(raw["number"]),
        title=str(raw["title"]),
        branch=str(raw["headRefName"]),
        head_oid=str(raw["headRefOid"]),
        head_committer_date=head_committer_date,
        mergeable=str(raw.get("mergeable") or "UNKNOWN"),
        review_decision=str(raw.get("reviewDecision") or "REVIEW_REQUIRED"),
        checks_state=summarize_checks(raw.get("statusCheckRollup") or []),
        unresolved_threads=unresolved,
        new_copilot_comments_since_push=new_copilot,
    )


def _fetch_unresolved_threads(repo: str, pr_number: int) -> int:
    """Count unresolved review threads via GraphQL (not exposed on `pr view`)."""
    owner, name = repo.split("/", 1)
    query = """
      query($owner: String!, $name: String!, $number: Int!, $after: String) {
        repository(owner: $owner, name: $name) {
          pullRequest(number: $number) {
            reviewThreads(first: 100, after: $after) {
              nodes { isResolved }
              pageInfo { hasNextPage endCursor }
            }
          }
        }
      }
    """
    after: str | None = None
    nodes: list[dict] = []
    while True:
        args = [
            "api",
            "graphql",
            "-f",
            f"query={query}",
            "-f",
            f"owner={owner}",
            "-f",
            f"name={name}",
            "-F",
            f"number={pr_number}",
        ]
        if after is not None:
            args += ["-f", f"after={after}"]
        try:
            raw = json.loads(run_gh(args))
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            return 0
        page = raw["data"]["repository"]["pullRequest"]["reviewThreads"]
        nodes.extend(page["nodes"])
        if not page["pageInfo"]["hasNextPage"]:
            break
        after = page["pageInfo"]["endCursor"]
    return count_unresolved_threads(nodes)


def _fetch_commit_committer_date(repo: str, sha: str) -> str:
    """Return the committer date for `sha` as ISO8601, or empty on failure.

    `gh api --jq` prints scalar jq results unquoted, so we use the stripped stdout directly rather
    than `json.loads()` — see review on PR #950.
    """
    try:
        return run_gh(
            ["api", f"repos/{repo}/commits/{sha}", "--jq", ".commit.committer.date"]
        ).strip()
    except subprocess.CalledProcessError:
        return ""


def _count_new_copilot_comments(repo: str, pr_number: int, since_iso: str) -> int:
    """Count Copilot review comments created strictly after `since_iso`."""
    if not since_iso:
        return 0
    try:
        body = run_gh(
            [
                "api",
                f"repos/{repo}/pulls/{pr_number}/comments",
                "--paginate",
                "--jq",
                "[.[] | {login: .user.login, created_at}]",
            ]
        )
        comments = json.loads(body)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return 0
    return count_copilot_after(comments, since_iso)


def count_copilot_after(comments: list[dict], since_iso: str) -> int:
    """Pure helper — counts Copilot comments strictly after `since_iso`.

    Empty `since_iso` returns 0: we can't know "new since push" without a
    push timestamp, so suppressing the count beats reporting a misleading one.
    """
    if not since_iso:
        return 0
    total = 0
    for comment in comments:
        login = str(comment.get("login") or "")
        if not COPILOT_LOGIN_RE.search(login):
            continue
        created = str(comment.get("created_at") or "")
        if created <= since_iso:
            continue
        total += 1
    return total


def summarize_checks(rollup: list[dict]) -> str:
    """Collapse a status-check rollup into one of: passing / pending / failing."""
    has_failing = False
    has_pending = False
    for entry in rollup:
        conclusion = str(entry.get("conclusion") or "").upper()
        state = str(entry.get("state") or "").upper()
        status = str(entry.get("status") or "").upper()
        if conclusion in {"FAILURE", "TIMED_OUT", "STARTUP_FAILURE", "ACTION_REQUIRED"}:
            has_failing = True
            continue
        if state in {"FAILURE", "ERROR"}:
            has_failing = True
            continue
        if status in {"IN_PROGRESS", "QUEUED", "PENDING", "WAITING"}:
            has_pending = True
            continue
        if state == "PENDING":
            has_pending = True
    if has_failing:
        return "failing"
    if has_pending:
        return "pending"
    return "passing"


def count_unresolved_threads(threads: list[dict]) -> int:
    """Count review threads that are still unresolved."""
    return sum(1 for thread in threads if not bool(thread.get("isResolved")))


def fetch_phase_numbering(repo: str, phase_number: int, task_prefix: str) -> PhaseTaskNumbering:
    """Walk Phase N's sub-issues, regex-extract task numbers from titles.

    Paginates GraphQL with `after`/`pageInfo.endCursor` so phases with >100
    sub-issues don't silently truncate the used-numbers list.
    """
    owner, name = repo.split("/", 1)
    query = """
      query($owner: String!, $name: String!, $number: Int!, $after: String) {
        repository(owner: $owner, name: $name) {
          issue(number: $number) {
            subIssues(first: 100, after: $after) {
              nodes { number title }
              pageInfo { hasNextPage endCursor }
            }
          }
        }
      }
    """
    after: str | None = None
    nodes: list[dict] = []
    while True:
        args = [
            "api",
            "graphql",
            "-f",
            f"query={query}",
            "-f",
            f"owner={owner}",
            "-f",
            f"name={name}",
            "-F",
            f"number={phase_number}",
        ]
        if after is not None:
            args += ["-f", f"after={after}"]
        raw = json.loads(run_gh(args))
        page = raw["data"]["repository"]["issue"]["subIssues"]
        nodes.extend(page["nodes"])
        if not page["pageInfo"]["hasNextPage"]:
            break
        after = page["pageInfo"]["endCursor"]
    titles = [str(node.get("title") or "") for node in nodes]
    used = extract_task_numbers(titles, task_prefix)
    return PhaseTaskNumbering(
        phase_number=phase_number, task_prefix=task_prefix, used_numbers=used
    )


def extract_task_numbers(titles: Iterable[str], task_prefix: str) -> list[int]:
    """Pull `<task_prefix>.N` numbers out of issue titles.

    Pure for unit tests.
    """
    regex = re.compile(TASK_TITLE_RE_TEMPLATE.format(prefix=re.escape(task_prefix)))
    used: list[int] = []
    for title in titles:
        match = regex.match(title)
        if not match:
            continue
        used.append(int(match.group(1)))
    used.sort()
    return used


def list_worktrees() -> list[dict]:
    """Parse `git worktree list --porcelain` into a list of plain dicts."""
    raw = run_git(["worktree", "list", "--porcelain"])
    return parse_worktree_porcelain(raw)


def parse_worktree_porcelain(text: str) -> list[dict]:
    """Pure parser for `git worktree list --porcelain` output."""
    entries: list[dict] = []
    current: dict = {}
    for line in text.splitlines():
        if line.strip() == "":
            if current:
                entries.append(current)
                current = {}
            continue
        if line.startswith("worktree "):
            current["path"] = line[len("worktree ") :].strip()
        elif line.startswith("HEAD "):
            current["head"] = line[len("HEAD ") :].strip()
        elif line.startswith("branch "):
            ref = line[len("branch ") :].strip()
            current["branch"] = ref.removeprefix("refs/heads/")
        elif line.startswith("detached"):
            current["detached"] = True
        elif line.startswith("bare"):
            current["bare"] = True
    if current:
        entries.append(current)
    return entries


def annotate_worktrees(
    raw_entries: list[dict],
    pr_by_branch: dict[str, tuple[int, str]],
) -> list[WorktreeEntry]:
    """Cross-reference each worktree with its branch's PR state.

    `pr_by_branch` maps branch -> (pr_number, state). State values match
    `gh pr view --json state` — MERGED / OPEN / CLOSED.
    """
    out: list[WorktreeEntry] = []
    for entry in raw_entries:
        if entry.get("bare"):
            continue
        branch = entry.get("branch")
        pr_pair = pr_by_branch.get(branch) if branch else None
        pr_number = pr_pair[0] if pr_pair else None
        pr_state = pr_pair[1] if pr_pair else None
        out.append(
            WorktreeEntry(
                path=str(entry.get("path", "")),
                branch=branch,
                head=str(entry.get("head", "")),
                associated_pr_number=pr_number,
                associated_pr_state=pr_state,
                safe_to_remove=(pr_state in {"MERGED", "CLOSED"}),
            )
        )
    return out


def fetch_pr_state_by_branch(repo: str, branches: list[str]) -> dict[str, tuple[int, str]]:
    """For each branch name, look up its associated PR's number + state."""
    out: dict[str, tuple[int, str]] = {}
    for branch in branches:
        if not branch:
            continue
        try:
            raw = json.loads(
                run_gh(
                    [
                        "pr",
                        "view",
                        branch,
                        "--repo",
                        repo,
                        "--json",
                        "number,state",
                    ]
                )
            )
        except subprocess.CalledProcessError:
            continue
        out[branch] = (int(raw["number"]), str(raw.get("state", "")))
    return out


def update_chain_from_state(
    chain: Chain,
    done_since: list[ClosedPR],
    in_flight: list[InFlightPR],
) -> Chain:
    """Reconcile chain.yaml's status/pr_number columns against live PR state.

    Matching rule: chain title prefix vs. PR title prefix. PR titles often
    add a `(#N)` suffix that conventional-commit linters insert at merge —
    we match on the leading `prefix:` segment, which is stable.
    """
    merged_by_prefix = {_title_prefix(pr.title): pr for pr in done_since}
    open_by_prefix = {_title_prefix(pr.title): pr for pr in in_flight}
    new_rows: list[ChainPR] = []
    for row in chain.plan_prs:
        chain_prefix = _title_prefix(row.title)
        merged_pr = merged_by_prefix.get(chain_prefix)
        open_pr = open_by_prefix.get(chain_prefix)
        if merged_pr is not None:
            new_rows.append(replace(row, status="merged", pr_number=merged_pr.number))
        elif open_pr is not None:
            new_rows.append(replace(row, status="in_flight", pr_number=open_pr.number))
        else:
            new_rows.append(replace(row))
    return Chain(
        tracking_issue=chain.tracking_issue,
        repo=chain.repo,
        parent_phase=chain.parent_phase,
        task_prefix=chain.task_prefix,
        plan_prs=new_rows,
    )


def _title_prefix(title: str) -> str:
    """Return the conventional-commit prefix segment, lowercased.

    `feat(pipeline): foo (#123)` -> `feat(pipeline): foo`. We strip the trailing `(#N)` so chain-
    row titles (which don't carry the number) match merged-PR titles (which often do).
    """
    cleaned = re.sub(r"\s*\(#\d+\)\s*$", "", title.strip())
    return cleaned.lower()


def derive_state(
    repo: str,
    tracking_issue: int,
    chain_path: Path,
    *,
    now: datetime | None = None,
) -> DerivedState:
    """Top-level orchestration.

    Performs every gh + git call, returns the bundle.
    """
    chain = load_chain(chain_path)
    # `tracking_issue` / `repo` args control which issue we *query*. They are
    # session-scoped overrides from the CLI — never propagate them back into
    # the manifest, or `--issue 999` would silently rewrite chain.yaml.
    comments = fetch_issue_comments(repo, tracking_issue)
    prior_handoffs = filter_handoff_comments(comments)
    since_iso = prior_handoffs[0].created_at if prior_handoffs else None
    done_since = fetch_done_since(repo, tracking_issue, since_iso)
    in_flight = fetch_in_flight(repo, tracking_issue)
    chain = update_chain_from_state(chain, done_since, in_flight)
    phase_numbering = fetch_phase_numbering(repo, chain.parent_phase, chain.task_prefix)
    raw_worktrees = list_worktrees()
    branches = [entry.get("branch") for entry in raw_worktrees if entry.get("branch")]
    pr_by_branch = fetch_pr_state_by_branch(repo, [b for b in branches if b])
    worktrees = annotate_worktrees(raw_worktrees, pr_by_branch)
    return DerivedState(
        repo=repo,
        tracking_issue=tracking_issue,
        chain=chain,
        prior_handoffs=prior_handoffs,
        done_since=done_since,
        in_flight=in_flight,
        phase_numbering=phase_numbering,
        worktrees=worktrees,
        now_utc=(now or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC"),
    )
