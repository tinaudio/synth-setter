"""Compose, post, and print the /handoff artifacts.

Reads chain.yaml + live GitHub state via `discover_state`, renders the
comment + prompt templates, optionally posts the comment to the tracking
issue, prints the prompt to stdout, and (default) saves the prompt to
`.claude/handoffs/handoff-YYYY-MM-DD-HHMM.md`.

Usage examples:

    python3 .claude/skills/handoff/helpers/write_handoff.py
    python3 .claude/skills/handoff/helpers/write_handoff.py --issue 882
    python3 .claude/skills/handoff/helpers/write_handoff.py --dry-run
    python3 .claude/skills/handoff/helpers/write_handoff.py --comment-only
    python3 .claude/skills/handoff/helpers/write_handoff.py --prompt-only
    python3 .claude/skills/handoff/helpers/write_handoff.py --force
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

if __package__:
    from . import discover_state as ds
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import discover_state as ds  # type: ignore[import-not-found, no-redef]

SKILL_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = SKILL_DIR / "templates"
CHAIN_PATH = SKILL_DIR / "chain.yaml"
# Anchored to the `.claude/` tree (two levels above the skill dir) so the save location is stable
# regardless of CWD — running the writer from anywhere on disk drops prompts in the same place.
DEFAULT_HANDOFFS_DIR = SKILL_DIR.parent.parent / "handoffs"
IDEMPOTENCY_GUARD = timedelta(minutes=30)


@dataclass(frozen=True)
class Surprise:
    """One user-curated surprise row for the comment's Surprises section."""

    category: str
    note: str


@dataclass
class HandoffContext:
    """Bundles every field the templates reference."""

    repo: str
    tracking_issue: int
    chain: ds.Chain
    prior_handoffs: list[ds.PriorHandoff]
    done_since: list[ds.ClosedPR]
    in_flight: list[ds.InFlightPR]
    phase_numbering: ds.PhaseTaskNumbering
    worktrees: list[ds.WorktreeEntry]
    now_utc: str
    surprises: list[Surprise]
    anti_patterns: list[str]
    handoff_comment_url: str | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the writer's CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--issue", type=int, default=None, help="Override tracking-issue number")
    parser.add_argument("--chain", type=Path, default=CHAIN_PATH, help="Path to chain.yaml")
    parser.add_argument("--repo", type=str, default=None, help="Override repo (owner/name)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Render both, post nothing, print everything"
    )
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--comment-only", action="store_true", help="Post comment, don't print prompt"
    )
    output_mode.add_argument(
        "--prompt-only", action="store_true", help="Print prompt, don't post comment"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the idempotency guard (refuses to post within 30min of a prior handoff)",
    )
    parser.add_argument(
        "--surprise",
        action="append",
        default=[],
        help='Add a surprise. Format: "<category>: <note>". Repeat for multiple.',
    )
    parser.add_argument(
        "--anti-pattern",
        action="append",
        default=[],
        help="Add an anti-pattern line (the prompt's 'don't retry these' list). Repeat for multiple.",
    )
    parser.add_argument(
        "--no-save-prompt",
        action="store_true",
        help="Don't save the prompt to .claude/handoffs/handoff-<timestamp>.md",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """End-to-end: derive state, render, post comment, print + save prompt.

    Catches `subprocess.CalledProcessError` and `json.JSONDecodeError` at the
    top level so `gh`/`git` failures surface as one-line stderr messages
    instead of stack traces (per SKILL.md's Step 4 contract).
    """
    try:
        return _main(argv)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip() if isinstance(exc.stderr, str) else ""
        cmd = " ".join(map(str, exc.cmd)) if isinstance(exc.cmd, (list, tuple)) else str(exc.cmd)
        sys.stderr.write(f"command failed (exit {exc.returncode}): {cmd}\n")
        if stderr:
            sys.stderr.write(stderr + "\n")
        return exc.returncode or 1
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"could not parse JSON from gh: {exc}\n")
        return 1


def _main(argv: list[str] | None) -> int:
    """Inner main — exception handling is owned by `main()`."""
    args = parse_args(argv)

    chain_path = args.chain.resolve()
    if not chain_path.exists():
        sys.stderr.write(f"chain.yaml not found at {chain_path}\n")
        return 2
    chain = ds.load_chain(chain_path)
    repo = args.repo or chain.repo
    tracking_issue = args.issue or chain.tracking_issue

    sys.stderr.write(f"Deriving state for {repo} tracking #{tracking_issue}...\n")
    state = ds.derive_state(repo, tracking_issue, chain_path)

    if not args.force and not args.dry_run:
        guard_msg = check_idempotency_guard(state.prior_handoffs)
        if guard_msg:
            sys.stderr.write(guard_msg + "\nRerun with --force to override.\n")
            return 3

    surprises = [parse_surprise(raw) for raw in args.surprise]
    anti_patterns = list(args.anti_pattern)

    # Don't persist status/pr_number changes back to chain.yaml when a CLI
    # override is in play — the reconciliation was computed against a
    # different issue/repo than the manifest's canonical pair.
    overrides_in_play = args.issue is not None or args.repo is not None
    if not args.dry_run and not overrides_in_play and not ds.chains_equivalent(chain, state.chain):
        ds.save_chain(chain_path, state.chain)
        sys.stderr.write("chain.yaml updated with new status/pr_number values\n")
    elif not args.dry_run and overrides_in_play:
        sys.stderr.write(
            "chain.yaml not updated: --issue/--repo override in play; "
            "status/pr_number reconciliation kept session-local.\n"
        )

    context = HandoffContext(
        repo=state.repo,
        tracking_issue=state.tracking_issue,
        chain=state.chain,
        prior_handoffs=state.prior_handoffs,
        done_since=state.done_since,
        in_flight=state.in_flight,
        phase_numbering=state.phase_numbering,
        worktrees=state.worktrees,
        now_utc=state.now_utc,
        surprises=surprises,
        anti_patterns=anti_patterns,
        handoff_comment_url=None,
    )
    comment_md = render(context, "comment.md.j2")
    prompt_md = render(context, "prompt.md.j2")

    if args.dry_run:
        print(f"===== COMMENT (would-post to issue #{tracking_issue}) =====")
        print(comment_md)
        print("===== PROMPT =====")
        print(prompt_md)
        return 0

    posted_url = None
    if not args.prompt_only:
        posted_url = post_comment(repo, tracking_issue, comment_md)
        sys.stderr.write(f"Posted handoff comment: {posted_url}\n")
        context.handoff_comment_url = posted_url
        prompt_md = render(context, "prompt.md.j2")

    if not args.comment_only:
        if not args.no_save_prompt:
            saved = save_prompt_locally(prompt_md, state.now_utc)
            sys.stderr.write(f"Saved prompt to {saved}\n")
        print(prompt_md)

    return 0


def render(context: HandoffContext, template_name: str) -> str:
    """Render `template_name` against `context`."""
    env = Environment(  # noqa: S701 — output is markdown for GitHub, not HTML
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        undefined=StrictUndefined,
        trim_blocks=False,
        lstrip_blocks=False,
        keep_trailing_newline=True,
    )
    env.globals["render_ascii_graph"] = render_ascii_graph
    template = env.get_template(template_name)
    payload = {
        "repo": context.repo,
        "tracking_issue": context.tracking_issue,
        "chain": context.chain,
        "prior_handoffs": context.prior_handoffs,
        "done_since": context.done_since,
        "in_flight": context.in_flight,
        "phase_numbering": context.phase_numbering,
        "worktrees": context.worktrees,
        "now_utc": context.now_utc,
        "surprises": context.surprises,
        "anti_patterns": context.anti_patterns,
        "handoff_comment_url": context.handoff_comment_url,
    }
    return template.render(**payload)


def render_ascii_graph(rows: list[ds.ChainPR]) -> str:
    """Topological-level ASCII graph of `rows`.

    Each row's level is `1 + max(level(dep))` over dependencies that are
    also in `rows`. Same-level rows are listed on consecutive lines under
    their parent. Pure function (no I/O) — exported as a Jinja global.
    """
    if not rows:
        return "(no remaining PRs)"
    by_id = {row.id: row for row in rows}
    level: dict[str, int] = {}
    for row in rows:
        level[row.id] = _compute_level(row.id, by_id, level)
    levels: dict[int, list[str]] = {}
    for row_id, lvl in level.items():
        levels.setdefault(lvl, []).append(row_id)
    out_lines: list[str] = []
    for lvl in sorted(levels):
        indent = "  " * (lvl - 1)
        for row_id in levels[lvl]:
            row = by_id[row_id]
            deps_in = [d for d in row.depends_on if d in by_id]
            arrow = f"  ← {', '.join(deps_in)}" if deps_in else "  (root)"
            out_lines.append(f"{indent}{row_id}{arrow}")
    return "\n".join(out_lines)


def _compute_level(
    row_id: str,
    by_id: dict[str, ds.ChainPR],
    cache: dict[str, int],
    visiting: tuple[str, ...] = (),
) -> int:
    """Topological level = 1 + max level of dependencies that are in `by_id`.

    Tracks the current DFS path in `visiting` so cycles surface as a
    descriptive ValueError instead of a Python RecursionError.
    """
    if row_id in cache:
        return cache[row_id]
    if row_id in visiting:
        cycle_path = " → ".join([*visiting, row_id])
        raise ValueError(f"Cycle detected in chain.yaml dependencies: {cycle_path}")
    row = by_id[row_id]
    deps_in = [d for d in row.depends_on if d in by_id]
    next_visiting = (*visiting, row_id)
    lvl = 1 + max((_compute_level(d, by_id, cache, next_visiting) for d in deps_in), default=0)
    cache[row_id] = lvl
    return lvl


def check_idempotency_guard(prior_handoffs: list[ds.PriorHandoff]) -> str | None:
    """Return a string message if a handoff was posted within 30min, else None."""
    if not prior_handoffs:
        return None
    most_recent = prior_handoffs[0]
    created = parse_iso(most_recent.created_at)
    if created is None:
        return None
    age = datetime.now(timezone.utc) - created
    if age < IDEMPOTENCY_GUARD:
        minutes = int(age.total_seconds() // 60)
        return (
            f"A prior handoff was posted {minutes}min ago at {most_recent.url}. "
            f"Refusing to post within the {IDEMPOTENCY_GUARD.seconds // 60}min guard window."
        )
    return None


def parse_iso(text: str) -> datetime | None:
    """Parse a GitHub ISO8601 timestamp into a tz-aware datetime."""
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_surprise(raw: str) -> Surprise:
    """Parse `<category>: <note>` into a Surprise."""
    match = re.match(r"^([^:]+):\s*(.+)$", raw.strip(), flags=re.DOTALL)
    if not match:
        return Surprise(category="Other", note=raw.strip())
    return Surprise(category=match.group(1).strip(), note=match.group(2).strip())


def post_comment(repo: str, issue_number: int, body: str) -> str:
    """Post the rendered comment to the tracking issue, return its html_url."""
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as fh:
        fh.write(body)
        body_path = Path(fh.name)
    try:
        raw = subprocess.run(  # noqa: S603 — argv list, no shell
            [
                "gh",
                "issue",
                "comment",
                str(issue_number),
                "--repo",
                repo,
                "--body-file",
                str(body_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    finally:
        body_path.unlink(missing_ok=True)
    url = raw.stdout.strip().splitlines()[-1] if raw.stdout.strip() else ""
    return url


def save_prompt_locally(prompt_md: str, now_utc: str) -> Path:
    """Save the rendered prompt under `.claude/handoffs/`."""
    DEFAULT_HANDOFFS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = now_utc.replace(" UTC", "").replace(":", "").replace(" ", "-")
    out_path = DEFAULT_HANDOFFS_DIR / f"handoff-{stamp}.md"
    out_path.write_text(prompt_md)
    return out_path


if __name__ == "__main__":
    raise SystemExit(main())
