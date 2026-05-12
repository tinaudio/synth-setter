#!/usr/bin/env python
"""Queue shell commands from a text file as background jobs.

Reads a path to a UTF-8 text file containing one shell command per line and
submits each line as a separate background job in a configurable group.
Blank lines and lines whose first non-whitespace character is ``#`` are
skipped.

The current backend is `pueue <https://github.com/Nukesor/pueue>`_ — it is
treated as an implementation detail and may be swapped without changing this
CLI's user-facing flags or output. References to ``pueue`` below describe
the current backend's behavior.

Example file::

    # train sweeps
    python src/train.py experiment=surge/full_ffn
    python src/train.py experiment=surge/full_ffn data.batch_size=64

Example invocation::

    python scripts/job_queue.py sweeps/train.txt \\
        --group train --parallel 2 --label-prefix sweep

Each line becomes one ``pueue add`` invocation. Pueue runs the command through
a shell, so all shell syntax (``=``, ``&&``, env-var expansion) works as
written. Submission is fail-fast: a non-zero ``pueue add`` aborts the run and
the remaining commands are not enqueued.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import click

DEFAULT_GROUP = "default"
DEFAULT_PARALLEL = 1

# pueued -d returns as soon as the daemon is forked, but the daemon needs a
# moment to bind its socket. The next `pueue` call then races with that bind.
# We re-probe `pueue status` with linear backoff until either it succeeds or
# the budget is exhausted — same approach as the validate workflow's readiness
# loop, kept here so callers don't need to wrap.
DAEMON_READY_RETRIES = 20
DAEMON_READY_SLEEP_SECONDS = 0.25

# Type alias for the subprocess-runner the orchestration helpers depend on.
# Tests inject a fake; production passes ``subprocess.run``.
RunnerFn = Callable[..., "subprocess.CompletedProcess[str]"]


def parse_command_file(path: Path) -> list[str]:
    """Return one entry per non-comment, non-blank line of ``path``.

    Each returned string is a single shell command line, ready to be passed
    verbatim to ``pueue add``. Both leading and trailing whitespace are
    stripped — sweep files commonly indent commands inside grouped sections
    (e.g. ``  python x.py``) and we want the queued task to run as
    ``python x.py``, not ``  python x.py``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    text = path.read_text(encoding="utf-8")
    commands: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            continue
        commands.append(line.lstrip())
    return commands


def build_pueue_add_args(
    command: str,
    group: str,
    working_dir: Path | None,
    label: str | None,
) -> list[str]:
    """Return the argv for ``pueue add`` that enqueues ``command``.

    The command line is passed as a single positional after ``--`` so pueue
    runs it through its own shell wrapper and arbitrary shell syntax (``=``,
    ``&&``, ``$VAR``) is preserved without us having to shlex-split.
    """
    args = ["pueue", "add", "--group", group]
    if working_dir is not None:
        args.extend(["--working-directory", str(working_dir)])
    if label:
        args.extend(["--label", label])
    args.append("--")
    args.append(command)
    return args


def _run(runner: RunnerFn, args: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
    """Wrap ``runner`` to set capture/text defaults consistently across callsites."""
    return runner(args, check=check, capture_output=True, text=True)


def ensure_daemon_running(
    runner: RunnerFn,
    sleeper: Callable[[float], None] = time.sleep,
) -> None:
    """Start ``pueued -d`` if no daemon is reachable, then wait for it to bind.

    Detection uses ``pueue status``: a non-zero exit means the daemon is not
    listening (or not installed correctly), and we attempt a daemonize. After
    spawning we re-probe ``pueue status`` with bounded retries to avoid the
    race where the daemon has been forked but hasn't bound its socket yet —
    the next ``pueue group`` would otherwise fail intermittently.

    Raises:
        RuntimeError: If the daemon never becomes reachable inside the retry
            budget after a fresh start.
    """
    probe = runner(["pueue", "status"], check=False, capture_output=True, text=True)
    if probe.returncode == 0:
        return
    _run(runner, ["pueued", "-d"], check=True)
    for _ in range(DAEMON_READY_RETRIES):
        probe = runner(["pueue", "status"], check=False, capture_output=True, text=True)
        if probe.returncode == 0:
            return
        sleeper(DAEMON_READY_SLEEP_SECONDS)
    raise RuntimeError(
        f"pueued was started but never became reachable within "
        f"{DAEMON_READY_RETRIES * DAEMON_READY_SLEEP_SECONDS:.1f}s"
    )


def ensure_group(group: str, parallel: int, runner: RunnerFn) -> None:
    """Create ``group`` if absent and set its parallel-slot count to ``parallel``."""
    listing = _run(runner, ["pueue", "group"], check=True)
    if f"'{group}'" not in listing.stdout:
        _run(runner, ["pueue", "group", "add", group], check=True)
    _run(runner, ["pueue", "parallel", str(parallel), "--group", group], check=True)


def enqueue_all(
    commands: list[str],
    group: str,
    working_dir: Path | None,
    label_prefix: str,
    runner: RunnerFn,
) -> None:
    """Submit each command via ``pueue add``; fail-fast on first error.

    When ``label_prefix`` is non-empty, each task gets a label of the form
    ``{prefix}-{index}`` so the resulting queue is easy to filter (``pueue
    status -g GROUP``) and clean up (``pueue clean -g GROUP``).
    """
    for idx, command in enumerate(commands):
        label = f"{label_prefix}-{idx}" if label_prefix else None
        args = build_pueue_add_args(command, group, working_dir, label)
        _run(runner, args, check=True)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "command_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--group",
    default=DEFAULT_GROUP,
    show_default=True,
    help="Job-queue group to enqueue into. Created if missing.",
)
@click.option(
    "--parallel",
    type=click.IntRange(min=1),
    default=DEFAULT_PARALLEL,
    show_default=True,
    help="Concurrent task slots for --group.",
)
@click.option(
    "--working-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Directory each job runs in. Defaults to the backend's own CWD.",
)
@click.option(
    "--label-prefix",
    default="",
    show_default=False,
    help="Prefix for per-task labels (each label becomes '<prefix>-<index>'). Empty = no labels.",
)
@click.option(
    "--start-daemon/--no-start-daemon",
    default=True,
    show_default=True,
    help="Start the backend daemon (`pueued -d`) if it isn't already reachable.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the backend commands that would run, without invoking the backend.",
)
def main(
    command_file: Path,
    group: str,
    parallel: int,
    working_dir: Path | None,
    label_prefix: str,
    start_daemon: bool,
    dry_run: bool,
) -> None:
    """Queue each line of COMMAND_FILE as a separate background job."""
    commands = parse_command_file(command_file)
    if not commands:
        raise click.UsageError(f"no commands found in {command_file} (only blanks/comments?)")

    click.echo(f"Loaded {len(commands)} command(s) from {command_file}")
    click.echo(f"Group: {group} (parallel={parallel})")

    if dry_run:
        for idx, command in enumerate(commands):
            label = f"{label_prefix}-{idx}" if label_prefix else None
            args = build_pueue_add_args(command, group, working_dir, label)
            click.echo(" ".join(args))
        return

    runner: RunnerFn = subprocess.run  # noqa: S603 — argv built from validated CLI inputs

    if start_daemon:
        ensure_daemon_running(runner)
    ensure_group(group, parallel, runner)

    try:
        enqueue_all(commands, group, working_dir, label_prefix, runner)
    except subprocess.CalledProcessError as exc:
        msg = exc.stderr.strip() if exc.stderr else f"exit {exc.returncode}"
        raise click.ClickException(f"job submission failed: {msg}") from exc

    click.echo(f"Enqueued {len(commands)} job(s) into group '{group}'.")


if __name__ == "__main__":
    main()  # pragma: no cover
