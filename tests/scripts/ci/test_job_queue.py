"""Tests for scripts/ci/job_queue.py — line-by-line command queueing.

The CLI shells out to the pueue binary, so the tests pin the public typed API
(parse_command_file, build_pueue_add_args, ensure_group, enqueue_all) and use
fake subprocess runners injected as callables — no real `pueue` process is ever
spawned. The contract under test is "given this input, what backend args are
emitted, in what order?"
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from scripts.ci.job_queue import (
    DEFAULT_GROUP,
    build_pueue_add_args,
    enqueue_all,
    ensure_daemon_running,
    ensure_group,
    main,
    parse_command_file,
)


def _no_sleep(seconds: float) -> None:
    """Sleep stub for tests — drops the requested duration to keep retry loops instant.

    :param seconds: ignored.
    """
    del seconds


# ---------------------------------------------------------------------------
# parse_command_file — file parsing
# ---------------------------------------------------------------------------


def test_parse_command_file_returns_each_nonblank_line(tmp_path: Path) -> None:
    """Each non-blank source line round-trips into the returned list.

    :param tmp_path: pytest tmp dir fixture.
    """
    file = tmp_path / "cmds.txt"
    file.write_text("python a.py\npython b.py\npython c.py\n")
    assert parse_command_file(file) == ["python a.py", "python b.py", "python c.py"]


def test_parse_command_file_skips_blank_lines(tmp_path: Path) -> None:
    """Blank / whitespace-only lines are dropped silently.

    :param tmp_path: pytest tmp dir fixture.
    """
    file = tmp_path / "cmds.txt"
    file.write_text("python a.py\n\n\npython b.py\n\n")
    assert parse_command_file(file) == ["python a.py", "python b.py"]


def test_parse_command_file_skips_comment_lines(tmp_path: Path) -> None:
    """``#``-prefixed lines (with or without leading whitespace) are dropped.

    :param tmp_path: pytest tmp dir fixture.
    """
    file = tmp_path / "cmds.txt"
    file.write_text("# header comment\npython a.py\n  # indented comment\npython b.py\n")
    assert parse_command_file(file) == ["python a.py", "python b.py"]


def test_parse_command_file_strips_trailing_whitespace(tmp_path: Path) -> None:
    """Trailing spaces/tabs are stripped so the returned command is exec-clean.

    :param tmp_path: pytest tmp dir fixture.
    """
    file = tmp_path / "cmds.txt"
    file.write_text("python a.py   \npython b.py\t\n")
    assert parse_command_file(file) == ["python a.py", "python b.py"]


def test_parse_command_file_raises_on_missing_file(tmp_path: Path) -> None:
    """A missing path surfaces as FileNotFoundError, not a silent empty list.

    :param tmp_path: pytest tmp dir fixture.
    """
    with pytest.raises(FileNotFoundError):
        parse_command_file(tmp_path / "nope.txt")


def test_parse_command_file_returns_empty_list_for_only_comments(tmp_path: Path) -> None:
    """A file with only comments/blanks parses to ``[]``.

    :param tmp_path: pytest tmp dir fixture.
    """
    file = tmp_path / "cmds.txt"
    file.write_text("# comment 1\n# comment 2\n\n")
    assert parse_command_file(file) == []


def test_parse_command_file_strips_leading_whitespace_from_indented_commands(
    tmp_path: Path,
) -> None:
    """Indented commands inside grouped sections are normalized — no leading space leaks into the.

    queued task (else pueue would try to run ``  python x.py`` verbatim).

    :param tmp_path: pytest tmp dir fixture.
    """
    file = tmp_path / "cmds.txt"
    file.write_text("  python a.py\n\t\tpython b.py\n    python c.py --flag=1\n")
    assert parse_command_file(file) == [
        "python a.py",
        "python b.py",
        "python c.py --flag=1",
    ]


# ---------------------------------------------------------------------------
# build_pueue_add_args — pueue CLI arg construction
# ---------------------------------------------------------------------------


def test_build_pueue_add_args_minimal_invocation() -> None:
    """Minimal arg set is ``pueue add --group GROUP -- COMMAND``."""
    args = build_pueue_add_args(
        command="python train.py",
        group=DEFAULT_GROUP,
        working_dir=None,
        label=None,
    )
    assert args == ["pueue", "add", "--group", DEFAULT_GROUP, "--", "python train.py"]


def test_build_pueue_add_args_with_working_dir(tmp_path: Path) -> None:
    """A working_dir is forwarded as ``--working-directory <path>``.

    :param tmp_path: pytest tmp dir fixture.
    """
    args = build_pueue_add_args(
        command="python train.py",
        group="train",
        working_dir=tmp_path,
        label=None,
    )
    assert "--working-directory" in args
    assert str(tmp_path) in args


def test_build_pueue_add_args_with_label() -> None:
    """A non-empty label is forwarded as ``--label <label>``."""
    args = build_pueue_add_args(
        command="python train.py",
        group="train",
        working_dir=None,
        label="exp-42",
    )
    assert "--label" in args
    assert "exp-42" in args


def test_build_pueue_add_args_command_is_last_positional() -> None:
    """The command is the single positional after ``--``."""
    args = build_pueue_add_args(
        command="echo hi",
        group="train",
        working_dir=None,
        label="x",
    )
    sentinel_idx = args.index("--")
    assert args[sentinel_idx + 1 :] == ["echo hi"]


# ---------------------------------------------------------------------------
# ensure_daemon_running / ensure_group — backend orchestration
# ---------------------------------------------------------------------------


class FakeRunner:
    """Capture subprocess invocations and return scripted exit codes.

    ``results`` maps a tuple-of-argv to ``(returncode, stdout)``. Unmatched
    argv defaults to ``(0, "")``. When ``check=True`` is passed and the
    scripted returncode is non-zero, raises ``CalledProcessError`` to mirror
    ``subprocess.run``'s contract.
    """

    def __init__(self, results: dict[tuple[str, ...], tuple[int, str]] | None = None) -> None:
        """Build a runner with optional scripted responses.

        :param results: argv→(returncode, stdout) overrides. Unset keys default
            to ``(0, "")``.
        """
        self.results = results or {}
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Record the invocation and return the scripted result.

        :param args: argv that production code would pass to ``subprocess.run``.
        :param \\*\\*kwargs: subprocess.run kwargs; only ``check`` is honored here
            (mirrors ``subprocess.run`` raising on non-zero when ``check=True``).
        :returns: a ``CompletedProcess`` with the scripted returncode and stdout.
        :rtype: subprocess.CompletedProcess[str]
        :raises subprocess.CalledProcessError: when the scripted returncode is
            non-zero and ``check=True`` was passed.
        """
        self.calls.append(list(args))
        key = tuple(args)
        rc, out = self.results.get(key, (0, ""))
        if rc != 0 and kwargs.get("check"):
            raise subprocess.CalledProcessError(returncode=rc, cmd=args, output=out, stderr="")
        return subprocess.CompletedProcess(args=args, returncode=rc, stdout=out, stderr="")


def test_ensure_daemon_running_noop_when_status_succeeds() -> None:
    """If ``pueue status`` returns 0, the daemon is already up — don't run ``pueued -d``."""
    runner = FakeRunner(results={("pueue", "status"): (0, "")})
    ensure_daemon_running(runner, sleeper=_no_sleep)
    cmds = [tuple(c) for c in runner.calls]
    assert ("pueued", "-d") not in cmds


class _StatefulPueueStatusRunner:
    """FakeRunner variant where ``pueue status`` flips from failing → succeeding after ``pueued.

    -d``.

    Models the real race: status fails before the daemon binds its socket,
    succeeds afterwards. Unlike ``FakeRunner`` this runner does NOT honor
    ``check`` — every call path in ``ensure_daemon_running`` that this runner
    sees uses ``check=False`` for the probe and ``check=True`` only for
    ``pueued -d`` (which always succeeds in this fake), so there's no path
    where we'd need to raise.
    """

    def __init__(self, succeeds_after_calls: int) -> None:
        """Track when post-daemonize status probes start succeeding.

        :param succeeds_after_calls: number of post-``pueued -d`` ``pueue status``
            calls that must fail before status starts returning 0.
        """
        self._post_daemon_status_calls = 0
        self._daemon_started = False
        self._succeeds_after = succeeds_after_calls
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Capture argv, then return a CompletedProcess whose rc reflects daemon state.

        :param args: argv passed by production code.
        :param \\*\\*kwargs: subprocess.run kwargs; ignored by this runner (see
            class docstring for why).
        :returns: ``CompletedProcess`` with rc=0 once the daemon is "bound",
            else rc=1 for ``pueue status``. ``pueued -d`` always returns rc=0
            and flips the internal daemon-started flag.
        :rtype: subprocess.CompletedProcess[str]
        """
        del kwargs
        self.calls.append(list(args))
        if args == ["pueued", "-d"]:
            self._daemon_started = True
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        if args == ["pueue", "status"]:
            if not self._daemon_started:
                return subprocess.CompletedProcess(args=args, returncode=1, stdout="", stderr="")
            self._post_daemon_status_calls += 1
            rc = 0 if self._post_daemon_status_calls > self._succeeds_after else 1
            return subprocess.CompletedProcess(args=args, returncode=rc, stdout="", stderr="")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")


def test_ensure_daemon_running_starts_daemon_when_status_fails() -> None:
    """If ``pueue status`` returns non-zero, run ``pueued -d`` to daemonize before continuing."""
    runner = _StatefulPueueStatusRunner(succeeds_after_calls=0)
    ensure_daemon_running(runner, sleeper=_no_sleep)
    cmds = [tuple(c) for c in runner.calls]
    assert ("pueued", "-d") in cmds


def test_ensure_daemon_running_retries_until_daemon_binds_socket() -> None:
    """Status fails for the first few post-daemonize probes, then succeeds — must not raise."""
    runner = _StatefulPueueStatusRunner(succeeds_after_calls=3)
    sleeps: list[float] = []
    ensure_daemon_running(runner, sleeper=sleeps.append)
    cmds = [tuple(c) for c in runner.calls]
    # 1 pre-daemon status probe + 1 pueued -d + 4 post-daemon status probes (3 fail, 1 ok)
    assert cmds.count(("pueue", "status")) == 5
    assert cmds.count(("pueued", "-d")) == 1
    # Slept exactly 3 times (once after each failed post-daemonize probe).
    assert len(sleeps) == 3


def test_ensure_daemon_running_raises_when_daemon_never_binds() -> None:
    """If ``pueue status`` keeps failing past the budget, raise instead of silently continuing."""
    runner = _StatefulPueueStatusRunner(succeeds_after_calls=10_000)
    with pytest.raises(RuntimeError, match="never became reachable"):
        ensure_daemon_running(runner, sleeper=_no_sleep)


def test_ensure_group_creates_missing_group_and_sets_parallel() -> None:
    """A group not in ``pueue group`` output is created, then its parallelism is set."""
    runner = FakeRunner(
        results={("pueue", "group"): (0, "Group 'default'\n  Parallel: 1\n")},
    )
    ensure_group("train", parallel=4, runner=runner)
    cmds = [tuple(c) for c in runner.calls]
    assert ("pueue", "group", "add", "train") in cmds
    assert ("pueue", "parallel", "4", "--group", "train") in cmds


def test_ensure_group_skips_create_when_group_exists() -> None:
    """A group already in ``pueue group`` output is not recreated."""
    runner = FakeRunner(
        results={("pueue", "group"): (0, "Group 'default'\nGroup 'train'\n")},
    )
    ensure_group("train", parallel=2, runner=runner)
    cmds = [tuple(c) for c in runner.calls]
    assert ("pueue", "group", "add", "train") not in cmds
    assert ("pueue", "parallel", "2", "--group", "train") in cmds


# ---------------------------------------------------------------------------
# enqueue_all — full queue submission
# ---------------------------------------------------------------------------


def test_enqueue_all_emits_one_pueue_add_per_command() -> None:
    """Each input command produces exactly one ``pueue add``, preserving order."""
    runner = FakeRunner()
    enqueue_all(
        commands=["python a.py", "python b.py", "python c.py"],
        group="train",
        working_dir=None,
        label_prefix="exp",
        runner=runner,
    )
    add_calls = [c for c in runner.calls if c[:2] == ["pueue", "add"]]
    assert len(add_calls) == 3
    assert add_calls[0][-1] == "python a.py"
    assert add_calls[1][-1] == "python b.py"
    assert add_calls[2][-1] == "python c.py"


def test_enqueue_all_applies_label_prefix_with_index() -> None:
    """A non-empty label_prefix yields per-task labels of the form ``<prefix>-<idx>``."""
    runner = FakeRunner()
    enqueue_all(
        commands=["python a.py", "python b.py"],
        group="train",
        working_dir=None,
        label_prefix="run",
        runner=runner,
    )
    add_calls = [c for c in runner.calls if c[:2] == ["pueue", "add"]]
    labels = [c[c.index("--label") + 1] for c in add_calls]
    assert labels == ["run-0", "run-1"]


def test_enqueue_all_omits_label_when_prefix_empty() -> None:
    """An empty label_prefix means no ``--label`` flag is emitted."""
    runner = FakeRunner()
    enqueue_all(
        commands=["python a.py"],
        group="train",
        working_dir=None,
        label_prefix="",
        runner=runner,
    )
    add_calls = [c for c in runner.calls if c[:2] == ["pueue", "add"]]
    assert "--label" not in add_calls[0]


def test_enqueue_all_propagates_working_dir(tmp_path: Path) -> None:
    """A working_dir flows through to each ``pueue add`` call as ``--working-directory``.

    :param tmp_path: pytest tmp dir fixture.
    """
    runner = FakeRunner()
    enqueue_all(
        commands=["python a.py"],
        group="train",
        working_dir=tmp_path,
        label_prefix="",
        runner=runner,
    )
    add_call = next(c for c in runner.calls if c[:2] == ["pueue", "add"])
    assert "--working-directory" in add_call
    assert str(tmp_path) in add_call


def test_enqueue_all_raises_on_pueue_add_failure() -> None:
    """A non-zero ``pueue add`` aborts the run and skips the remaining commands."""
    runner = FakeRunner(
        results={
            ("pueue", "add", "--group", "train", "--", "python b.py"): (1, ""),
        },
    )
    with pytest.raises(subprocess.CalledProcessError):
        enqueue_all(
            commands=["python a.py", "python b.py", "python c.py"],
            group="train",
            working_dir=None,
            label_prefix="",
            runner=runner,
        )
    add_calls = [c for c in runner.calls if c[:2] == ["pueue", "add"]]
    assert len(add_calls) == 2


# ---------------------------------------------------------------------------
# main (click CLI) — end-to-end with --dry-run
# ---------------------------------------------------------------------------


def test_main_dry_run_prints_one_pueue_add_per_line(tmp_path: Path) -> None:
    """--dry-run emits one ``pueue add`` per non-comment line of the input file.

    :param tmp_path: pytest tmp dir fixture.
    """
    file = tmp_path / "cmds.txt"
    file.write_text("# header\npython a.py\n\npython b.py\n")
    result = CliRunner().invoke(
        main,
        [str(file), "--dry-run", "--group", "train", "--parallel", "3"],
    )
    assert result.exit_code == 0, result.output
    assert result.output.count("pueue add") == 2
    assert "python a.py" in result.output
    assert "python b.py" in result.output


def test_main_dry_run_quotes_command_arg_to_preserve_argv_boundaries(tmp_path: Path) -> None:
    """The user command (last positional after ``--``) is a SINGLE shlex-quoted token.

    Without ``shlex.join``, ``" ".join(args)`` would print
    ``pueue add ... -- python src/synth_setter/cli/train.py experiment=surge/full_ffn`` which
    reads as four separate trailing args; with ``shlex.join`` the command is
    quoted so the dry-run line is faithful to the real argv.

    :param tmp_path: pytest tmp dir fixture.
    """
    file = tmp_path / "cmds.txt"
    file.write_text("python src/synth_setter/cli/train.py experiment=surge/full_ffn\n")
    result = CliRunner().invoke(main, [str(file), "--dry-run", "--group", "train"])
    assert result.exit_code == 0, result.output
    # shlex.join quotes whenever the arg contains shell-significant chars (incl. space).
    assert "'python src/synth_setter/cli/train.py experiment=surge/full_ffn'" in result.output


def test_main_dry_run_does_not_invoke_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--dry-run must not call subprocess.run (no daemon, no pueue invocation).

    :param tmp_path: pytest tmp dir fixture.
    :param monkeypatch: pytest monkeypatch fixture used to install a trip-wire.
    """
    file = tmp_path / "cmds.txt"
    file.write_text("python a.py\n")

    def explode(*_args: Any, **_kwargs: Any) -> None:
        """Trip-wire: any subprocess.run call under --dry-run is a regression.

        :param \\*_args: ignored.
        :param \\*\\*_kwargs: ignored.
        :raises AssertionError: always — invocation indicates a regression.
        """
        del _args, _kwargs
        raise AssertionError("subprocess.run must not be called under --dry-run")

    monkeypatch.setattr(subprocess, "run", explode)
    result = CliRunner().invoke(main, [str(file), "--dry-run", "--no-start-daemon"])
    assert result.exit_code == 0, result.output


def test_main_errors_when_file_is_empty(tmp_path: Path) -> None:
    """A file with no real commands is a usage error — no daemon side-effects.

    :param tmp_path: pytest tmp dir fixture.
    """
    file = tmp_path / "cmds.txt"
    file.write_text("# only comments\n\n")
    result = CliRunner().invoke(main, [str(file), "--dry-run", "--no-start-daemon"])
    assert result.exit_code != 0
    assert "no commands" in result.output.lower()
