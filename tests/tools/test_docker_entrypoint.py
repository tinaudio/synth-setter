"""Tests for ``synth_setter.tools.docker_entrypoint`` (click CLI with per-mode spec parsing).

The entrypoint is a click group with five subcommands:
  - idle                → execs ``sleep infinity``
  - passthrough ARGV... → execs ARGV (or errors on empty)
  - generate_dataset    → parses --spec into DatasetSpec, calls run(spec)
  - render_eval         → ClickException "tracked in #410"
  - train               → ClickException "tracked in #409"

Tests use click.testing.CliRunner + function-level monkeypatches (unittest.mock
at module scope where needed). No subprocess calls; everything runs in-process.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from synth_setter.tools import docker_entrypoint as _docker_entrypoint_module


@pytest.fixture()
def _detach_pytest_live_logging_handler() -> Iterator[None]:
    """Detach pytest's live-logging handler for tests whose CliRunner callback hits an error path.

    Why: when ``log_cli=True`` (project default), pytest installs
    ``_LiveLoggingStreamHandler`` on the root logger. Its ``emit()`` opens a
    ``global_and_fixture_disabled`` ctx that suspends pytest's global capture,
    which closes the captured stream that ``CliRunner.isolation()`` puts on
    ``sys.stdout``/``stderr``. A ``logger.error(...)`` inside the click callback
    under test then triggers ``CliRunner.invoke()``'s finally-block ``.getvalue()``
    on a closed buffer → ``ValueError: I/O operation on closed file``. Tracked
    in #730.

    Opt-in via ``@pytest.mark.usefixtures("_detach_pytest_live_logging_handler")``
    on tests that exercise that code path. Tests that don't drive ``logger.error``
    inside ``CliRunner.invoke`` shouldn't pay the indirection cost.
    """
    root = logging.getLogger()
    detached = [
        (i, h)
        for i, h in enumerate(root.handlers)
        if type(h).__name__ == "_LiveLoggingStreamHandler"
    ]
    for _, h in detached:
        root.removeHandler(h)
    try:
        yield
    finally:
        for i, h in sorted(detached):
            root.handlers.insert(i, h)


@pytest.fixture()
def entrypoint() -> ModuleType:
    """Return the loaded docker_entrypoint module."""
    return _docker_entrypoint_module


@pytest.fixture()
def runner() -> CliRunner:
    """Fresh click CliRunner per test."""
    return CliRunner()


def _valid_spec_payload() -> dict[str, Any]:
    """Return a JSON-serializable dict that validates as DatasetSpec."""
    return {
        "task_name": "test-dataset",
        "run_id": "test-dataset-20260328T120000000Z",
        "r2_prefix": "data/test-dataset/test-dataset-20260328T120000000Z/",
        "created_at": "2026-03-28T12:00:00+00:00",
        "git_sha": "a" * 40,
        "is_repo_dirty": False,
        "output_format": "hdf5",
        "train_val_test_sizes": [10000, 0, 0],
        "base_seed": 42,
        "r2_bucket": "intermediate-data",
        "render": {
            "plugin_path": "FakePlugin.vst3",
            "preset_path": "presets/surge-base.vstpreset",
            "param_spec_name": "surge_simple",
            "renderer_version": "1.3.4",
            "sample_rate": 16000,
            "channels": 2,
            "velocity": 100,
            "signal_duration_seconds": 4.0,
            "min_loudness": -55.0,
            "samples_per_render_batch": 32,
            "samples_per_shard": 10000,
            "open_gui_every_render": False,
        },
    }


def _write_spec_file(tmp_path: Path, payload: dict[str, Any] | str | None = None) -> Path:
    """Write ``payload`` (defaults to a valid spec) to a JSON file and return its path."""
    spec_path = tmp_path / "spec.json"
    if payload is None:
        payload = _valid_spec_payload()
    if isinstance(payload, str):
        spec_path.write_text(payload)
    else:
        spec_path.write_text(json.dumps(payload))
    return spec_path


# ---------------------------------------------------------------------------
# cli group — no-subcommand behaviour
# ---------------------------------------------------------------------------


class TestCliGroup:
    """Invoking the click group with no subcommand must fail loudly, not exit 0."""

    def test_no_subcommand_exits_nonzero(self, runner: CliRunner, entrypoint: ModuleType) -> None:
        """`docker_entrypoint.py` with no args exits non-zero (fail-loud).

        Rationale: click's default group behavior is to print help and exit 0 when no
        subcommand is given. For a container entrypoint, that's a silent-success no-op
        if the container is started without a subcommand. We raise a UsageError so
        `docker run <image>` fails rather than quietly doing nothing.
        """
        result = runner.invoke(entrypoint.cli, [])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# idle
# ---------------------------------------------------------------------------


class TestIdle:
    """Idle subcommand execs ``sleep infinity``."""

    def test_idle_execs_sleep_infinity(self, runner: CliRunner, entrypoint: ModuleType) -> None:
        """Idle execs ``sleep`` with argv ``["sleep", "infinity"]``."""
        calls: list[tuple[str, list[str]]] = []

        def fake_execvp(program: str, argv: list[str]) -> None:
            calls.append((program, list(argv)))

        with patch.object(entrypoint.os, "execvp", fake_execvp):
            result = runner.invoke(entrypoint.cli, ["idle"])

        assert result.exit_code == 0, result.output
        assert calls == [("sleep", ["sleep", "infinity"])]

    @pytest.mark.usefixtures("_detach_pytest_live_logging_handler")
    def test_idle_exec_failure_becomes_click_exception(
        self, runner: CliRunner, entrypoint: ModuleType
    ) -> None:
        """Surface ``sleep`` exec failure as a ClickException exit, not a traceback."""

        def fake_execvp(program: str, argv: list[str]) -> None:
            raise FileNotFoundError(2, "No such file or directory", program)

        with patch.object(entrypoint.os, "execvp", fake_execvp):
            result = runner.invoke(entrypoint.cli, ["idle"])

        assert result.exit_code != 0
        # Must NOT bubble as a raw OSError — _exec_or_click_error should convert.
        assert not isinstance(result.exception, FileNotFoundError)
        assert not isinstance(result.exception, OSError)


# ---------------------------------------------------------------------------
# passthrough
# ---------------------------------------------------------------------------


class TestPassthrough:
    """Passthrough subcommand execs its trailing argv, or errors on empty."""

    def test_passthrough_with_args_execs_them(
        self, runner: CliRunner, entrypoint: ModuleType
    ) -> None:
        """Passthrough ARGV...

        execs argv[0] with the full argv list.
        """
        calls: list[tuple[str, list[str]]] = []

        def fake_execvp(program: str, argv: list[str]) -> None:
            calls.append((program, list(argv)))

        with patch.object(entrypoint.os, "execvp", fake_execvp):
            result = runner.invoke(entrypoint.cli, ["passthrough", "echo", "hello"])

        assert result.exit_code == 0, result.output
        assert calls == [("echo", ["echo", "hello"])]

    def test_passthrough_accepts_unknown_flags_without_delimiter(
        self, runner: CliRunner, entrypoint: ModuleType
    ) -> None:
        """Flags in the trailing argv are forwarded without requiring a ``--`` delimiter."""
        calls: list[tuple[str, list[str]]] = []

        def fake_execvp(program: str, argv: list[str]) -> None:
            calls.append((program, list(argv)))

        with patch.object(entrypoint.os, "execvp", fake_execvp):
            result = runner.invoke(
                entrypoint.cli,
                ["passthrough", "rclone", "--checksum", "src", "dst"],
            )

        assert result.exit_code == 0, result.output
        assert calls == [("rclone", ["rclone", "--checksum", "src", "dst"])]

    @pytest.mark.usefixtures("_detach_pytest_live_logging_handler")
    def test_passthrough_exec_failure_becomes_click_exception(
        self, runner: CliRunner, entrypoint: ModuleType
    ) -> None:
        """If the target command can't be exec'd, the failure surfaces as a ClickException exit."""

        def fake_execvp(program: str, argv: list[str]) -> None:
            raise FileNotFoundError(2, "No such file or directory", program)

        with patch.object(entrypoint.os, "execvp", fake_execvp):
            result = runner.invoke(entrypoint.cli, ["passthrough", "nonexistent-cmd"])

        assert result.exit_code != 0
        assert not isinstance(result.exception, FileNotFoundError)
        assert not isinstance(result.exception, OSError)

    def test_passthrough_no_args_exits_nonzero_with_error(
        self, runner: CliRunner, entrypoint: ModuleType
    ) -> None:
        """Passthrough with no trailing argv prints an error and exits non-zero."""
        calls: list[tuple[str, list[str]]] = []

        def fake_execvp(program: str, argv: list[str]) -> None:  # pragma: no cover
            calls.append((program, list(argv)))

        with patch.object(entrypoint.os, "execvp", fake_execvp):
            result = runner.invoke(entrypoint.cli, ["passthrough"])

        assert result.exit_code != 0
        combined = result.output + (result.stderr_bytes.decode() if result.stderr_bytes else "")
        assert combined.strip() != ""
        assert calls == []


# ---------------------------------------------------------------------------
# generate_dataset
# ---------------------------------------------------------------------------


class TestGenerateDataset:
    """generate_dataset parses --spec into DatasetSpec, calls run() in-process."""

    def test_happy_path_parses_spec_and_calls_run(
        self, runner: CliRunner, entrypoint: ModuleType, tmp_path: Path
    ) -> None:
        """Valid --spec JSON is parsed into DatasetSpec and passed to run()."""
        from synth_setter.pipeline.schemas.spec import DatasetSpec

        payload = _valid_spec_payload()
        spec_path = _write_spec_file(tmp_path, payload)

        captured: list[DatasetSpec] = []

        def fake_run(spec: DatasetSpec) -> None:
            captured.append(spec)

        exit_calls: list[int] = []
        with (
            patch.object(entrypoint, "run", fake_run),
            patch.object(entrypoint.os, "_exit", lambda code: exit_calls.append(code)),
        ):
            result = runner.invoke(entrypoint.cli, ["generate_dataset", "--spec", str(spec_path)])

        assert result.exit_code == 0, result.output
        assert len(captured) == 1
        parsed = captured[0]
        assert parsed.run_id == payload["run_id"]
        assert parsed.r2_bucket == payload["r2_bucket"]
        assert parsed.render.samples_per_shard == payload["render"]["samples_per_shard"]
        assert parsed.shards[0].filename == "shard-000000.h5"

    def test_happy_path_calls_os_exit_zero_after_run(
        self, runner: CliRunner, entrypoint: ModuleType, tmp_path: Path
    ) -> None:
        """generate_dataset must call ``os._exit(0)`` after run() returns (#735)."""
        spec_path = _write_spec_file(tmp_path)

        exit_calls: list[int] = []
        with (
            patch.object(entrypoint, "run", lambda _spec: None),
            patch.object(entrypoint.os, "_exit", lambda code: exit_calls.append(code)),
        ):
            result = runner.invoke(entrypoint.cli, ["generate_dataset", "--spec", str(spec_path)])

        assert result.exit_code == 0, result.output
        assert exit_calls == [0], (
            f"expected os._exit(0) after run() returns (#735 workaround), got {exit_calls}"
        )

    @pytest.mark.usefixtures("_detach_pytest_live_logging_handler")
    def test_failed_spec_load_does_not_call_os_exit(
        self, runner: CliRunner, entrypoint: ModuleType, tmp_path: Path
    ) -> None:
        """os._exit(0) must not fire if spec loading fails — run() never executed."""
        missing = tmp_path / "does-not-exist.json"

        exit_calls: list[int] = []
        with patch.object(entrypoint.os, "_exit", lambda code: exit_calls.append(code)):
            result = runner.invoke(entrypoint.cli, ["generate_dataset", "--spec", str(missing)])

        assert result.exit_code != 0
        assert exit_calls == [], (
            f"os._exit must not fire when run() didn't execute, got {exit_calls}"
        )

    def test_missing_spec_flag_exits_two(self, runner: CliRunner, entrypoint: ModuleType) -> None:
        """generate_dataset without --spec exits with click's usage-error code (2)."""
        result = runner.invoke(entrypoint.cli, ["generate_dataset"])
        assert result.exit_code == 2

    @pytest.mark.usefixtures("_detach_pytest_live_logging_handler")
    def test_nonexistent_spec_path_exits_nonzero_via_clickexception(
        self, runner: CliRunner, entrypoint: ModuleType, tmp_path: Path
    ) -> None:
        missing = tmp_path / "does-not-exist.json"
        result = runner.invoke(entrypoint.cli, ["generate_dataset", "--spec", str(missing)])
        assert result.exit_code != 0
        assert not isinstance(result.exception, FileNotFoundError)
        assert not isinstance(result.exception, OSError)

    @pytest.mark.usefixtures("_detach_pytest_live_logging_handler")
    def test_malformed_json_spec_exits_nonzero_without_calling_run(
        self, runner: CliRunner, entrypoint: ModuleType, tmp_path: Path
    ) -> None:
        """Malformed JSON in --spec causes a non-zero exit and does NOT call run()."""
        spec_path = _write_spec_file(tmp_path, "{not-valid-json")

        def fake_run(spec: object) -> None:  # pragma: no cover
            raise AssertionError("run() must not be called for malformed JSON")

        with patch.object(entrypoint, "run", fake_run):
            result = runner.invoke(entrypoint.cli, ["generate_dataset", "--spec", str(spec_path)])

        assert result.exit_code != 0

    @pytest.mark.usefixtures("_detach_pytest_live_logging_handler")
    def test_invalid_spec_shape_exits_nonzero_without_calling_run(
        self, runner: CliRunner, entrypoint: ModuleType, tmp_path: Path
    ) -> None:
        """Valid JSON that doesn't satisfy DatasetSpec fails without calling run()."""
        spec_path = _write_spec_file(tmp_path, {})

        def fake_run(spec: object) -> None:  # pragma: no cover
            raise AssertionError("run() must not be called when the spec fails validation")

        with patch.object(entrypoint, "run", fake_run):
            result = runner.invoke(entrypoint.cli, ["generate_dataset", "--spec", str(spec_path)])

        assert result.exit_code != 0

    @pytest.mark.usefixtures("_detach_pytest_live_logging_handler")
    def test_binary_spec_file_exits_nonzero_without_calling_run(
        self, runner: CliRunner, entrypoint: ModuleType, tmp_path: Path
    ) -> None:
        """A spec file that can't be decoded as UTF-8 surfaces as a ClickException.

        Read-side failures (UnicodeDecodeError, OSError) must be caught in _parse_spec so they
        produce a clean CLI error rather than a traceback. No handoff to run().
        """
        spec_path = tmp_path / "binary.json"
        spec_path.write_bytes(b"\xff\xfe\x00\x01not-utf-8\x80")

        def fake_run(spec: object) -> None:  # pragma: no cover
            raise AssertionError("run() must not be called when the spec can't be read")

        with patch.object(entrypoint, "run", fake_run):
            result = runner.invoke(entrypoint.cli, ["generate_dataset", "--spec", str(spec_path)])

        assert result.exit_code != 0
        # The decode error must NOT bubble out as a raw exception — _parse_spec
        # must convert it into a ClickException so users see a clean CLI error.
        assert not isinstance(result.exception, UnicodeDecodeError)
        assert not isinstance(result.exception, OSError)


# ---------------------------------------------------------------------------
# render_eval / train — ClickException stubs with issue pointers
# ---------------------------------------------------------------------------


class TestRenderEval:
    """render_eval subcommand is a deliberate placeholder pointing at #410."""

    @pytest.mark.usefixtures("_detach_pytest_live_logging_handler")
    def test_render_eval_fails_loudly_with_issue_pointer(
        self, runner: CliRunner, entrypoint: ModuleType, tmp_path: Path
    ) -> None:
        """render_eval --spec <path> fails cleanly mentioning #410, no traceback.

        Uses ClickException (not NotImplementedError) so click's driver prints a clean
        ``Error: ...`` line and exits non-zero without dumping a Python traceback into
        container logs. We verify the behavior by calling the command's callback
        directly (bypassing click's stderr-capture indirection) and asserting the
        exception type + message.
        """
        import click

        spec_path = _write_spec_file(tmp_path)

        def fake_run(spec: object) -> None:  # pragma: no cover
            raise AssertionError("run() must not be called in render_eval")

        # End-to-end behavior: exits non-zero, does not bubble NotImplementedError.
        with patch.object(entrypoint, "run", fake_run):
            result = runner.invoke(entrypoint.cli, ["render_eval", "--spec", str(spec_path)])
        assert result.exit_code != 0
        assert not isinstance(result.exception, NotImplementedError)

        # Message-content check: invoke the callback directly to inspect the raised
        # ClickException without click's stderr indirection.
        with pytest.raises(click.ClickException) as exc_info:
            entrypoint.render_eval.callback(spec_path)
        assert "#410" in exc_info.value.message


class TestTrain:
    """Train subcommand is a deliberate placeholder pointing at #409."""

    @pytest.mark.usefixtures("_detach_pytest_live_logging_handler")
    def test_train_fails_loudly_with_issue_pointer(
        self, runner: CliRunner, entrypoint: ModuleType, tmp_path: Path
    ) -> None:
        """Train --spec <path> fails cleanly mentioning #409, no traceback.

        Same shape as the render_eval test: end-to-end check via CliRunner for the
        exit semantics, direct callback invocation for the message content.
        """
        import click

        spec_path = _write_spec_file(tmp_path)

        def fake_run(spec: object) -> None:  # pragma: no cover
            raise AssertionError("run() must not be called in train")

        with patch.object(entrypoint, "run", fake_run):
            result = runner.invoke(entrypoint.cli, ["train", "--spec", str(spec_path)])
        assert result.exit_code != 0
        assert not isinstance(result.exception, NotImplementedError)

        with pytest.raises(click.ClickException) as exc_info:
            entrypoint.train.callback(spec_path)
        assert "#409" in exc_info.value.message
