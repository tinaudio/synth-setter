"""Tests for scripts/docker_entrypoint.py — click-based CLI with per-mode spec parsing.

The entrypoint is a click group with five subcommands:
  - idle                → execs ``sleep infinity``
  - passthrough ARGV... → execs ARGV (or errors on empty)
  - generate_dataset    → parses --spec into DatasetPipelineSpec, calls run(spec)
  - render_eval         → ClickException "tracked in #410"
  - train               → ClickException "tracked in #409"

Tests use click.testing.CliRunner + function-level monkeypatches (unittest.mock
at module scope where needed). No subprocess calls; everything runs in-process.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINT_PATH = REPO_ROOT / "scripts" / "docker_entrypoint.py"


def _load_entrypoint_module() -> ModuleType:
    """Import scripts/docker_entrypoint.py as a module for in-process testing."""
    module_name = "_docker_entrypoint_under_test"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, ENTRYPOINT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def entrypoint() -> ModuleType:
    """The loaded docker_entrypoint module."""
    return _load_entrypoint_module()


@pytest.fixture()
def runner() -> CliRunner:
    """Fresh click CliRunner per test."""
    return CliRunner()


def _valid_spec_payload() -> dict[str, Any]:
    """Return a JSON-serializable dict that validates as DatasetPipelineSpec."""
    return {
        "run_id": "test-dataset-20260328T120000Z",
        "r2_prefix": "data/test-dataset/test-dataset-20260328T120000Z/",
        "created_at": "2026-03-28T12:00:00Z",
        "code_version": "a" * 40,
        "is_repo_dirty": False,
        "param_spec": "surge_simple",
        "renderer_version": "1.3.4",
        "output_format": "hdf5",
        "sample_rate": 16000,
        "shard_size": 10000,
        "base_seed": 42,
        "num_params": 92,
        "r2_bucket": "intermediate-data",
        "splits": {"train": 1, "val": 0, "test": 0},
        "plugin_path": "FakePlugin.vst3",
        "preset_path": "presets/surge-base.vstpreset",
        "channels": 2,
        "velocity": 100,
        "signal_duration_seconds": 4.0,
        "min_loudness": -55.0,
        "sample_batch_size": 32,
        "shards": [{"shard_id": 0, "filename": "shard-000000.h5", "seed": 42}],
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

    def test_idle_exec_failure_becomes_click_exception(
        self, runner: CliRunner, entrypoint: ModuleType
    ) -> None:
        """If ``sleep`` can't be exec'd, the failure surfaces as a ClickException exit, not a
        traceback."""

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
    """generate_dataset parses --spec into DatasetPipelineSpec, calls run() in-process."""

    def test_happy_path_parses_spec_and_calls_run(
        self, runner: CliRunner, entrypoint: ModuleType, tmp_path: Path
    ) -> None:
        """Valid --spec JSON is parsed into DatasetPipelineSpec and passed to run()."""
        from pipeline.schemas.spec import DatasetPipelineSpec

        payload = _valid_spec_payload()
        spec_path = _write_spec_file(tmp_path, payload)

        captured: list[DatasetPipelineSpec] = []

        def fake_run(spec: DatasetPipelineSpec) -> None:
            captured.append(spec)

        with patch.object(entrypoint, "run", fake_run):
            result = runner.invoke(entrypoint.cli, ["generate_dataset", "--spec", str(spec_path)])

        assert result.exit_code == 0, result.output
        assert len(captured) == 1
        parsed = captured[0]
        assert parsed.run_id == payload["run_id"]
        assert parsed.r2_bucket == payload["r2_bucket"]
        assert parsed.shard_size == payload["shard_size"]
        assert parsed.shards[0].filename == "shard-000000.h5"

    def test_missing_spec_flag_exits_two(self, runner: CliRunner, entrypoint: ModuleType) -> None:
        """generate_dataset without --spec exits with click's usage-error code (2)."""
        result = runner.invoke(entrypoint.cli, ["generate_dataset"])
        assert result.exit_code == 2

    def test_nonexistent_spec_path_exits_two(
        self, runner: CliRunner, entrypoint: ModuleType, tmp_path: Path
    ) -> None:
        """--spec pointing at a non-existent file exits 2 (click ``exists=True`` enforcement)."""
        missing = tmp_path / "does-not-exist.json"
        result = runner.invoke(entrypoint.cli, ["generate_dataset", "--spec", str(missing)])
        assert result.exit_code == 2

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

    def test_invalid_spec_shape_exits_nonzero_without_calling_run(
        self, runner: CliRunner, entrypoint: ModuleType, tmp_path: Path
    ) -> None:
        """Valid JSON that doesn't satisfy DatasetPipelineSpec fails without calling run()."""
        spec_path = _write_spec_file(tmp_path, {})

        def fake_run(spec: object) -> None:  # pragma: no cover
            raise AssertionError("run() must not be called when the spec fails validation")

        with patch.object(entrypoint, "run", fake_run):
            result = runner.invoke(entrypoint.cli, ["generate_dataset", "--spec", str(spec_path)])

        assert result.exit_code != 0

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


# ---------------------------------------------------------------------------
# Structural invariants
# ---------------------------------------------------------------------------


class TestModeSpecTypesMapping:
    """Invariants on the _MODE_SPEC_TYPES module-level mapping."""

    def test_every_key_is_a_click_command_in_the_group(self, entrypoint: ModuleType) -> None:
        """Every key in _MODE_SPEC_TYPES names a click command registered on ``cli``."""
        for mode in entrypoint._MODE_SPEC_TYPES:
            assert mode in entrypoint.cli.commands, (
                f"_MODE_SPEC_TYPES key {mode!r} has no corresponding click command"
            )

    def test_every_value_is_a_pydantic_basemodel_subclass(self, entrypoint: ModuleType) -> None:
        """Every value in _MODE_SPEC_TYPES is a concrete subclass of pydantic.BaseModel."""
        for mode, spec_type in entrypoint._MODE_SPEC_TYPES.items():
            assert isinstance(spec_type, type), (
                f"_MODE_SPEC_TYPES[{mode!r}] must be a class, got {spec_type!r}"
            )
            assert issubclass(spec_type, BaseModel), (
                f"_MODE_SPEC_TYPES[{mode!r}] must subclass pydantic.BaseModel"
            )

    def test_render_eval_and_train_are_not_registered(self, entrypoint: ModuleType) -> None:
        """render_eval and train are deliberately absent — they have no spec type yet."""
        assert "render_eval" not in entrypoint._MODE_SPEC_TYPES
        assert "train" not in entrypoint._MODE_SPEC_TYPES
