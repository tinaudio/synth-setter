"""Tests for synth_setter.cli.generate_dataset — spec-driven run.

The entrypoint's public surface:

- ``main()``: launcher-side orchestrator. Composes the cfg, writes the local
  ``input_spec.json`` mirror, runs ``r2_io.ensure_r2_env_loaded`` (dotenv +
  auth ping), uploads the canonical spec via ``spec_io.upload_spec``, then
  either calls ``run(spec)`` inline (local-run) or dispatches to a SkyPilot
  worker pod.
- ``run(spec)``: per-rank renderer. For each owned shard in ``spec.shards``,
  shells out to ``generate_vst_dataset.py``, uploads the shard to R2 at
  ``r2:{bucket}/{prefix}/``, and unlinks the local file. No longer uploads
  the spec — ``main()`` does that once on the launcher host.

``TestRun`` tests share a ``patched_subprocess`` fixture that pulls in
``fake_r2_remote`` (see ``tests/pipeline/conftest.py``) and patches
``subprocess.check_call`` with the ``_materialize_or_passthrough_rclone``
dispatcher: renderer calls write the expected empty shard file (mirroring the
contract of ``generate_vst_dataset.py``); rclone calls fall through to the
real binary against the local-typed remote. Orchestration assertions
(call counts, render/upload ordering, partitioning) are state-based —
asserting on materialized objects under ``fake_r2_remote`` — rather than
introspecting an ``_rclone_copy`` mock. ``main()`` tests still stub
``write_spec_locally`` / ``upload_spec`` / ``ensure_r2_env_loaded`` to keep
the cfg-composition surface isolated from R2.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from synth_setter.cli.generate_dataset import (
    VST_HEADLESS_WRAPPER,
    build_generate_args,
    run,
)
from synth_setter.pipeline.schemas.spec import DatasetSpec, RenderConfig
from tests.helpers.subprocess_args import find_script_index

# Reusable VST3 bundle with a real Contents/moduleinfo.json so
# extract_renderer_version (called by run) returns a deterministic version
# without loading any .so via pedalboard. Version inside is "1.0.0-test" — the
# specs built in this file pin renderer_version to the same string so the
# constraint check passes.
TEST_PLUGIN_VST3 = Path(__file__).resolve().parent.parent / "fixtures" / "TestPlugin.vst3"
TEST_PLUGIN_VERSION = "1.0.0-test"


def _materialize_shard(args: list[str]) -> int:
    """subprocess.check_call side effect that writes the expected shard file.

    Mirrors the production contract: generate_vst_dataset.py exits 0 only after writing the
    HDF5 to its output path. Tests that don't supply this side effect would trip the
    `shard_path.is_file()` check in `_render_and_upload_shard`.
    """
    script_idx = find_script_index(args)
    output_file = Path(args[script_idx + 1])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(b"")
    return 0


# Captured at import time so the rclone-passthrough side-effect below can
# call the real subprocess.check_call without recursing back through any
# patch that targets the same symbol the production code uses.
_REAL_CHECK_CALL = subprocess.check_call


def _materialize_or_passthrough_rclone(args: list[str]) -> int:
    """Dispatch on the first argv element: rclone calls fall through to real subprocess.

    Every state-based ``TestRun`` test patches the same ``subprocess.check_call``
    symbol the renderer AND the rclone shard upload both go through, so this
    side-effect distinguishes them: renderer calls write the expected shard
    file (as ``_materialize_shard`` does); rclone calls invoke the real binary
    (via ``_REAL_CHECK_CALL``) so the upload actually lands a file on the
    fake-local remote.

    :param args: argv list passed to ``subprocess.check_call``.
    :returns: 0 on renderer simulation; rclone's exit code on the real subprocess.
    """
    if args and args[0] == "rclone":
        return _REAL_CHECK_CALL(args)  # noqa: S603 — test-only passthrough
    return _materialize_shard(args)


def _renderer_argv_lists(mock: MagicMock) -> list[list[str]]:
    """Return argv lists from non-rclone calls recorded by a patched ``check_call``.

    The dispatcher routes both renderer and rclone invocations through one
    ``subprocess.check_call`` mock, so tests that want to introspect just the
    renderer args (script path, flag set, headless wrapper) filter the
    interleaved call list through this helper.

    :param mock: A patched ``subprocess.check_call`` mock.
    :returns: argv lists from invocations whose first element is not ``"rclone"``.
    """
    return [
        call.args[0]
        for call in mock.call_args_list
        if not (call.args and call.args[0] and call.args[0][0] == "rclone")
    ]


def _base_spec_kwargs(tmp_path: Path, **overrides: object) -> dict[str, object]:
    """Return valid DatasetSpec kwargs for direct construction."""
    kwargs: dict[str, object] = {
        "task_name": "test-dataset",
        "run_id": "test-dataset-20260328T120000000Z",
        "created_at": datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc),
        "git_sha": "a" * 40,
        "is_repo_dirty": False,
        "output_format": "hdf5",
        "train_val_test_sizes": [10000, 0, 0],
        "base_seed": 42,
        "r2": {
            "bucket": "intermediate-data",
            "prefix": "data/test-dataset/test-dataset-20260328T120000000Z/",
        },
        "render": {
            "plugin_path": str(TEST_PLUGIN_VST3),
            "preset_path": "presets/surge-base.vstpreset",
            "param_spec_name": "surge_simple",
            "renderer_version": TEST_PLUGIN_VERSION,
            "sample_rate": 16000,
            "channels": 2,
            "velocity": 100,
            "signal_duration_seconds": 4.0,
            "min_loudness": -55.0,
            "samples_per_render_batch": 32,
            "samples_per_shard": 10000,
            "gui_toggle_cadence": "never",
        },
    }
    kwargs.update(overrides)
    return kwargs


@pytest.fixture()
def spec(tmp_path: Path) -> DatasetSpec:
    """Return a valid single-shard DatasetSpec."""
    return DatasetSpec(**_base_spec_kwargs(tmp_path))  # type: ignore[arg-type]


def _multi_shard_spec(tmp_path: Path, n: int = 3) -> DatasetSpec:
    """Return a DatasetSpec with ``n`` shards (deterministic filenames/seeds)."""
    kwargs = _base_spec_kwargs(
        tmp_path,
        train_val_test_sizes=[10000 * n, 0, 0],
    )
    return DatasetSpec(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# load_spec_from_uri — local path, file:// URI, r2:// URI dispatch
# ---------------------------------------------------------------------------


class TestLoadSpecFromUri:
    """``load_spec_from_uri`` accepts bare paths, ``file://`` URIs, and ``r2://`` URIs."""

    def test_bare_local_path_is_read_directly(self, spec: DatasetSpec, tmp_path: Path) -> None:
        """A non-URI argument is treated as a filesystem path.

        :param spec: Fixture-provided ``DatasetSpec``.
        :param tmp_path: Pytest tmp dir for the local spec JSON.
        """
        from synth_setter.pipeline.spec_io import load_spec_from_uri

        spec_path = tmp_path / "spec.json"
        spec_path.write_text(spec.model_dump_json())

        loaded = load_spec_from_uri(str(spec_path))

        assert loaded.task_name == spec.task_name

    def test_file_uri_is_read_from_local_disk(self, spec: DatasetSpec, tmp_path: Path) -> None:
        """A ``file://`` URI is decoded to a local path and read directly.

        :param spec: Fixture-provided ``DatasetSpec``.
        :param tmp_path: Pytest tmp dir for the local spec JSON.
        """
        from synth_setter.pipeline.spec_io import load_spec_from_uri

        spec_path = tmp_path / "spec.json"
        spec_path.write_text(spec.model_dump_json())

        loaded = load_spec_from_uri(spec_path.as_uri())

        assert loaded.task_name == spec.task_name


# ---------------------------------------------------------------------------
# run — full flow orchestration
# ---------------------------------------------------------------------------


class TestRun:
    """Run() orchestrates: generate → upload shard, per owned shard.

    No spec upload.
    """

    @pytest.fixture(autouse=True)
    def _set_default_skypilot_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default to a single-worker rank/world for tests that don't care about partitioning.

        ``run()`` now requires ``SYNTH_SETTER_WORKER_RANK`` / ``SYNTH_SETTER_NUM_WORKERS`` to be set
        (silent default removed — see ``read_rank_world_from_env``). Most tests in this class
        exercise behaviors orthogonal to partitioning, so set rank=0/world=1 by default; tests
        that probe multi-worker partitioning override via ``monkeypatch.setenv`` and tests for
        the missing-env contract override via ``monkeypatch.delenv``.
        """
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")

    @pytest.fixture(autouse=True)
    def _default_shard_absent_in_r2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default object_size → None so every shard is treated as absent (full render path).

        Tests that exercise the skip-existing path override this with their own
        ``monkeypatch.setattr`` on ``pipeline.r2_io.object_size``.
        """
        monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda *_a, **_k: None)

    @pytest.fixture()
    def patched_subprocess(self, fake_r2_remote: Path) -> Iterator[MagicMock]:  # noqa: ARG002
        """Patch ``subprocess.check_call`` with the renderer/rclone dispatcher.

        Pulls in ``fake_r2_remote`` (consumed by the rclone passthrough — see
        ``_materialize_or_passthrough_rclone``) so rclone copies land on the
        local-typed remote rooted at the tmp dir instead of hitting real R2.
        Yielding the mock lets tests introspect ``call_args_list`` (typically
        via ``_renderer_argv_lists`` to filter out interleaved rclone calls)
        and override ``side_effect`` per-test when a failure or no-write
        renderer is needed.

        :param fake_r2_remote: Local-typed R2 remote root (fixture-activation
            only — referenced via the ARG002 noqa).
        :yields MagicMock: Patched ``subprocess.check_call`` mock.
        """
        with patch(
            "synth_setter.cli.generate_dataset.subprocess.check_call",
            side_effect=_materialize_or_passthrough_rclone,
        ) as mock_check_call:
            yield mock_check_call

    def test_invokes_generate_vst_dataset_with_spec_derived_args(
        self,
        patched_subprocess: MagicMock,
        spec: DatasetSpec,
    ) -> None:
        """subprocess.check_call invokes generate_vst_dataset.py with spec-derived args.

        :param patched_subprocess: Subprocess dispatcher used to introspect the
            single renderer call's argv.
        :param spec: Fixture-provided ``DatasetSpec``.
        """
        run(spec)

        renderer_calls = _renderer_argv_lists(patched_subprocess)
        assert len(renderer_calls) == 1
        args = renderer_calls[0]
        # args = [VST_HEADLESS_WRAPPER (linux only), python, generate_vst_dataset.py, ...]
        assert any("generate_vst_dataset.py" in a for a in args)
        assert str(spec.render.samples_per_shard) in args

    def test_shard_generation_runs_under_headless_vst_wrapper(
        self,
        patched_subprocess: MagicMock,
        spec: DatasetSpec,
    ) -> None:
        """Prefix the VST subprocess with ``run-linux-vst-headless.sh`` on Linux.

        X11 bootstrap lives at the audio-rendering boundary (this subprocess), keeping the outer
        pipeline X11-agnostic. The wrapper is Linux-only (Xvfb is a Linux X11 server); on macOS and
        other platforms the generator is invoked directly without a wrapper prefix.

        :param patched_subprocess: Subprocess dispatcher used to introspect the
            renderer argv (looking for the headless-wrapper prefix on Linux).
        :param spec: Fixture-provided ``DatasetSpec``.
        """
        run(spec)

        renderer_calls = _renderer_argv_lists(patched_subprocess)
        assert len(renderer_calls) == 1
        args = renderer_calls[0]
        if sys.platform == "linux":
            assert args[0] == VST_HEADLESS_WRAPPER
            assert args[2] == "src/synth_setter/data/vst/generate_vst_dataset.py"
        else:
            assert VST_HEADLESS_WRAPPER not in args
            assert args[1] == "src/synth_setter/data/vst/generate_vst_dataset.py"

    def test_uploads_shard_to_r2_after_generation(
        self,
        spec: DatasetSpec,
        fake_r2_remote: Path,
        patched_subprocess: MagicMock,  # noqa: ARG002
    ) -> None:
        """Shard lands at the R2 URI implied by ``spec.r2`` and the shard's filename.

        State-based: no mock on ``_rclone_copy`` — the real ``rclone copy`` runs
        against the fake-local R2 remote rooted at ``fake_r2_remote``, and the
        test asserts on the materialized object on disk. The renderer subprocess
        ``check_call`` is patched via the shared ``patched_subprocess`` fixture
        so we don't actually shell out to the VST generator; the dispatcher's
        renderer branch writes the same empty HDF5 file that the renderer would.

        :param spec: Fixture-provided ``DatasetSpec``.
        :param fake_r2_remote: Local-typed rclone remote rooted at a tmp dir.
        :param patched_subprocess: Fixture-activation only (handles the
            ``subprocess.check_call`` patch).
        """
        run(spec)

        landed = fake_r2_remote / spec.r2.bucket / spec.r2.prefix / spec.shards[0].filename
        assert landed.is_file()

    def test_subprocess_failure_propagates(
        self,
        patched_subprocess: MagicMock,
        spec: DatasetSpec,
        fake_r2_remote: Path,
    ) -> None:
        """CalledProcessError from generate_vst_dataset propagates to caller.

        :param patched_subprocess: Subprocess dispatcher; overridden here to
            unconditionally raise so the renderer call short-circuits.
        :param spec: Fixture-provided ``DatasetSpec``.
        :param fake_r2_remote: Local-typed rclone remote — asserted empty since
            no shard should land when the renderer fails.
        """
        patched_subprocess.side_effect = subprocess.CalledProcessError(
            1, "generate_vst_dataset.py"
        )

        with pytest.raises(subprocess.CalledProcessError):
            run(spec)

        # No rclone copy reached the fake remote.
        assert not (fake_r2_remote / spec.r2.bucket / spec.r2.prefix).exists()

    def test_rclone_failure_propagates(
        self,
        patched_subprocess: MagicMock,  # noqa: ARG002
        spec: DatasetSpec,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """CalledProcessError from rclone (shard upload path) propagates to caller.

        Forces a real rclone failure by pointing the ``r2:`` remote at a
        nonexistent backend type. The renderer side of the dispatcher still
        materializes the shard file (so the source-existence check passes);
        ``rclone copy`` then raises ``CalledProcessError`` when it tries to
        construct the destination backend.

        :param patched_subprocess: Fixture-activation only (the renderer path
            still materializes the shard so the rclone source exists).
        :param spec: Fixture-provided ``DatasetSpec``.
        :param monkeypatch: Used to invalidate the ``r2:`` remote type so the
            real rclone subprocess exits non-zero on the copy.
        """
        monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "this-backend-does-not-exist")

        with pytest.raises(subprocess.CalledProcessError):
            run(spec)

    def test_run_with_three_shards_renders_each_shard(
        self,
        patched_subprocess: MagicMock,
        fake_r2_remote: Path,
        tmp_path: Path,
    ) -> None:
        """Multi-shard run invokes generate_vst_dataset.py once per shard, in order.

        :param patched_subprocess: Dispatcher mock; ``_renderer_argv_lists``
            filters out rclone calls so we can introspect the per-shard
            output-path argv.
        :param fake_r2_remote: All three shards should land in this remote.
        :param tmp_path: Pytest tmp dir used by ``_multi_shard_spec``.
        """
        spec = _multi_shard_spec(tmp_path, n=3)

        run(spec)

        renderer_calls = _renderer_argv_lists(patched_subprocess)
        assert len(renderer_calls) == 3
        rendered_filenames = [
            Path(args[find_script_index(args) + 1]).name for args in renderer_calls
        ]
        assert rendered_filenames == [s.filename for s in spec.shards]
        # State-based proof: every shard landed in the fake remote.
        for shard in spec.shards:
            assert (fake_r2_remote / spec.r2.bucket / spec.r2.prefix / shard.filename).is_file()

    def test_each_shard_uploaded_after_its_render(
        self,
        fake_r2_remote: Path,  # noqa: ARG002
        tmp_path: Path,
    ) -> None:
        """Render and upload are interleaved per shard: render0, upload0, render1, upload1, ...

        :param fake_r2_remote: Fixture-activation only — the rclone passthrough
            in the custom dispatcher needs the local-typed remote.
        :param tmp_path: Pytest tmp dir used by ``_multi_shard_spec``.
        """
        spec = _multi_shard_spec(tmp_path, n=3)
        events: list[str] = []

        def _record_dispatcher(args: list[str]) -> int:
            if args and args[0] == "rclone":
                events.append("rclone")
                return _REAL_CHECK_CALL(args)
            events.append("renderer")
            return _materialize_shard(args)

        with patch(
            "synth_setter.cli.generate_dataset.subprocess.check_call",
            side_effect=_record_dispatcher,
        ):
            run(spec)

        assert events == [
            "renderer",  # shard 0
            "rclone",
            "renderer",  # shard 1
            "rclone",
            "renderer",  # shard 2
            "rclone",
        ]

    def test_local_shard_file_removed_after_upload(
        self,
        fake_r2_remote: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Each shard's local HDF5 is unlinked between its upload and the next render.

        Verifies the disk-bounding invariant via an interleaved event stream
        (renderer / rclone / unlink). For every shard the test asserts the
        per-shard sequence ``(renderer, rclone, unlink)`` on the same path —
        which proves both ordering (shard N is unlinked *before* shard N+1's
        renderer runs) and final-shard coverage (the last shard's unlink is
        recorded, not masked by the outer ``TemporaryDirectory`` teardown).

        :param fake_r2_remote: Local-typed R2 remote rooted at a tmp dir.
        :param tmp_path: Pytest tmp dir used by ``_multi_shard_spec``.
        :param monkeypatch: Used to wrap ``Path.unlink`` with a call-recording
            spy that delegates to the real implementation.
        """
        spec = _multi_shard_spec(tmp_path, n=3)
        events: list[tuple[str, Path]] = []

        def _record_dispatcher(args: list[str]) -> int:
            if args and args[0] == "rclone":
                # rclone copy argv ends with [src, dest]; the src is the
                # just-rendered shard's local path.
                events.append(("rclone", Path(args[-2])))
                return _REAL_CHECK_CALL(args)
            out_path = Path(args[find_script_index(args) + 1])
            events.append(("renderer", out_path))
            return _materialize_shard(args)

        real_unlink = Path.unlink

        def _spy_unlink(self: Path, missing_ok: bool = False) -> None:
            events.append(("unlink", self))
            real_unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", _spy_unlink)

        with patch(
            "synth_setter.cli.generate_dataset.subprocess.check_call",
            side_effect=_record_dispatcher,
        ):
            run(spec)

        # Restrict unlink events to those targeting an uploaded shard source —
        # the spy captures every Path.unlink in-process, but only shard-source
        # unlinks are part of the disk-bounding contract under test.
        upload_sources = [p for kind, p in events if kind == "rclone"]
        shard_paths = set(upload_sources)
        shard_events = [
            (kind, p)
            for kind, p in events
            if kind in ("renderer", "rclone") or (kind == "unlink" and p in shard_paths)
        ]
        expected = [
            entry
            for src in upload_sources
            for entry in (("renderer", src), ("rclone", src), ("unlink", src))
        ]
        assert len(upload_sources) == 3
        assert shard_events == expected, (
            "per-shard sequence (renderer, rclone, unlink) was not maintained: "
            f"got {shard_events!r}, expected {expected!r}"
        )
        for shard in spec.shards:
            assert (fake_r2_remote / spec.r2.bucket / spec.r2.prefix / shard.filename).is_file()

    def test_subprocess_failure_in_second_shard_propagates_immediately(
        self,
        fake_r2_remote: Path,
        tmp_path: Path,
    ) -> None:
        """Mid-loop subprocess failure raises immediately; later shards are not attempted.

        :param fake_r2_remote: Local-typed R2 remote — asserted to contain only
            shard 0 (rclone runs for it; shard 1's renderer raises; shard 2
            never runs).
        :param tmp_path: Pytest tmp dir used by ``_multi_shard_spec``.
        """
        spec = _multi_shard_spec(tmp_path, n=3)
        renderer_call_count = 0

        def _side_effect(args: list[str]) -> int:
            nonlocal renderer_call_count
            if args and args[0] == "rclone":
                return _REAL_CHECK_CALL(args)
            renderer_call_count += 1
            if renderer_call_count == 2:
                raise subprocess.CalledProcessError(1, "generate_vst_dataset.py")
            return _materialize_shard(args)

        with patch(
            "synth_setter.cli.generate_dataset.subprocess.check_call",
            side_effect=_side_effect,
        ):
            with pytest.raises(subprocess.CalledProcessError):
                run(spec)

        assert renderer_call_count == 2
        # State-based proof of fail-fast: only shard 0 landed in R2.
        bucket_prefix = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
        assert (bucket_prefix / spec.shards[0].filename).is_file()
        assert not (bucket_prefix / spec.shards[1].filename).exists()
        assert not (bucket_prefix / spec.shards[2].filename).exists()

    def test_subprocess_exits_zero_without_writing_shard_raises(
        self,
        fake_r2_remote: Path,
        spec: DatasetSpec,
    ) -> None:
        """If the renderer exits 0 but never wrote the expected shard file, fail loudly.

        Catches a generator bug at the rendering boundary instead of letting it surface as a less-
        direct rclone "source not found" further down the pipeline.

        :param fake_r2_remote: Local-typed R2 remote — must remain empty since
            no shard file is written and rclone is therefore never invoked.
        :param spec: Fixture-provided ``DatasetSpec``.
        """
        # Renderer-only side effect: return 0 without writing the shard file,
        # so the ``shard_path.is_file()`` guard raises before any rclone call.
        with patch(
            "synth_setter.cli.generate_dataset.subprocess.check_call",
            return_value=0,
        ):
            with pytest.raises(RuntimeError, match="did not write expected shard file"):
                run(spec)

        assert not (fake_r2_remote / spec.r2.bucket / spec.r2.prefix).exists()

    def test_renderer_version_mismatch_raises_before_uploads(
        self,
        patched_subprocess: MagicMock,
        fake_r2_remote: Path,
        tmp_path: Path,
    ) -> None:
        """Fail before any rclone/subprocess work when plugin version disagrees with spec.

        This prevents emitting a shard tagged with the wrong renderer_version.

        :param patched_subprocess: Subprocess dispatcher; asserted never invoked.
        :param fake_r2_remote: Local-typed R2 remote — asserted empty.
        :param tmp_path: Pytest tmp dir used by ``_base_spec_kwargs``.
        """
        kwargs = _base_spec_kwargs(tmp_path)
        kwargs["render"] = {**kwargs["render"], "renderer_version": "999.999.999"}  # type: ignore[dict-item]
        spec = DatasetSpec(**kwargs)  # type: ignore[arg-type]

        with pytest.raises(RuntimeError, match="Renderer version mismatch"):
            run(spec)
        patched_subprocess.assert_not_called()
        assert not (fake_r2_remote / spec.r2.bucket / spec.r2.prefix).exists()

    def test_run_raises_when_skypilot_env_missing(
        self,
        patched_subprocess: MagicMock,
        fake_r2_remote: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Missing partition env → ValueError before any rclone or subprocess work.

        Removes the silent-default smell where a worker invoked without partition env would
        otherwise duplicate every shard across every node.

        :param patched_subprocess: Subprocess dispatcher; asserted never invoked.
        :param fake_r2_remote: Local-typed R2 remote — asserted empty.
        :param tmp_path: Pytest tmp dir used by ``_multi_shard_spec``.
        :param monkeypatch: Used to unset the rank/world env vars.
        """
        monkeypatch.delenv("SYNTH_SETTER_WORKER_RANK", raising=False)
        monkeypatch.delenv("SYNTH_SETTER_NUM_WORKERS", raising=False)
        spec = _multi_shard_spec(tmp_path, n=3)

        with pytest.raises(ValueError) as excinfo:
            run(spec)
        message = str(excinfo.value)
        assert "SYNTH_SETTER_WORKER_RANK" in message
        assert "SYNTH_SETTER_NUM_WORKERS" in message
        patched_subprocess.assert_not_called()
        assert not (fake_r2_remote / spec.r2.bucket / spec.r2.prefix).exists()

    def test_rank_0_of_2_renders_only_first_half_of_shards(
        self,
        patched_subprocess: MagicMock,
        fake_r2_remote: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Worker 0 of a 2-node partition with 3 shards renders shards 0 and 1 only.

        :param patched_subprocess: Subprocess dispatcher used to introspect
            renderer argv.
        :param fake_r2_remote: Local-typed R2 remote — asserted to contain only
            shards 0 and 1.
        :param tmp_path: Pytest tmp dir used by ``_multi_shard_spec``.
        :param monkeypatch: Used to set the rank/world env vars.
        """
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "2")
        spec = _multi_shard_spec(tmp_path, n=3)

        run(spec)

        rendered_filenames = [
            Path(args[find_script_index(args) + 1]).name
            for args in _renderer_argv_lists(patched_subprocess)
        ]
        assert rendered_filenames == [spec.shards[0].filename, spec.shards[1].filename]
        bucket_prefix = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
        assert (bucket_prefix / spec.shards[0].filename).is_file()
        assert (bucket_prefix / spec.shards[1].filename).is_file()
        assert not (bucket_prefix / spec.shards[2].filename).exists()

    def test_rank_1_of_2_renders_only_remaining_shard(
        self,
        patched_subprocess: MagicMock,
        fake_r2_remote: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Worker 1 of a 2-node partition with 3 shards renders shard 2 only.

        :param patched_subprocess: Subprocess dispatcher used to introspect
            renderer argv.
        :param fake_r2_remote: Local-typed R2 remote — asserted to contain only
            shard 2.
        :param tmp_path: Pytest tmp dir used by ``_multi_shard_spec``.
        :param monkeypatch: Used to set the rank/world env vars.
        """
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "1")
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "2")
        spec = _multi_shard_spec(tmp_path, n=3)

        run(spec)

        rendered_filenames = [
            Path(args[find_script_index(args) + 1]).name
            for args in _renderer_argv_lists(patched_subprocess)
        ]
        assert rendered_filenames == [spec.shards[2].filename]
        bucket_prefix = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
        assert not (bucket_prefix / spec.shards[0].filename).exists()
        assert not (bucket_prefix / spec.shards[1].filename).exists()
        assert (bucket_prefix / spec.shards[2].filename).is_file()

    def test_excess_worker_renders_no_shards(
        self,
        patched_subprocess: MagicMock,
        fake_r2_remote: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When world > num_shards, the excess workers exit cleanly without rendering.

        A 4-node partition over 3 shards leaves worker 3 with an empty range — it renders zero
        shards and makes no rclone calls.

        :param patched_subprocess: Subprocess dispatcher; asserted never invoked.
        :param fake_r2_remote: Local-typed R2 remote — asserted empty.
        :param tmp_path: Pytest tmp dir used by ``_multi_shard_spec``.
        :param monkeypatch: Pytest fixture used to set partition env vars.
        """
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "3")
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "4")
        spec = _multi_shard_spec(tmp_path, n=3)

        run(spec)

        patched_subprocess.assert_not_called()
        assert not (fake_r2_remote / spec.r2.bucket / spec.r2.prefix).exists()

    # Skip-existing-shards — see #750.

    def test_run_skips_render_when_shard_already_in_r2(
        self,
        patched_subprocess: MagicMock,
        fake_r2_remote: Path,
        spec: DatasetSpec,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Object present (size > 0) → renderer is not invoked, shard upload is not attempted.

        :param patched_subprocess: Subprocess dispatcher; asserted never invoked.
        :param fake_r2_remote: Local-typed R2 remote — asserted empty (the
            probe stub returns "present" without seeding an actual file).
        :param spec: Fixture-provided ``DatasetSpec``.
        :param monkeypatch: Used to override the probe to claim the shard is
            already in R2.
        """
        monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda *_a, **_k: 12345)

        run(spec)

        patched_subprocess.assert_not_called()
        assert not (fake_r2_remote / spec.r2.bucket / spec.r2.prefix).exists()

    def test_run_renders_when_object_absent(
        self,
        patched_subprocess: MagicMock,
        fake_r2_remote: Path,
        spec: DatasetSpec,
    ) -> None:
        """Object absent (None) → render proceeds as before.

        Relies on the autouse ``_default_shard_absent_in_r2`` fixture's default of None.

        :param patched_subprocess: Subprocess dispatcher; renderer is asserted
            to fire exactly once.
        :param fake_r2_remote: Local-typed R2 remote — shard should land here.
        :param spec: Fixture-provided ``DatasetSpec``.
        """
        run(spec)

        renderer_calls = _renderer_argv_lists(patched_subprocess)
        assert len(renderer_calls) == 1
        assert (
            fake_r2_remote / spec.r2.bucket / spec.r2.prefix / spec.shards[0].filename
        ).is_file()

    def test_run_renders_when_object_zero_size(
        self,
        patched_subprocess: MagicMock,
        fake_r2_remote: Path,
        spec: DatasetSpec,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Zero-byte object is treated as absent — defensive against half-uploaded objects.

        :param patched_subprocess: Subprocess dispatcher; renderer is asserted
            to fire exactly once.
        :param fake_r2_remote: Local-typed R2 remote — shard should land here.
        :param spec: Fixture-provided ``DatasetSpec``.
        :param monkeypatch: Used to override the probe to report 0 bytes.
        """
        monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda *_a, **_k: 0)

        run(spec)

        renderer_calls = _renderer_argv_lists(patched_subprocess)
        assert len(renderer_calls) == 1
        assert (
            fake_r2_remote / spec.r2.bucket / spec.r2.prefix / spec.shards[0].filename
        ).is_file()

    def test_run_skip_path_probes_full_object_uri_per_shard(
        self,
        patched_subprocess: MagicMock,  # noqa: ARG002
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Probe is called once per assigned shard with the full object URI under r2.prefix.

        :param patched_subprocess: Fixture-activation only (renderer
            materializes shards so rclone copies have a valid source).
        :param tmp_path: Pytest tmp dir used by ``_multi_shard_spec``.
        :param monkeypatch: Used to install the probe-URI capture stub.
        """
        spec = _multi_shard_spec(tmp_path, n=3)
        probed_uris: list[str] = []

        def _probe(uri: str) -> None:
            probed_uris.append(uri)
            return None

        monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", _probe)

        run(spec)

        assert probed_uris == [
            f"r2://{spec.r2.bucket}/{spec.r2.prefix}{shard.filename}" for shard in spec.shards
        ]

    def test_run_renders_only_absent_shards_in_mixed_run(
        self,
        patched_subprocess: MagicMock,
        fake_r2_remote: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Mid-run resumption: shard 0 already in R2, shards 1 and 2 absent → render only 1 and 2.

        :param patched_subprocess: Subprocess dispatcher used to introspect
            renderer argv.
        :param fake_r2_remote: Local-typed R2 remote — shards 1 and 2 land
            here; shard 0 does not (it was reported "present" by the probe).
        :param tmp_path: Pytest tmp dir used by ``_multi_shard_spec``.
        :param monkeypatch: Used to install the per-shard probe stub.
        """
        spec = _multi_shard_spec(tmp_path, n=3)

        def _present_only_for_shard_0(uri: str) -> int | None:
            return 9999 if uri.endswith("shard-000000.h5") else None

        monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", _present_only_for_shard_0)

        run(spec)

        rendered_filenames = [
            Path(args[find_script_index(args) + 1]).name
            for args in _renderer_argv_lists(patched_subprocess)
        ]
        assert rendered_filenames == ["shard-000001.h5", "shard-000002.h5"]
        bucket_prefix = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
        assert not (bucket_prefix / "shard-000000.h5").exists()
        assert (bucket_prefix / "shard-000001.h5").is_file()
        assert (bucket_prefix / "shard-000002.h5").is_file()

    @patch("synth_setter.cli.generate_dataset.logger")
    def test_run_logs_summary_with_rendered_and_skipped_counts(
        self,
        mock_logger: MagicMock,
        patched_subprocess: MagicMock,  # noqa: ARG002
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """End-of-run summary reports rendered/skipped counts over the assigned range.

        :param mock_logger: Patched ``generate_dataset.logger`` for capturing
            the summary line.
        :param patched_subprocess: Fixture-activation only (renderer
            materializes shards so rclone copies have a valid source).
        :param tmp_path: Pytest tmp dir used by ``_multi_shard_spec``.
        :param monkeypatch: Used to install the per-shard probe stub.
        """
        spec = _multi_shard_spec(tmp_path, n=3)

        def _present_only_for_shard_0(uri: str) -> int | None:
            return 9999 if uri.endswith("shard-000000.h5") else None

        monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", _present_only_for_shard_0)

        run(spec)

        info_messages = [str(c.args[0]) for c in mock_logger.info.call_args_list]
        summary_lines = [m for m in info_messages if "rendered=" in m and "skipped=" in m]
        assert len(summary_lines) == 1, f"expected exactly one summary line, got: {info_messages}"
        assert "rendered=2" in summary_lines[0]
        assert "skipped=1" in summary_lines[0]

    def test_run_probe_failure_propagates(
        self,
        patched_subprocess: MagicMock,
        spec: DatasetSpec,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-zero rclone exit during the probe propagates as CalledProcessError.

        :param patched_subprocess: Subprocess dispatcher; asserted never
            invoked (the probe failure raises before any render/upload).
        :param spec: Fixture-provided ``DatasetSpec``.
        :param monkeypatch: Used to install the raising probe stub.
        """

        def _raise(*_a: object, **_k: object) -> None:
            raise subprocess.CalledProcessError(1, ["rclone", "lsf"])

        monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", _raise)

        with pytest.raises(subprocess.CalledProcessError):
            run(spec)

        patched_subprocess.assert_not_called()

    def test_render_retries_transient_failure_when_max_retries_set(
        self,
        fake_r2_remote: Path,
        tmp_path: Path,
    ) -> None:
        """``max_retries=1`` + flaky renderer (1 fail, then success) → shard lands in R2.

        The renderer-subprocess retry loop covers transient X11 / Xvfb init races
        and pedalboard load hiccups on first call into a fresh subprocess. Rclone
        sits outside the loop (rclone has its own --retries=3); only the renderer
        call is wrapped.

        :param fake_r2_remote: Local-typed R2 remote — asserted to contain the
            shard after the retried render succeeds.
        :param tmp_path: Pytest tmp dir used by ``_base_spec_kwargs``.
        """
        kwargs = _base_spec_kwargs(tmp_path)
        kwargs["render"] = {**kwargs["render"], "max_retries": 1}  # type: ignore[dict-item]
        spec = DatasetSpec(**kwargs)  # type: ignore[arg-type]
        renderer_calls = 0

        def _flaky_dispatcher(args: list[str]) -> int:
            nonlocal renderer_calls
            if args and args[0] == "rclone":
                return _REAL_CHECK_CALL(args)
            renderer_calls += 1
            if renderer_calls == 1:
                raise subprocess.CalledProcessError(1, "generate_vst_dataset.py")
            return _materialize_shard(args)

        with patch(
            "synth_setter.cli.generate_dataset.subprocess.check_call",
            side_effect=_flaky_dispatcher,
        ):
            run(spec)

        assert renderer_calls == 2
        landed = fake_r2_remote / spec.r2.bucket / spec.r2.prefix / spec.shards[0].filename
        assert landed.is_file()

    def test_parallel_render_uses_thread_pool_and_uploads_all_shards(
        self,
        fake_r2_remote: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``render.parallel=True`` + 4 shards → ≥2 worker threads; every shard uploads.

        Pins ``available_cpus`` to 8 so the dispatch heuristic
        ``min(max(1, available_cpus() // 2), len(my_range))`` resolves to 4
        workers regardless of CI runner CPU count. The dispatcher stub blocks
        each render until the second thread enters, forcing the pool to
        actually parallelize.

        :param fake_r2_remote: Local-typed R2 remote — asserted to contain
            every shard after the parallel dispatch.
        :param tmp_path: Pytest tmp dir used by ``_base_spec_kwargs``.
        :param monkeypatch: Pins ``available_cpus`` so pool size is deterministic.
        """
        monkeypatch.setattr("synth_setter.cli.generate_dataset.available_cpus", lambda: 8)
        kwargs = _base_spec_kwargs(tmp_path, train_val_test_sizes=[40000, 0, 0])
        kwargs["render"] = {**kwargs["render"], "parallel": True}  # type: ignore[dict-item]
        spec = DatasetSpec(**kwargs)  # type: ignore[arg-type]
        assert len(spec.shards) == 4
        thread_ids: set[int] = set()
        lock = threading.Lock()
        two_threads_seen = threading.Event()

        def _thread_recording_dispatcher(args: list[str]) -> int:
            if args and args[0] == "rclone":
                return _REAL_CHECK_CALL(args)
            with lock:
                thread_ids.add(threading.get_ident())
                if len(thread_ids) >= 2:
                    two_threads_seen.set()
            two_threads_seen.wait(timeout=5.0)
            return _materialize_shard(args)

        with patch(
            "synth_setter.cli.generate_dataset.subprocess.check_call",
            side_effect=_thread_recording_dispatcher,
        ):
            run(spec)

        assert len(thread_ids) >= 2
        bucket_prefix = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
        for shard in spec.shards:
            assert (bucket_prefix / shard.filename).is_file()

    def test_parallel_render_propagates_subprocess_failure(
        self,
        fake_r2_remote: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One failing render in a parallel pool → CalledProcessError surfaces fail-fast.

        Pins ``available_cpus`` to 4 so pool size resolves to 2 workers. The
        first thread to enter fails; the in-flight peer can complete, but
        the remaining two futures get ``cancel_futures=True``-aborted before
        they start. Net effect: renderer fires at most twice (1 fail + ≤1
        in-flight peer), and at most one shard lands in R2.

        :param fake_r2_remote: Local-typed R2 remote — used for state-based
            cancellation check.
        :param tmp_path: Pytest tmp dir used by ``_base_spec_kwargs``.
        :param monkeypatch: Pins ``available_cpus`` so pool size is deterministic.
        """
        monkeypatch.setattr("synth_setter.cli.generate_dataset.available_cpus", lambda: 4)
        kwargs = _base_spec_kwargs(tmp_path, train_val_test_sizes=[40000, 0, 0])
        kwargs["render"] = {**kwargs["render"], "parallel": True}  # type: ignore[dict-item]
        spec = DatasetSpec(**kwargs)  # type: ignore[arg-type]
        renderer_call_count = 0
        lock = threading.Lock()

        def _one_failing(args: list[str]) -> int:
            nonlocal renderer_call_count
            if args and args[0] == "rclone":
                return _REAL_CHECK_CALL(args)
            with lock:
                renderer_call_count += 1
                this_attempt = renderer_call_count
            if this_attempt == 1:
                raise subprocess.CalledProcessError(1, "generate_vst_dataset.py")
            return _materialize_shard(args)

        with patch(
            "synth_setter.cli.generate_dataset.subprocess.check_call",
            side_effect=_one_failing,
        ):
            with pytest.raises(subprocess.CalledProcessError):
                run(spec)

        bucket_prefix = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
        landed = sum(1 for shard in spec.shards if (bucket_prefix / shard.filename).is_file())
        assert landed <= 1
        assert renderer_call_count <= 2

    def test_render_raises_after_exhausting_max_retries(
        self,
        fake_r2_remote: Path,
        tmp_path: Path,
    ) -> None:
        """``max_retries=2`` + always-failing renderer → 3 attempts then propagate.

        Confirms the retry budget is bounded: ``max_retries + 1`` total attempts,
        then ``CalledProcessError`` surfaces and no shard is uploaded.

        :param fake_r2_remote: Local-typed R2 remote — asserted empty since
            every render attempt failed.
        :param tmp_path: Pytest tmp dir used by ``_base_spec_kwargs``.
        """
        kwargs = _base_spec_kwargs(tmp_path)
        kwargs["render"] = {**kwargs["render"], "max_retries": 2}  # type: ignore[dict-item]
        spec = DatasetSpec(**kwargs)  # type: ignore[arg-type]
        renderer_calls = 0

        def _always_fails(args: list[str]) -> int:
            nonlocal renderer_calls
            if args and args[0] == "rclone":
                return _REAL_CHECK_CALL(args)
            renderer_calls += 1
            raise subprocess.CalledProcessError(1, "generate_vst_dataset.py")

        with patch(
            "synth_setter.cli.generate_dataset.subprocess.check_call",
            side_effect=_always_fails,
        ):
            with pytest.raises(subprocess.CalledProcessError):
                run(spec)

        assert renderer_calls == 3
        assert not (fake_r2_remote / spec.r2.bucket / spec.r2.prefix).exists()


# ---------------------------------------------------------------------------
# build_generate_args — arg construction from spec + shard
# ---------------------------------------------------------------------------


class TestBuildGenerateArgs:
    """build_generate_args() produces correct CLI arg lists from spec + shard."""

    def test_output_file_uses_shard_filename(self, spec: DatasetSpec, tmp_path: Path) -> None:
        """Output file path is {output_dir}/{shard.filename}."""
        shard = spec.shards[0]

        args = build_generate_args(spec, shard, tmp_path)

        assert args[2] == str(tmp_path / "shard-000000.h5")

    def test_samples_per_shard_passed_as_option(self, spec: DatasetSpec) -> None:
        """samples_per_shard is emitted as ``--samples_per_shard <count>`` flag.

        The CLI no longer takes a positional ``num_samples`` — every renderer
        config field is exposed as a flag, including the per-shard sample count.
        """
        shard = spec.shards[0]

        args = build_generate_args(spec, shard, Path("out"))

        flag_idx = args.index("--samples_per_shard")
        assert args[flag_idx + 1] == str(spec.render.samples_per_shard)

    def test_all_render_config_fields_passed_as_options(self, spec: DatasetSpec) -> None:
        """The flag set equals ``RenderConfig.model_fields`` — auto-derived parity guard."""
        shard = spec.shards[0]

        args = build_generate_args(spec, shard, Path("out"))

        option_keys: set[str] = {arg.lstrip("-") for arg in args if arg.startswith("--")}

        assert option_keys == set(RenderConfig.model_fields.keys())

    def test_args_start_with_python_and_script(self, spec: DatasetSpec) -> None:
        """First arg is the Python executable, second is the generation script."""
        shard = spec.shards[0]

        args = build_generate_args(spec, shard, Path("out"))

        assert args[1] == "src/synth_setter/data/vst/generate_vst_dataset.py"


# ---------------------------------------------------------------------------
# spec_from_cfg — Hydra-composed cfg → DatasetSpec
# ---------------------------------------------------------------------------


class TestSpecFromCfg:
    """``spec_from_cfg`` drops Hydra-only groups and constructs a DatasetSpec."""

    def test_drops_non_spec_groups(self, valid_dataset_spec_kwargs: dict[str, object]) -> None:
        """``data``, ``paths``, ``hydra`` are dropped so strict validation passes.

        DatasetSpec is configured with ``extra="forbid"``; if any of these groups leaked through,
        construction would raise on the unknown field. The assertion is implicit in the absence
        of a ValidationError. After the ``R2Location`` migration ``r2`` is *not* dropped —
        it composes from ``configs/r2/default.yaml`` directly into ``DatasetSpec.r2``.
        """
        from omegaconf import OmegaConf

        from synth_setter.cli.generate_dataset import spec_from_cfg

        cfg_dict: dict[str, object] = dict(valid_dataset_spec_kwargs)
        cfg_dict["data"] = {"sample_rate": 16000}
        cfg_dict["paths"] = {"root_dir": "/fake-root"}
        cfg_dict["hydra"] = {"runtime": {"output_dir": "/fake-out"}}

        spec = spec_from_cfg(OmegaConf.create(cfg_dict))

        assert spec.task_name == valid_dataset_spec_kwargs["task_name"]

    def test_r2_group_flows_into_nested_r2_field(
        self, valid_dataset_spec_kwargs: dict[str, object]
    ) -> None:
        """The ``r2`` group composes directly into ``DatasetSpec.r2`` (no flat-key indirection).

        Mirrors the production composition after the ``R2Location`` migration:
        ``configs/dataset.yaml`` no longer interpolates flat ``r2_bucket`` /
        ``r2_prefix_root`` keys — the group's content lands at ``cfg.r2`` and
        passes through to ``DatasetSpec.r2``.

        :param valid_dataset_spec_kwargs: Baseline spec kwargs from conftest.
        """
        from omegaconf import OmegaConf

        from synth_setter.cli.generate_dataset import spec_from_cfg

        kwargs = dict(valid_dataset_spec_kwargs)
        kwargs["r2"] = {"bucket": "from-group-bucket", "prefix_root": "data"}

        spec = spec_from_cfg(OmegaConf.create(kwargs))

        assert spec.r2.bucket == "from-group-bucket"
        assert spec.r2.prefix_root == "data"


# PROJECT_ROOT-bootstrap behavior is exercised end-to-end by tests/pipeline/test_configs/
# test_experiment_yamls.py — those tests fail with an InterpolationResolutionError if the
# module's import-time `rootutils.setup_root(...)` ever stops setting PROJECT_ROOT.


# ---------------------------------------------------------------------------
# _build_worker_cmd — shell-quoted cmd injection for sky.Task.run
# ---------------------------------------------------------------------------


class TestBuildWorkerCmd:
    """The worker cmd reconstructs the operator's Hydra invocation under bash."""

    @pytest.fixture()
    def spec(self, tmp_path: Path) -> DatasetSpec:
        """Reusable DatasetSpec for worker-cmd construction (no I/O — pure kwargs).

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :return: A ``DatasetSpec`` built from base kwargs.
        """
        return DatasetSpec(**_base_spec_kwargs(tmp_path))  # type: ignore[arg-type]

    def test_cmd_uses_from_hydra_console_script(self, spec: DatasetSpec) -> None:
        """The worker reproduces the composition by re-entering the from_hydra entry point.

        :param spec: Fixture-provided ``DatasetSpec``.
        """
        from synth_setter.cli.generate_dataset import _build_worker_cmd

        cmd = _build_worker_cmd(["experiment=foo"], spec)
        assert "synth-setter-generate-dataset-from-hydra" in cmd
        assert "experiment=foo" in cmd

    def test_cmd_cds_to_worker_repo_root_not_launcher_repo(self, spec: DatasetSpec) -> None:
        """Cd target is the worker checkout, not the launcher's path.

        :param spec: Fixture-provided ``DatasetSpec``.
        """
        from synth_setter.cli.generate_dataset import _WORKER_REPO_ROOT, _build_worker_cmd

        cmd = _build_worker_cmd([], spec)
        assert cmd.startswith(f"cd {_WORKER_REPO_ROOT}")
        assert _WORKER_REPO_ROOT == "/home/build/synth-setter"

    def test_cmd_runs_sync_worker_checkout_before_exec(self, spec: DatasetSpec) -> None:
        """sync_worker_checkout.sh bypasses dev-snapshot bake-lag when WORKER_GIT_REF is set.

        :param spec: Fixture-provided ``DatasetSpec``.
        """
        from synth_setter.cli.generate_dataset import _build_worker_cmd

        cmd = _build_worker_cmd([], spec)
        sync_idx = cmd.find("bash scripts/sync_worker_checkout.sh")
        exec_idx = cmd.find("exec synth-setter-generate-dataset-from-hydra")
        assert sync_idx != -1, f"sync step missing from cmd: {cmd!r}"
        assert exec_idx != -1, f"exec step missing from cmd: {cmd!r}"
        assert sync_idx < exec_idx, "sync_worker_checkout must run before exec"

    def test_cmd_pins_spec_created_at_via_hydra_override(self, spec: DatasetSpec) -> None:
        """Worker compose must inherit launcher's created_at to land on the same r2.prefix.

        :param spec: Fixture-provided ``DatasetSpec``.
        """
        from synth_setter.cli.generate_dataset import _build_worker_cmd

        cmd = _build_worker_cmd([], spec)
        # `+key=value` is Hydra's add-key syntax; spec.created_at.isoformat() goes in verbatim
        # (no surrounding quotes added by shlex when the value has no shell metachars).
        assert f"+created_at={spec.created_at.isoformat()}" in cmd

    def test_cmd_shell_quotes_overrides_with_spaces(self, spec: DatasetSpec) -> None:
        """Spaces and special chars in an override survive bash interpretation in run:.

        :param spec: Fixture-provided ``DatasetSpec``.
        """
        from synth_setter.cli.generate_dataset import _build_worker_cmd

        cmd = _build_worker_cmd(["task_name=value with space"], spec)
        # shlex.quote wraps the whole assignment in single quotes; the bare-word form
        # would be split into two argv items by bash.
        assert "'task_name=value with space'" in cmd

    def test_cmd_handles_empty_operator_overrides(self, spec: DatasetSpec) -> None:
        """No operator overrides → cmd is just cd + sync + exec + pinned-runtime override.

        :param spec: Fixture-provided ``DatasetSpec``.
        """
        from synth_setter.cli.generate_dataset import _build_worker_cmd

        cmd = _build_worker_cmd([], spec)
        assert cmd.startswith("cd ")
        assert " && exec synth-setter-generate-dataset-from-hydra " in cmd
        # No bash-interpretable trailing whitespace that would surface as an empty argv item.
        assert cmd == cmd.rstrip()


# ---------------------------------------------------------------------------
# main — dispatching CLI entry: local vs SkyPilot
# ---------------------------------------------------------------------------


class TestMainDispatchBranches:
    """``main()`` composes the dataset cfg from argv, then dispatches local or via SkyPilot."""

    @pytest.fixture(autouse=True)
    def _set_default_skypilot_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set single-worker rank/world env so the local branch's run() succeeds.

        :param monkeypatch: Pytest fixture used to set env vars.
        """
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")

    @pytest.fixture(autouse=True)
    def _stub_spec_io_in_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub the spec_io helpers + ``ensure_r2_env_loaded`` so ``main()`` doesn't shell out.

        Tests access the mocks via ``gd.write_spec_locally``,
        ``gd.upload_spec``, and ``gd.r2_io.ensure_r2_env_loaded`` to keep test
        signatures stable for pydoclint.

        :param monkeypatch: Pytest fixture used to patch the helpers.
        """
        import synth_setter.cli.generate_dataset as gd

        write_mock = MagicMock(side_effect=lambda spec, out: Path(out) / "input_spec.json")
        upload_mock = MagicMock(return_value="r2://stub-bucket/stub-key/input_spec.json")
        monkeypatch.setattr("synth_setter.cli.generate_dataset.write_spec_locally", write_mock)
        monkeypatch.setattr("synth_setter.cli.generate_dataset.upload_spec", upload_mock)
        monkeypatch.setattr(gd.r2_io, "ensure_r2_env_loaded", MagicMock(return_value=None))

    def test_compute_template_null_calls_run_locally(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """compute_template=null routes to run(spec) with a DatasetSpec; dispatch stays unused.

        :param monkeypatch: Pytest fixture used to patch argv and module functions.
        """
        import synth_setter.cli.generate_dataset as gd
        import synth_setter.pipeline.skypilot_launch as sl

        # Use a real experiment so cfg.skypilot_launch resolves; override the plugin path
        # to the test VST3 so run() — which we replace below — sees the right spec shape.
        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
        ]
        monkeypatch.setattr("sys.argv", argv)

        recorded: dict[str, object] = {}

        def _fake_run(spec: object) -> None:
            recorded["spec"] = spec

        def _dispatch_must_not_fire(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("dispatch_via_skypilot must not be called on the local branch")

        monkeypatch.setattr(gd, "run", _fake_run)
        monkeypatch.setattr(sl, "dispatch_via_skypilot", _dispatch_must_not_fire)

        gd.main()

        spec = recorded.get("spec")
        assert isinstance(spec, DatasetSpec)
        assert spec.render.plugin_path == str(TEST_PLUGIN_VST3)

    def test_compute_template_set_calls_dispatch_via_skypilot(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """compute_template=<path> routes through dispatch_via_skypilot with cmd populated.

        :param monkeypatch: Pytest fixture used to patch argv and module functions.
        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        import synth_setter.cli.generate_dataset as gd
        import synth_setter.pipeline.skypilot_launch as sl

        # A bare-minimum compute YAML the loader will accept (resources + envs, no run:).
        template = tmp_path / "template.yaml"
        template.write_text("resources:\n  cloud: runpod\nenvs:\n  X: ''\n")

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
            f"skypilot_launch.compute_template={template}",
        ]
        monkeypatch.setattr("sys.argv", argv)

        recorded: dict[str, object] = {}

        def _fake_dispatch(spec: object, sky_cfg: object, **_kwargs: object) -> None:
            recorded["spec"] = spec
            recorded["sky_cfg"] = sky_cfg

        monkeypatch.setattr(sl, "dispatch_via_skypilot", _fake_dispatch)

        def _run_must_not_fire(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("run must not be called on the dispatch branch")

        monkeypatch.setattr(gd, "run", _run_must_not_fire)

        gd.main()

        assert "sky_cfg" in recorded
        sky_cfg = recorded["sky_cfg"]
        assert sky_cfg.compute_template == str(template)  # type: ignore[attr-defined]
        assert sky_cfg.cmd is not None  # type: ignore[attr-defined]
        # Every operator-supplied override (sans argv[0]) round-trips into the worker cmd
        # so the worker reproduces this composition byte-for-byte.
        for override in argv[1:]:
            assert override in sky_cfg.cmd, (  # type: ignore[attr-defined]
                f"override {override!r} missing from worker cmd: {sky_cfg.cmd!r}"  # type: ignore[attr-defined]
            )

    def test_operator_supplied_cmd_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A `+skypilot_launch.cmd=…` override is rejected before any dispatch fires.

        Uses Hydra's `+key=value` add-syntax because the key isn't in
        configs/skypilot_launch/default.yaml (struct-mode would otherwise reject it before our
        guard runs).

        :param monkeypatch: Pytest fixture used to set ``sys.argv``.
        """
        import synth_setter.cli.generate_dataset as gd
        import synth_setter.pipeline.skypilot_launch as sl

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
            "+skypilot_launch.cmd=rm -rf /",
        ]
        monkeypatch.setattr("sys.argv", argv)

        def _run_must_not_fire(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("run must not be called when cmd is rejected")

        def _dispatch_must_not_fire(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("dispatch_via_skypilot must not be called when cmd is rejected")

        monkeypatch.setattr(gd, "run", _run_must_not_fire)
        monkeypatch.setattr(sl, "dispatch_via_skypilot", _dispatch_must_not_fire)

        with pytest.raises(ValueError, match="skypilot_launch.cmd is launcher-internal"):
            gd.main()


class TestMainSpecPersistence:
    """``main()`` writes the local spec, loads R2 env, uploads the canonical spec on every path.

    The R2 upload is launcher-side and happens once per ``main()`` invocation:
    after the local write, before the local-run / dispatch branch is taken.
    Workers in the dispatch path no longer re-upload the spec (the worker's
    ``run(spec)`` writes shards only); the canonical R2 object exists before
    any worker boots.
    """

    @pytest.fixture(autouse=True)
    def _set_default_skypilot_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Single-worker rank/world so the local branch's ``run()`` shim succeeds.

        :param monkeypatch: Pytest fixture used to set env vars.
        """
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")

    @pytest.fixture(autouse=True)
    def _stub_run_and_spec_io(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub ``run()``, the spec_io helpers, and ``r2_io.ensure_r2_env_loaded``.

        Tests assert via the module-level mocks ``gd.write_spec_locally``,
        ``gd.upload_spec``, and ``gd.r2_io.ensure_r2_env_loaded`` to keep test
        signatures stable for pydoclint.

        :param monkeypatch: Pytest fixture used to patch module-level callables.
        """
        import synth_setter.cli.generate_dataset as gd

        monkeypatch.setattr(gd, "run", lambda _spec: None)
        monkeypatch.setattr(
            gd,
            "write_spec_locally",
            MagicMock(side_effect=lambda spec, out: Path(out) / "input_spec.json"),
        )
        monkeypatch.setattr(
            gd,
            "upload_spec",
            MagicMock(return_value="r2://stub-bucket/stub-key/input_spec.json"),
        )
        monkeypatch.setattr(gd.r2_io, "ensure_r2_env_loaded", MagicMock(return_value=None))

    @staticmethod
    def _dispatch_argv(template_path: Path) -> list[str]:
        """Build argv that triggers the dispatch branch of ``main()``.

        :param template_path: Path to a minimal SkyPilot compute template the
            ``skypilot_launch`` cfg loader will accept.
        :return: ``sys.argv`` overrides setting ``compute_template``.
        """
        return [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
            f"skypilot_launch.compute_template={template_path}",
        ]

    @staticmethod
    def _write_minimal_template(tmp_path: Path) -> Path:
        """Write the bare-minimum compute template YAML the loader accepts.

        :param tmp_path: Pytest fixture providing a fresh test directory.
        :return: Path to the written template.
        """
        template = tmp_path / "template.yaml"
        template.write_text("resources:\n  cloud: runpod\nenvs:\n  X: ''\n")
        return template

    def test_main_writes_local_spec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``main()`` calls ``write_spec_locally`` with the composed spec + output_dir.

        :param monkeypatch: Pytest fixture used to patch ``sys.argv``.
        """
        import synth_setter.cli.generate_dataset as gd

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
        ]
        monkeypatch.setattr("sys.argv", argv)

        gd.main()

        gd.write_spec_locally.assert_called_once()  # type: ignore[attr-defined]
        called_spec, called_out = gd.write_spec_locally.call_args[0]  # type: ignore[attr-defined]
        assert isinstance(called_spec, DatasetSpec)
        assert isinstance(called_out, Path)

    def test_local_run_uploads_spec_from_main(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Local-run branch uploads the spec from ``main()`` exactly once.

        :param monkeypatch: Pytest fixture used to patch ``sys.argv``.
        """
        import synth_setter.cli.generate_dataset as gd

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
        ]
        monkeypatch.setattr("sys.argv", argv)

        gd.main()

        gd.upload_spec.assert_called_once()  # type: ignore[attr-defined]
        called_spec = gd.upload_spec.call_args[0][0]  # type: ignore[attr-defined]
        assert isinstance(called_spec, DatasetSpec)

    def test_dispatch_branch_uploads_spec_from_main(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Dispatch branch also uploads from ``main()`` — worker no longer re-uploads.

        :param monkeypatch: Pytest fixture used to patch ``sys.argv``.
        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        import synth_setter.cli.generate_dataset as gd
        import synth_setter.pipeline.skypilot_launch as sl

        template = self._write_minimal_template(tmp_path)
        monkeypatch.setattr("sys.argv", self._dispatch_argv(template))
        monkeypatch.setattr(sl, "dispatch_via_skypilot", lambda *_a, **_k: None)

        gd.main()

        gd.upload_spec.assert_called_once()  # type: ignore[attr-defined]
        gd.write_spec_locally.assert_called_once()  # type: ignore[attr-defined]

    def test_main_ensures_r2_env_loaded_before_upload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``ensure_r2_env_loaded`` runs before ``upload_spec`` so creds are in place.

        :param monkeypatch: Pytest fixture used to patch ``sys.argv``.
        """
        import synth_setter.cli.generate_dataset as gd

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
        ]
        monkeypatch.setattr("sys.argv", argv)

        manager = MagicMock()
        manager.attach_mock(gd.r2_io.ensure_r2_env_loaded, "ensure_env")  # type: ignore[arg-type]
        manager.attach_mock(gd.upload_spec, "upload_spec")  # type: ignore[arg-type]

        gd.main()

        call_names = [c[0] for c in manager.mock_calls]
        assert call_names.index("ensure_env") < call_names.index("upload_spec")

    def test_dispatch_branch_passes_canonical_spec_uri_kwarg(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``main()`` threads ``spec.r2.input_spec_uri()`` into ``dispatch_via_skypilot``.

        PR-2 contract: the launcher passes the canonical spec URI (including
        the run's prefix) as a kwarg so the worker reads the same R2 object
        ``main()`` just uploaded. Uses ``spec.r2.input_spec_uri()`` (not
        ``spec.r2.uri(INPUT_SPEC_FILENAME)``) — the former includes the prefix.

        :param monkeypatch: Pytest fixture used to patch ``sys.argv`` + dispatch.
        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        import synth_setter.cli.generate_dataset as gd
        import synth_setter.pipeline.skypilot_launch as sl

        template = self._write_minimal_template(tmp_path)
        monkeypatch.setattr("sys.argv", self._dispatch_argv(template))

        recorded: dict[str, object] = {}

        def _fake_dispatch(spec: DatasetSpec, sky_cfg: object, **kwargs: object) -> None:
            recorded["spec"] = spec
            recorded["kwargs"] = kwargs

        monkeypatch.setattr(sl, "dispatch_via_skypilot", _fake_dispatch)

        gd.main()

        spec = recorded["spec"]
        assert isinstance(spec, DatasetSpec)
        kwargs = recorded["kwargs"]
        assert isinstance(kwargs, dict)
        assert kwargs["spec_uri"] == spec.r2.input_spec_uri()
