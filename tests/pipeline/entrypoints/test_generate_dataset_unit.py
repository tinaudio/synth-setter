"""Tests for synth_setter.cli.generate_dataset — spec-driven run.

The entrypoint's public surface:

- ``main()``: launcher-side orchestrator. Composes the cfg, writes the local
  ``input_spec.json`` mirror, runs ``r2_io.ensure_r2_env_loaded`` (dotenv +
  auth ping), uploads the canonical spec via ``spec_io.upload_spec``, then
  either calls ``generate(spec, work_dir, loggers)`` inline (local-run) or dispatches
  to a SkyPilot worker pod.
- ``generate(spec, work_dir, loggers)``: per-rank renderer. For each owned shard in
  ``spec.shards``, shells out to ``generate_vst_dataset.py`` writing into
  ``work_dir``, then uploads the shard to R2 at ``r2:{bucket}/{prefix}/``;
  rendered shards are retained under ``work_dir`` for downstream consumption.
  ``main()`` writes the canonical spec to R2 once on the launcher host.

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

import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest

from synth_setter.cli.generate_dataset import (
    build_generate_args,
    generate,
)
from synth_setter.data.vst.shapes import DATASET_FIELD_DTYPES
from synth_setter.pipeline.constants import INPUT_SPEC_FILENAME, STATS_NPZ_FILENAME
from synth_setter.pipeline.schemas.spec import DatasetSpec, RenderConfig
from synth_setter.resources import vst_headless_wrapper
from tests.helpers.render_subprocess import (
    REAL_CHECK_CALL as _REAL_CHECK_CALL,
)
from tests.helpers.render_subprocess import (
    materialize_or_passthrough_rclone as _materialize_or_passthrough_rclone,
)
from tests.helpers.render_subprocess import (
    materialize_shard as _materialize_shard,
)
from tests.helpers.subprocess_args import find_script_index

VST_HEADLESS_WRAPPER = str(vst_headless_wrapper())

# Reusable VST3 bundle with a real Contents/moduleinfo.json so
# extract_renderer_version (called by generate) returns a deterministic version
# without loading any .so via pedalboard. Version inside is "1.0.0-test" — the
# specs built in this file pin renderer_version to the same string so the
# constraint check passes.
TEST_PLUGIN_VST3 = Path(__file__).resolve().parent.parent / "fixtures" / "TestPlugin.vst3"
TEST_PLUGIN_VERSION = "1.0.0-test"


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
            "sample_rate": 44100,
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


class TestLoadSpecFromRoot:
    """``load_spec_from_root`` resolves ``input_spec.json`` under a dataset-root URI."""

    def test_trailing_slash_root_resolves_spec(self, spec: DatasetSpec, tmp_path: Path) -> None:
        """A root URI ending in ``/`` resolves ``<root>input_spec.json``.

        :param spec: Fixture-provided ``DatasetSpec``.
        :param tmp_path: Pytest tmp dir hosting the run prefix.
        """
        from synth_setter.pipeline.constants import INPUT_SPEC_FILENAME
        from synth_setter.pipeline.spec_io import load_spec_from_root

        (tmp_path / INPUT_SPEC_FILENAME).write_text(spec.model_dump_json())

        loaded = load_spec_from_root(f"{tmp_path.as_uri()}/")

        assert loaded.task_name == spec.task_name

    def test_root_without_trailing_slash_resolves_spec(
        self, spec: DatasetSpec, tmp_path: Path
    ) -> None:
        """A root URI lacking a trailing ``/`` still resolves ``input_spec.json``.

        :param spec: Fixture-provided ``DatasetSpec``.
        :param tmp_path: Pytest tmp dir hosting the run prefix.
        """
        from synth_setter.pipeline.constants import INPUT_SPEC_FILENAME
        from synth_setter.pipeline.spec_io import load_spec_from_root

        (tmp_path / INPUT_SPEC_FILENAME).write_text(spec.model_dump_json())

        loaded = load_spec_from_root(tmp_path.as_uri())

        assert loaded.task_name == spec.task_name


class TestRun:
    """Render → upload, per owned shard.

    No spec upload — ``main()`` writes it once.
    """

    @pytest.fixture(autouse=True)
    def _set_default_skypilot_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pin rank=0/world=1 explicitly so partition-agnostic tests are insulated from host env.

        ``read_rank_world_from_env`` defaults to ``(0, 1)`` when both vars are
        absent, but this fixture also overwrites any value the developer's shell
        may have exported (e.g. an in-flight multi-worker debugging session) so
        the partition-agnostic tests in this class stay deterministic. Tests
        that probe multi-worker partitioning override via ``monkeypatch.setenv``;
        the default-fallback test overrides via ``monkeypatch.delenv``.
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
        tmp_path: Path,
    ) -> None:
        """subprocess.check_call invokes generate_vst_dataset.py with spec-derived args.

        :param patched_subprocess: Subprocess dispatcher used to introspect the
            single renderer call's argv.
        :param spec: Fixture-provided ``DatasetSpec``.
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        """
        generate(spec, tmp_path, [])

        renderer_calls = _renderer_argv_lists(patched_subprocess)
        assert len(renderer_calls) == 1
        args = renderer_calls[0]
        # args = [VST_HEADLESS_WRAPPER (linux only), python, generate_vst_dataset.py, ...]
        assert any("generate_vst_dataset.py" in a for a in args)
        assert str(spec.render.samples_per_shard) in args

    def test_aborts_before_render_when_copy_source_spec_is_missing(
        self,
        patched_subprocess: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A dataset-copy run whose source spec is unsynced fails before any render.

        :param patched_subprocess: Renderer/rclone dispatcher; asserted to receive
            no renderer call because the copy preflight aborts first.
        :param tmp_path: Work dir for ``generate()`` and the (empty) copy root.
        """
        copy_root = tmp_path / "source"
        copy_root.mkdir()
        spec = DatasetSpec(
            **_base_spec_kwargs(tmp_path, copy_dataset_root=str(copy_root))  # type: ignore[arg-type]
        )

        with pytest.raises(ValueError, match=INPUT_SPEC_FILENAME):
            generate(spec, tmp_path, [])

        assert _renderer_argv_lists(patched_subprocess) == []

    def test_aborts_before_render_when_copy_source_spec_mismatches(
        self,
        patched_subprocess: MagicMock,
        tmp_path: Path,
    ) -> None:
        """A dataset-copy run whose source spec disagrees fails before any render.

        :param patched_subprocess: Renderer/rclone dispatcher; asserted to receive
            no renderer call because the copy preflight aborts first.
        :param tmp_path: Work dir for ``generate()`` and the copy source root.
        """
        copy_root = tmp_path / "source"
        copy_root.mkdir()
        source = DatasetSpec(
            **_base_spec_kwargs(  # type: ignore[arg-type]
                tmp_path,
                render={**_base_spec_kwargs(tmp_path)["render"], "param_spec_name": "surge_xt"},  # type: ignore[dict-item]
            )
        )
        (copy_root / INPUT_SPEC_FILENAME).write_text(source.model_dump_json())
        spec = DatasetSpec(
            **_base_spec_kwargs(tmp_path, copy_dataset_root=str(copy_root))  # type: ignore[arg-type]
        )

        with pytest.raises(ValueError, match="param_spec_name"):
            generate(spec, tmp_path, [])

        assert _renderer_argv_lists(patched_subprocess) == []

    def test_shard_generation_runs_under_headless_vst_wrapper(
        self,
        patched_subprocess: MagicMock,
        spec: DatasetSpec,
        tmp_path: Path,
    ) -> None:
        """Prefix the VST subprocess with ``run-linux-vst-headless.sh`` on Linux.

        X11 bootstrap lives at the audio-rendering boundary (this subprocess), keeping the outer
        pipeline X11-agnostic. The wrapper is Linux-only (Xvfb is a Linux X11 server); on macOS and
        other platforms the generator is invoked directly without a wrapper prefix.

        :param patched_subprocess: Subprocess dispatcher used to introspect the
            renderer argv (looking for the headless-wrapper prefix on Linux).
        :param spec: Fixture-provided ``DatasetSpec``.
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        """
        generate(spec, tmp_path, [])

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
        tmp_path: Path,
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
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        """
        generate(spec, tmp_path, [])

        landed = fake_r2_remote / spec.r2.bucket / spec.r2.prefix / spec.shards[0].filename
        assert landed.is_file()

    def test_subprocess_failure_propagates(
        self,
        patched_subprocess: MagicMock,
        spec: DatasetSpec,
        fake_r2_remote: Path,
        tmp_path: Path,
    ) -> None:
        """CalledProcessError from generate_vst_dataset propagates to caller.

        :param patched_subprocess: Subprocess dispatcher; overridden here to
            unconditionally raise so the renderer call short-circuits.
        :param spec: Fixture-provided ``DatasetSpec``.
        :param fake_r2_remote: Local-typed rclone remote — asserted empty since
            no shard should land when the renderer fails.
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        """
        patched_subprocess.side_effect = subprocess.CalledProcessError(
            1, "generate_vst_dataset.py"
        )

        with pytest.raises(subprocess.CalledProcessError):
            generate(spec, tmp_path, [])

        # No rclone copy reached the fake remote.
        assert not (fake_r2_remote / spec.r2.bucket / spec.r2.prefix).exists()

    def test_rclone_failure_propagates(
        self,
        patched_subprocess: MagicMock,  # noqa: ARG002
        spec: DatasetSpec,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
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
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        """
        monkeypatch.setenv("RCLONE_CONFIG_R2_TYPE", "this-backend-does-not-exist")

        with pytest.raises(subprocess.CalledProcessError):
            generate(spec, tmp_path, [])

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

        generate(spec, tmp_path, [])

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
            generate(spec, tmp_path, [])

        assert events == [
            "renderer",  # shard 0
            "rclone",
            "renderer",  # shard 1
            "rclone",
            "renderer",  # shard 2
            "rclone",
        ]

    def test_shards_persist_after_upload(
        self,
        fake_r2_remote: Path,
        tmp_path: Path,
    ) -> None:
        """Every rendered shard remains at ``work_dir / shard.filename`` after upload.

        Pins the post-upload retention contract: ``finalize_dataset`` and
        post-mortem consumers expect shards to outlive the render+upload step.

        :param fake_r2_remote: Local-typed R2 remote — asserted to contain every
            shard alongside the on-disk copies.
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        """
        spec = _multi_shard_spec(tmp_path, n=3)

        with patch(
            "synth_setter.cli.generate_dataset.subprocess.check_call",
            side_effect=_materialize_or_passthrough_rclone,
        ):
            generate(spec, tmp_path, [])

        bucket_prefix = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
        for shard in spec.shards:
            assert (tmp_path / shard.filename).is_file()
            assert (bucket_prefix / shard.filename).is_file()

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
                generate(spec, tmp_path, [])

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
        tmp_path: Path,
    ) -> None:
        """If the renderer exits 0 but never wrote the expected shard file, fail loudly.

        Catches a generator bug at the rendering boundary instead of letting it surface as a less-
        direct rclone "source not found" further down the pipeline.

        :param fake_r2_remote: Local-typed R2 remote — must remain empty since
            no shard file is written and rclone is therefore never invoked.
        :param spec: Fixture-provided ``DatasetSpec``.
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        """
        # Renderer-only side effect: return 0 without writing the shard file,
        # so the ``shard_path.is_file()`` guard raises before any rclone call.
        with patch(
            "synth_setter.cli.generate_dataset.subprocess.check_call",
            return_value=0,
        ):
            with pytest.raises(RuntimeError, match="did not write expected shard file"):
                generate(spec, tmp_path, [])

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
            generate(spec, tmp_path, [])
        patched_subprocess.assert_not_called()
        assert not (fake_r2_remote / spec.r2.bucket / spec.r2.prefix).exists()

    def test_run_defaults_to_single_worker_when_skypilot_env_absent(
        self,
        patched_subprocess: MagicMock,
        fake_r2_remote: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No partition env → rank=0/world=1, the lone worker renders every shard.

        Underwrites the dev-experience contract that ``synth-setter-generate-dataset`` works
        out of the box without exporting ``SYNTH_SETTER_WORKER_RANK`` / ``SYNTH_SETTER_NUM_WORKERS``.

        :param patched_subprocess: Subprocess dispatcher used to introspect renderer argv per shard.
        :param fake_r2_remote: Local-typed R2 remote — asserted to contain every shard.
        :param tmp_path: Pytest tmp dir used by ``_multi_shard_spec``.
        :param monkeypatch: Used to unset the rank/world env vars.
        """
        monkeypatch.delenv("SYNTH_SETTER_WORKER_RANK", raising=False)
        monkeypatch.delenv("SYNTH_SETTER_NUM_WORKERS", raising=False)
        spec = _multi_shard_spec(tmp_path, n=3)

        generate(spec, tmp_path, [])

        rendered_filenames = {
            Path(args[find_script_index(args) + 1]).name
            for args in _renderer_argv_lists(patched_subprocess)
        }
        assert rendered_filenames == {shard.filename for shard in spec.shards}
        bucket_prefix = fake_r2_remote / spec.r2.bucket / spec.r2.prefix
        for shard in spec.shards:
            assert (bucket_prefix / shard.filename).is_file()

    def test_run_raises_on_partial_partition_env(
        self,
        patched_subprocess: MagicMock,
        fake_r2_remote: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Only world set (rank dropped) → ValueError before any rclone or subprocess work.

        Partial partition env almost always means a launcher dropped half its env injection;
        silently coercing it to single-worker would duplicate every shard across every node
        (#763). The default-to-(0, 1) fallback is gated on BOTH vars being absent.

        :param patched_subprocess: Subprocess dispatcher; asserted never invoked.
        :param fake_r2_remote: Local-typed R2 remote — asserted empty.
        :param tmp_path: Pytest tmp dir used by ``_multi_shard_spec``.
        :param monkeypatch: Used to set world but unset rank.
        """
        monkeypatch.delenv("SYNTH_SETTER_WORKER_RANK", raising=False)
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "2")
        spec = _multi_shard_spec(tmp_path, n=3)

        with pytest.raises(ValueError) as excinfo:
            generate(spec, tmp_path, [])
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

        generate(spec, tmp_path, [])

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

        generate(spec, tmp_path, [])

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

        generate(spec, tmp_path, [])

        patched_subprocess.assert_not_called()
        assert not (fake_r2_remote / spec.r2.bucket / spec.r2.prefix).exists()

    # Skip-existing-shards — see #750.

    def test_run_skips_render_when_shard_already_in_r2(
        self,
        patched_subprocess: MagicMock,
        fake_r2_remote: Path,
        spec: DatasetSpec,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Object present (size > 0) → renderer is not invoked, shard upload is not attempted.

        :param patched_subprocess: Subprocess dispatcher; asserted never invoked.
        :param fake_r2_remote: Local-typed R2 remote — asserted empty (the
            probe stub returns "present" without seeding an actual file).
        :param spec: Fixture-provided ``DatasetSpec``.
        :param monkeypatch: Used to override the probe to claim the shard is
            already in R2.
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        """
        monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda *_a, **_k: 12345)

        generate(spec, tmp_path, [])

        patched_subprocess.assert_not_called()
        assert not (fake_r2_remote / spec.r2.bucket / spec.r2.prefix).exists()

    def test_run_renders_when_object_absent(
        self,
        patched_subprocess: MagicMock,
        fake_r2_remote: Path,
        spec: DatasetSpec,
        tmp_path: Path,
    ) -> None:
        """Object absent (None) → render proceeds as before.

        Relies on the autouse ``_default_shard_absent_in_r2`` fixture's default of None.

        :param patched_subprocess: Subprocess dispatcher; renderer is asserted
            to fire exactly once.
        :param fake_r2_remote: Local-typed R2 remote — shard should land here.
        :param spec: Fixture-provided ``DatasetSpec``.
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        """
        generate(spec, tmp_path, [])

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
        tmp_path: Path,
    ) -> None:
        """Zero-byte object is treated as absent — defensive against half-uploaded objects.

        :param patched_subprocess: Subprocess dispatcher; renderer is asserted
            to fire exactly once.
        :param fake_r2_remote: Local-typed R2 remote — shard should land here.
        :param spec: Fixture-provided ``DatasetSpec``.
        :param monkeypatch: Used to override the probe to report 0 bytes.
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        """
        monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda *_a, **_k: 0)

        generate(spec, tmp_path, [])

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

        generate(spec, tmp_path, [])

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

        generate(spec, tmp_path, [])

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

        generate(spec, tmp_path, [])

        info_messages = [str(c.args[0]) for c in mock_logger.info.call_args_list]
        summary_lines = [m for m in info_messages if "rendered=" in m and "skipped=" in m]
        assert len(summary_lines) == 1, f"expected exactly one summary line, got: {info_messages}"
        assert "rendered=2" in summary_lines[0]
        assert "skipped=1" in summary_lines[0]

    @patch("synth_setter.cli.generate_dataset.logger")
    def test_run_logs_generation_speed_from_rendered_samples(
        self,
        mock_logger: MagicMock,
        patched_subprocess: MagicMock,  # noqa: ARG002
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Speed log line counts only ``rendered`` shards, not skipped ones.

        :param mock_logger: Captures the speed log line's content.
        :param patched_subprocess: Fixture-activation only.
        :param tmp_path: Used by ``_multi_shard_spec``.
        :param monkeypatch: Installs the probe stub that skips shard 0.
        """
        spec = _multi_shard_spec(tmp_path, n=3)

        def _present_only_for_shard_0(uri: str) -> int | None:
            return 9999 if uri.endswith("shard-000000.h5") else None

        monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", _present_only_for_shard_0)

        generate(spec, tmp_path, [])

        info_messages = [str(c.args[0]) for c in mock_logger.info.call_args_list]
        speed_lines = [m for m in info_messages if "generation speed:" in m]
        assert len(speed_lines) == 1, f"expected one speed line, got: {info_messages}"
        expected_samples = 2 * spec.render.samples_per_shard
        assert f"{expected_samples} samples" in speed_lines[0]
        assert "samples/s" in speed_lines[0]

    def test_run_probe_failure_propagates(
        self,
        patched_subprocess: MagicMock,
        spec: DatasetSpec,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """A non-zero rclone exit during the probe propagates as CalledProcessError.

        :param patched_subprocess: Subprocess dispatcher; asserted never
            invoked (the probe failure raises before any render/upload).
        :param spec: Fixture-provided ``DatasetSpec``.
        :param monkeypatch: Used to install the raising probe stub.
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        """

        def _raise(*_a: object, **_k: object) -> None:
            raise subprocess.CalledProcessError(1, ["rclone", "lsf"])

        monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", _raise)

        with pytest.raises(subprocess.CalledProcessError):
            generate(spec, tmp_path, [])

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
            generate(spec, tmp_path, [])

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
            generate(spec, tmp_path, [])

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
                generate(spec, tmp_path, [])

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
                generate(spec, tmp_path, [])

        assert renderer_calls == 3
        assert not (fake_r2_remote / spec.r2.bucket / spec.r2.prefix).exists()

    def test_shard_lands_in_work_dir_before_upload(
        self,
        patched_subprocess: MagicMock,  # noqa: ARG002
        spec: DatasetSpec,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Caller-supplied ``work_dir`` hosts the rendered shard at the upload boundary.

        Stubs ``_rclone_copy`` to snapshot the source path and its on-disk
        existence at the upload moment — proving the shard landed at
        ``work_dir / shard.filename`` before the upload runs.

        :param patched_subprocess: Fixture-activation only; renderer side of the
            dispatcher materializes the shard file into ``work_dir``.
        :param spec: Fixture-provided single-shard ``DatasetSpec``.
        :param tmp_path: Caller-supplied work_dir for this run.
        :param monkeypatch: Used to install the rclone-stub that captures the
            shard path + existence at the upload moment.
        """
        captured: dict[str, object] = {}

        def _capture_src(src: str, _dest: str) -> None:
            src_path = Path(src)
            captured["src"] = src_path
            captured["existed_at_upload"] = src_path.is_file()

        monkeypatch.setattr("synth_setter.cli.generate_dataset._rclone_copy", _capture_src)

        generate(spec, tmp_path, [])

        assert captured["src"] == tmp_path / spec.shards[0].filename
        assert captured["existed_at_upload"] is True

    def test_provenance_not_stamped_when_no_wandb_logger(
        self,
        patched_subprocess: MagicMock,  # noqa: ARG002
        spec: DatasetSpec,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``log_wandb_provenance`` is skipped when ``loggers`` owns no ``WandbLogger``.

        Provenance mutates the process-global ``wandb.run``; gating the call on a
        locally-owned ``WandbLogger`` keeps an empty-logger run from stamping a
        foreign in-process run, mirroring the ``_close_loggers`` ownership guard.

        :param patched_subprocess: Fixture-activation only; the renderer
            materializes the shard so the run reaches its summary.
        :param spec: Fixture-provided single-shard ``DatasetSpec``.
        :param tmp_path: Caller-supplied work_dir for ``generate()``.
        :param monkeypatch: Installs the ``log_wandb_provenance`` spy.
        """
        provenance = MagicMock()
        monkeypatch.setattr("synth_setter.cli.generate_dataset.log_wandb_provenance", provenance)

        generate(spec, tmp_path, [])

        provenance.assert_not_called()


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

    def test_copy_dataset_root_absent_when_unset(self, spec: DatasetSpec) -> None:
        """No ``--copy_dataset_root`` flag is emitted when the spec has no copy source.

        :param spec: Single-shard spec fixture with no ``copy_dataset_root``.
        """
        args = build_generate_args(spec, spec.shards[0], Path("out"))

        assert "--copy_dataset_root" not in args

    def test_copy_dataset_root_forwarded_when_set(self, tmp_path: Path) -> None:
        """``spec.copy_dataset_root`` is forwarded as a ``--copy_dataset_root`` flag.

        :param tmp_path: Pytest temp dir pinning the spec's plugin/output paths.
        """
        spec = DatasetSpec(
            **_base_spec_kwargs(tmp_path, copy_dataset_root="/data/source")  # type: ignore[arg-type]
        )

        args = build_generate_args(spec, spec.shards[0], Path("out"))

        flag_idx = args.index("--copy_dataset_root")
        assert args[flag_idx + 1] == "/data/source"


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
        cfg_dict["datamodule"] = {"sample_rate": 44100}
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


# PROJECT_ROOT-bootstrap behavior is exercised end-to-end by tests/pipeline/configs/
# test_experiment_yamls.py — those tests fail with an InterpolationResolutionError if the
# launcher's import-time `operator_workspace()` ever stops setting PROJECT_ROOT.


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
    def _set_default_skypilot_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Pin single-worker rank/world + isolate Hydra's per-run dir to ``tmp_path``.

        ``@hydra.main`` resolves ``${paths.log_dir}`` from ``${oc.env:PROJECT_ROOT}``;
        redirecting PROJECT_ROOT keeps the per-run dir under the test tree.

        :param monkeypatch: Pytest fixture used to set env vars.
        :param tmp_path: Per-test tmp dir hosting PROJECT_ROOT.
        """
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")
        monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))

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
        """``compute_template=null`` calls ``generate(spec, work_dir, loggers)`` inline.

        Dispatch (``dispatch_via_skypilot``) is never reached on this branch.

        :param monkeypatch: Pytest fixture used to patch argv and module functions.
        """
        import synth_setter.cli.generate_dataset as gd
        import synth_setter.pipeline.skypilot_launch as sl

        # Use a real experiment so cfg.skypilot_launch resolves; override the plugin path
        # to the test VST3 so generate() — which we replace below — sees the right spec shape.
        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
        ]
        monkeypatch.setattr("sys.argv", argv)

        recorded: dict[str, object] = {}

        def _fake_run(spec: object, _work_dir: object, _loggers: object) -> None:
            recorded["spec"] = spec

        def _dispatch_must_not_fire(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("dispatch_via_skypilot must not be called on the local branch")

        monkeypatch.setattr(gd, "generate", _fake_run)
        monkeypatch.setattr(sl, "dispatch_via_skypilot", _dispatch_must_not_fire)

        gd.main()

        spec = recorded.get("spec")
        assert isinstance(spec, DatasetSpec)
        assert spec.render.plugin_path == str(TEST_PLUGIN_VST3)

    def test_local_run_applies_extras_writing_tags_and_config_tree(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``main()`` runs ``extras(cfg)`` before generating, materializing its artifacts.

        ``dataset.yaml`` composes ``extras: default`` (``enforce_tags`` +
        ``print_config`` true) and a non-empty ``tags``, so ``extras(cfg)``
        exports ``tags.log`` and ``config_tree.log`` to ``cfg.paths.output_dir``.
        Asserting those files exist verifies the entrypoint applied extras via
        its observable side effects rather than mocking the call.

        :param monkeypatch: Pytest fixture used to patch argv + ``generate``.
        """
        import synth_setter.cli.generate_dataset as gd

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
        ]
        monkeypatch.setattr("sys.argv", argv)

        recorded: dict[str, Path] = {}

        def _fake_run(_spec: object, work_dir: Path, _loggers: object) -> None:
            recorded["work_dir"] = work_dir

        monkeypatch.setattr(gd, "generate", _fake_run)

        gd.main()

        output_dir = recorded["work_dir"]
        for artifact in ("tags.log", "config_tree.log"):
            path = output_dir / artifact
            assert path.is_file(), f"extras did not write {artifact}"
            assert path.stat().st_size > 0, f"{artifact} is empty"

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

        def _fake_dispatch(sky_cfg: object) -> None:
            recorded["sky_cfg"] = sky_cfg

        monkeypatch.setattr(sl, "dispatch_via_skypilot", _fake_dispatch)

        def _run_must_not_fire(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("generate must not be called on the dispatch branch")

        monkeypatch.setattr(gd, "generate", _run_must_not_fire)

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
        guard runs). ``HYDRA_FULL_ERROR=1`` makes ``@hydra.main`` re-raise the launcher-side
        ``ValueError`` instead of converting it to ``SystemExit(1)``, so the assertion pins the
        launcher contract directly rather than coupling to Hydra's error-handler formatting.

        :param monkeypatch: Pytest fixture used to set ``sys.argv`` and ``HYDRA_FULL_ERROR``.
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
        monkeypatch.setenv("HYDRA_FULL_ERROR", "1")

        def _run_must_not_fire(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("generate must not be called when cmd is rejected")

        def _dispatch_must_not_fire(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("dispatch_via_skypilot must not be called when cmd is rejected")

        monkeypatch.setattr(gd, "generate", _run_must_not_fire)
        monkeypatch.setattr(sl, "dispatch_via_skypilot", _dispatch_must_not_fire)

        with pytest.raises(ValueError, match="skypilot_launch.cmd is launcher-internal"):
            gd.main()

    def test_main_finalize_inline_true_invokes_finalize_from_spec(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """finalize_inline=true on the local-run branch invokes finalize_from_spec.

        Composes a real ``smoke-shard`` experiment with the new flag set,
        stubs ``generate`` to a no-op, and replaces ``finalize_from_spec``
        with a mock so the test pins the wire (call + spec identity)
        without needing real rclone against a finalize-shaped remote. The
        end-to-end marker upload is already covered by the Phase 1
        ``test_finalize_from_spec_uploads_stats_then_marker_at_canonical_uris``
        sibling test.

        :param monkeypatch: Pytest fixture used to patch argv +
            ``generate`` + ``finalize_from_spec``.
        """
        import synth_setter.cli.generate_dataset as gd

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
            "finalize_inline=true",
        ]
        monkeypatch.setattr("sys.argv", argv)

        captured: dict[str, object] = {}

        def _capture_spec(spec: object, _work_dir: Path, _loggers: object) -> None:
            captured["spec"] = spec

        monkeypatch.setattr(gd, "generate", _capture_spec)
        finalize_mock = MagicMock()
        monkeypatch.setattr(gd, "finalize_from_spec", finalize_mock)

        gd.main()

        finalize_mock.assert_called_once()
        called_spec, called_work_dir = finalize_mock.call_args[0]
        assert isinstance(called_spec, DatasetSpec)
        assert called_spec is captured["spec"]
        assert isinstance(called_work_dir, Path)

    def test_main_finalize_inline_default_false_skips_finalize(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Default ``finalize_inline=false`` leaves the existing local-run shape unchanged.

        Pins the opt-in invariant — no finalize fires when the operator
        omits the override.

        :param monkeypatch: Pytest fixture used to patch argv +
            ``generate`` + ``finalize_from_spec``.
        """
        import synth_setter.cli.generate_dataset as gd

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
        ]
        monkeypatch.setattr("sys.argv", argv)
        monkeypatch.setattr(gd, "generate", lambda _spec, _work_dir, _loggers: None)
        finalize_mock = MagicMock()
        monkeypatch.setattr(gd, "finalize_from_spec", finalize_mock)

        gd.main()

        finalize_mock.assert_not_called()

    @patch("synth_setter.cli.generate_dataset.logger")
    def test_main_finalize_inline_ignored_in_dispatch_branch(
        self,
        mock_logger: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """finalize_inline=true is ignored (with INFO log) when dispatching to SkyPilot.

        SkyPilot delegation hands the run to a worker pod; finalize must
        run out-of-band via the finalize-dataset workflow rather than fire
        from the launcher process. Pins both halves: ``finalize_from_spec``
        is not called, and an INFO log fires (wording unpinned).

        :param mock_logger: Patched ``generate_dataset.logger`` — the
            established loguru capture pattern in this file.
        :param monkeypatch: Pytest fixture used to patch argv + dispatch +
            ``finalize_from_spec`` (asserted unreached).
        :param tmp_path: Pytest fixture providing a fresh test directory for
            the minimal compute template.
        """
        import synth_setter.cli.generate_dataset as gd
        import synth_setter.pipeline.skypilot_launch as sl

        template = tmp_path / "template.yaml"
        template.write_text("resources:\n  cloud: runpod\nenvs:\n  X: ''\n")
        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
            f"skypilot_launch.compute_template={template}",
            "finalize_inline=true",
        ]
        monkeypatch.setattr("sys.argv", argv)
        monkeypatch.setattr(sl, "dispatch_via_skypilot", lambda *_a, **_k: None)
        monkeypatch.setattr(
            gd,
            "generate",
            lambda *_a, **_k: pytest.fail("generate must not fire on dispatch branch"),
        )
        finalize_mock = MagicMock()
        monkeypatch.setattr(gd, "finalize_from_spec", finalize_mock)

        gd.main()

        finalize_mock.assert_not_called()
        # State assertion above is the contract. The log check matches stable
        # tokens (knob name + "ignored") so a reworded message survives, but a
        # vacuous ``assert_called`` would not — ``main`` always emits INFO logs.
        info_messages = [str(c.args[0]) for c in mock_logger.info.call_args_list]
        ignored_lines = [m for m in info_messages if "finalize_inline=" in m and "ignored" in m]
        assert len(ignored_lines) == 1, (
            f"expected one INFO log marking the override ignored; got: {info_messages!r}"
        )

    def test_main_oracle_eval_inline_true_invokes_subprocess(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """oracle_eval_inline=true fires the eval subprocess once per split.

        Asserts the eval helper fires once per split, reading data **in place**
        from ``cfg.paths.output_dir`` (no download), with each Hydra run dir
        isolated under ``output_dir/oracle_eval/<split>/<run_id>/``.

        :param monkeypatch: Patches argv + the three module-level seams.
        """
        import synth_setter.cli.generate_dataset as gd

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
            "finalize_inline=true",
            "oracle_eval_inline=true",
            # Override smoke-shard's [12, 0, 0] — the zero-size guard rejects
            # train_val_test_sizes with any zero split for oracle_eval_inline.
            "train_val_test_sizes=[12, 4, 4]",
        ]
        monkeypatch.setattr("sys.argv", argv)
        monkeypatch.setattr(gd, "generate", lambda _spec, _work_dir, _loggers: None)
        monkeypatch.setattr(gd, "finalize_from_spec", MagicMock())
        oracle_mock = MagicMock()
        monkeypatch.setattr(gd, "_run_oracle_eval_subprocess", oracle_mock)

        # Capture the resolved output_dir so the eval's dataset_root can be
        # pinned to the exact dir generate+finalize wrote the shards into.
        observed: dict[str, object] = {}
        real_spec_from_cfg = gd.spec_from_cfg

        def _capture_output_dir(cfg: object) -> DatasetSpec:
            observed["output_dir"] = Path(cfg.paths.output_dir)  # type: ignore[attr-defined]
            observed["num_workers"] = cfg.datamodule.num_workers  # type: ignore[attr-defined]
            return real_spec_from_cfg(cfg)  # type: ignore[arg-type]

        monkeypatch.setattr(gd, "spec_from_cfg", _capture_output_dir)

        gd.main()

        # One invocation per split.
        assert oracle_mock.call_count == 3
        output_dir = observed["output_dir"]
        assert isinstance(output_dir, Path)
        splits = ("train", "val", "test")
        split_h5s = ("train.h5", "val.h5", "test.h5")
        # test stays bare ``audio/*``; train/val are namespaced so the shared
        # wandb run keeps one summary key per split instead of overwriting.
        split_prefixes = ("train/", "val/", "")
        for call, split, split_h5, prefix in zip(
            oracle_mock.call_args_list, splits, split_h5s, split_prefixes
        ):
            dataset_root, run_dir, _run_id = call[0]
            # The whole generation RenderConfig flows through (keyword-only) so
            # the eval re-renders through the same spec; smoke-shard is surge_simple.
            render_arg = call.kwargs["render"]
            assert render_arg.param_spec_name == "surge_simple"
            # The eval inherits the generate run's datamodule worker count verbatim,
            # so a Darwin override (num_workers=0) reaches the predict DataLoader.
            assert call.kwargs["num_workers"] == observed["num_workers"]
            assert render_arg.preset_path == "presets/surge-simple.vstpreset"
            # plugin_path is the TEST_PLUGIN_VST3 this test overrode at generation —
            # proving a non-default plugin flows through to the eval re-render.
            assert render_arg.plugin_path == str(TEST_PLUGIN_VST3)
            # The eval reads in place from the Hydra output_dir where the shards and
            # VDS splits already live — not a downloaded copy under oracle_eval/.
            assert dataset_root == output_dir
            # predict_file targets this split's HDF5.
            assert call.kwargs["predict_file"] == output_dir / split_h5
            # Run dir: oracle_eval/<split>/<run_id>
            assert run_dir.parent.parent.name == "oracle_eval", (
                f"eval run dir should land under "
                f"<output_dir>/oracle_eval/<split>/<run_id>/; got {run_dir!r}"
            )
            assert run_dir.parent.name == split
            assert run_dir.parent.parent.parent == dataset_root
            assert call.kwargs["metric_prefix"] == prefix

    def test_run_oracle_eval_subprocess_builds_expected_argv(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        spec: DatasetSpec,
    ) -> None:
        """Calls ``synth_setter.cli.eval`` as a subprocess and pins the contract argv.

        Pins the load-bearing overrides (``experiment=surge/fake_oracle``,
        ``datamodule.dataset_root``, ``ckpt_path=null``, ``mode=predict``), the
        wandb-resume trio that routes the eval's ``audio/*`` metrics onto the
        generate run, and every render field passed through from the generation
        ``RenderConfig`` so the eval re-renders identically (here a surge_xt spec).
        Runs the helper directly so cfg-resolution noise can't mask an argv drift.

        :param monkeypatch: Patches the module's ``subprocess.run``.
        :param tmp_path: Roots the distinct dataset-root and eval run dirs.
        :param spec: Source of a valid ``RenderConfig`` to derive the eval render from.
        """
        import synth_setter.cli.generate_dataset as gd

        run_mock = MagicMock()
        monkeypatch.setattr(gd.subprocess, "run", run_mock)

        dataset_root = tmp_path / "data"
        dataset_root.mkdir()
        for name in ("train.h5", "val.h5", "test.h5", "stats.npz"):
            (dataset_root / name).touch()
        run_dir = tmp_path / "oracle_eval" / "some-run-id"
        render = spec.render.model_copy(
            update={
                "param_spec_name": "surge_xt",
                "preset_path": "presets/surge-base.vstpreset",
                "plugin_path": "plugins/Surge XT.vst3",
            }
        )
        predict_file = dataset_root / "test.h5"
        gd._run_oracle_eval_subprocess(
            dataset_root,
            run_dir,
            "some-run-id",
            render=render,
            num_workers=7,
            predict_file=predict_file,
        )

        run_mock.assert_called_once()
        called_argv = run_mock.call_args[0][0]
        assert "-m" in called_argv
        assert "synth_setter.cli.eval" in called_argv
        assert "experiment=surge/fake_oracle" in called_argv
        # dataset_root and run_dir are distinct: split virtual datasets are
        # read in place beside their shards; eval outputs land in run_dir.
        assert f"datamodule.dataset_root={dataset_root}" in called_argv
        assert f"hydra.run.dir={run_dir}" in called_argv
        assert dataset_root != run_dir
        assert "ckpt_path=null" in called_argv
        # The eval resumes the generate run rather than opening a fresh one, so
        # its audio/* metrics share the run id (logger=null crashed Hydra — see #1331).
        assert "logger=wandb" in called_argv
        # id exists in logger/wandb.yaml (plain override); resume is absent (+append).
        assert "logger.wandb.id=some-run-id" in called_argv
        assert "+logger.wandb.resume=must" in called_argv
        # render_vst=true re-renders predicted params; surge_simple supplies the
        # group structure, while every render field predict_vst_audio renders with
        # is overridden from the generation RenderConfig so the re-render matches it.
        assert "render=surge_simple" in called_argv
        assert "render.param_spec_name=surge_xt" in called_argv
        assert "render.preset_path=presets/surge-base.vstpreset" in called_argv
        assert "render.plugin_path=plugins/Surge XT.vst3" in called_argv
        assert f"render.sample_rate={render.sample_rate}" in called_argv
        assert f"render.channels={render.channels}" in called_argv
        assert f"render.velocity={render.velocity}" in called_argv
        assert f"render.signal_duration_seconds={render.signal_duration_seconds}" in called_argv
        # batch_size=1 keeps the smoke-sized test split (4 samples) from
        # flooring to zero batches under the 128 default — see #1331.
        assert "datamodule.batch_size=1" in called_argv
        # Sentinel 7 (no config default) proves the value is forwarded, not hardcoded.
        assert "datamodule.num_workers=7" in called_argv
        assert "mode=predict" in called_argv
        # predict_file routes the datamodule to this split's HDF5.
        assert f"datamodule.predict_file={predict_file}" in called_argv
        # Default (test split) carries no prefix override: its keys stay bare
        # ``audio/*`` so existing sweeps/dashboards keep working.
        assert not any(a.startswith("+evaluation.metric_prefix=") for a in called_argv)

    def test_run_oracle_eval_subprocess_metric_prefix_adds_override(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        spec: DatasetSpec,
    ) -> None:
        """A non-empty ``metric_prefix`` appends the eval override that namespaces audio keys.

        The override routes through to ``cfg.evaluation.metric_prefix`` in the
        eval subprocess, which prepends it to every ``audio/*`` key so the
        per-split passes don't overwrite each other on the shared wandb run.

        :param monkeypatch: Patches the module's ``subprocess.run``.
        :param tmp_path: Roots the dataset-root and eval run dirs.
        :param spec: Source of a valid ``RenderConfig``.
        """
        import synth_setter.cli.generate_dataset as gd

        run_mock = MagicMock()
        monkeypatch.setattr(gd.subprocess, "run", run_mock)

        dataset_root = tmp_path / "data"
        dataset_root.mkdir()
        for name in ("train.h5", "val.h5", "test.h5", "stats.npz"):
            (dataset_root / name).touch()
        predict_file = dataset_root / "train.h5"
        gd._run_oracle_eval_subprocess(
            dataset_root,
            tmp_path / "oracle_eval" / "train" / "some-run-id",
            "some-run-id",
            render=spec.render,
            num_workers=0,
            predict_file=predict_file,
            metric_prefix="train/",
        )

        called_argv = run_mock.call_args[0][0]
        # ``+`` appends the key: it is absent from eval.yaml's evaluation group.
        assert "+evaluation.metric_prefix=train/" in called_argv

    def test_run_oracle_eval_subprocess_missing_local_artifacts_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        spec: DatasetSpec,
    ) -> None:
        """Unpopulated ``dataset_root`` ⇒ clear ``FileNotFoundError``, no eval subprocess.

        ``finalize_from_spec`` short-circuits when R2 already holds the
        ``dataset.complete`` marker, leaving ``output_dir`` without the splits
        on a resume; the preflight turns the downstream low-signal HDF5 read
        error into an actionable one before shelling out.

        :param monkeypatch: Patches ``subprocess.run`` to assert it never fires.
        :param tmp_path: Empty stand-in for an unpopulated ``output_dir``.
        :param spec: Source of a valid ``RenderConfig`` for the call signature.
        """
        import synth_setter.cli.generate_dataset as gd

        run_mock = MagicMock()
        monkeypatch.setattr(gd.subprocess, "run", run_mock)

        with pytest.raises(FileNotFoundError, match=r"test\.h5"):
            gd._run_oracle_eval_subprocess(
                tmp_path,
                tmp_path / "oracle_eval" / "test" / "rid",
                "rid",
                render=spec.render,
                num_workers=0,
                predict_file=tmp_path / "test.h5",
            )

        run_mock.assert_not_called()

    def test_run_oracle_eval_subprocess_missing_predict_file_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        spec: DatasetSpec,
    ) -> None:
        """Non-existent ``predict_file`` ⇒ ``FileNotFoundError`` before subprocess.

        All required artifacts are present in ``dataset_root`` so the existing
        preflight passes; the ``predict_file``-specific check then catches the
        absent path before shelling out.

        :param monkeypatch: Patches ``subprocess.run`` to assert it never fires.
        :param tmp_path: Roots the dataset dir and a missing predict path.
        :param spec: Source of a valid ``RenderConfig`` for the call signature.
        """
        import synth_setter.cli.generate_dataset as gd

        run_mock = MagicMock()
        monkeypatch.setattr(gd.subprocess, "run", run_mock)

        dataset_root = tmp_path / "data"
        dataset_root.mkdir()
        for name in ("train.h5", "val.h5", "test.h5", "stats.npz"):
            (dataset_root / name).touch()

        with pytest.raises(FileNotFoundError, match=r"predict_file"):
            gd._run_oracle_eval_subprocess(
                dataset_root,
                tmp_path / "oracle_eval" / "test" / "rid",
                "rid",
                render=spec.render,
                num_workers=0,
                predict_file=tmp_path / "nonexistent_split.h5",
            )

        run_mock.assert_not_called()

    def test_main_oracle_eval_inline_default_false_skips(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Opt-in invariant: default false ⇒ the eval subprocess never fires.

        :param monkeypatch: Patches argv + the three module-level seams.
        """
        import synth_setter.cli.generate_dataset as gd

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
            "finalize_inline=true",
        ]
        monkeypatch.setattr("sys.argv", argv)
        monkeypatch.setattr(gd, "generate", lambda _spec, _work_dir, _loggers: None)
        monkeypatch.setattr(gd, "finalize_from_spec", MagicMock())
        oracle_mock = MagicMock()
        monkeypatch.setattr(gd, "_run_oracle_eval_subprocess", oracle_mock)

        gd.main()

        oracle_mock.assert_not_called()

    def test_main_always_on_with_render_reload_skips_with_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``gui_toggle_cadence=always_on`` + ``plugin_reload_cadence=render`` is a no-op skip.

        The render arm of a cadence grid sweep hits this schema-invalid cell; ``main``
        logs a warning and returns before building the spec (which would otherwise raise
        in ``RenderConfig``), so the wandb trial completes instead of crashing.

        :param monkeypatch: Patches argv, the warning sink, and ``generate``.
        """
        import synth_setter.cli.generate_dataset as gd

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
            "render.plugin_reload_cadence=render",
            "render.gui_toggle_cadence=always_on",
        ]
        monkeypatch.setattr("sys.argv", argv)
        warn_mock = MagicMock()
        monkeypatch.setattr(gd.logger, "warning", warn_mock)
        generate_mock = MagicMock()
        monkeypatch.setattr(gd, "generate", generate_mock)

        gd.main()

        # Skipped before generation, with a warning (exact wording unpinned).
        generate_mock.assert_not_called()
        warn_mock.assert_called_once()

    def test_main_oracle_eval_inline_requires_finalize_inline(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``oracle_eval_inline=true`` without ``finalize_inline=true`` raises pre-``generate()``.

        :param monkeypatch: Patches argv + the three seams the test asserts are unreached.
        """
        import synth_setter.cli.generate_dataset as gd

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
            "oracle_eval_inline=true",
        ]
        monkeypatch.setattr("sys.argv", argv)
        monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
        generate_mock = MagicMock()
        finalize_mock = MagicMock()
        oracle_mock = MagicMock()
        monkeypatch.setattr(gd, "generate", generate_mock)
        monkeypatch.setattr(gd, "finalize_from_spec", finalize_mock)
        monkeypatch.setattr(gd, "_run_oracle_eval_subprocess", oracle_mock)

        with pytest.raises(ValueError, match="requires finalize_inline=true"):
            gd.main()
        generate_mock.assert_not_called()
        finalize_mock.assert_not_called()
        oracle_mock.assert_not_called()

    def test_main_oracle_eval_inline_rejects_wds(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``oracle_eval_inline=true`` + ``output_format=wds`` raises before ``generate()``.

        Eval reads HDF5 splits; WDS shards aren't consumed by the same loader.

        :param monkeypatch: Patches argv + the three seams the test asserts are unreached.
        """
        import synth_setter.cli.generate_dataset as gd

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
            "finalize_inline=true",
            "oracle_eval_inline=true",
            "output_format=wds",
        ]
        monkeypatch.setattr("sys.argv", argv)
        monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
        generate_mock = MagicMock()
        finalize_mock = MagicMock()
        oracle_mock = MagicMock()
        monkeypatch.setattr(gd, "generate", generate_mock)
        monkeypatch.setattr(gd, "finalize_from_spec", finalize_mock)
        monkeypatch.setattr(gd, "_run_oracle_eval_subprocess", oracle_mock)

        with pytest.raises(ValueError, match="only supports output_format=hdf5"):
            gd.main()
        generate_mock.assert_not_called()
        finalize_mock.assert_not_called()
        oracle_mock.assert_not_called()

    def test_main_oracle_eval_inline_rejects_zero_size_split(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fail-fast guard: ``oracle_eval_inline=true`` rejects ``[N, 0, 0]``-style sizes.

        ``SurgeDataModule.setup()`` opens train/val/test ``.h5`` unconditionally
        regardless of stage, so any zero-size split would FileNotFoundError
        deep inside Lightning. The launcher catches the misconfig up front.

        :param monkeypatch: Patches argv and the ``generate`` / ``finalize_from_spec``
            / oracle-eval seams; the test asserts none of them is reached.
        """
        import synth_setter.cli.generate_dataset as gd

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
            "finalize_inline=true",
            "oracle_eval_inline=true",
            # smoke-shard now defaults to [4, 4, 4]; pin a zero split to exercise the guard.
            "train_val_test_sizes=[4,0,0]",
        ]
        monkeypatch.setattr("sys.argv", argv)
        monkeypatch.setenv("HYDRA_FULL_ERROR", "1")
        generate_mock = MagicMock()
        finalize_mock = MagicMock()
        oracle_mock = MagicMock()
        monkeypatch.setattr(gd, "generate", generate_mock)
        monkeypatch.setattr(gd, "finalize_from_spec", finalize_mock)
        monkeypatch.setattr(gd, "_run_oracle_eval_subprocess", oracle_mock)

        with pytest.raises(ValueError, match="train_val_test_sizes > 0"):
            gd.main()
        generate_mock.assert_not_called()
        finalize_mock.assert_not_called()
        oracle_mock.assert_not_called()

    @patch("synth_setter.cli.generate_dataset.logger")
    def test_main_oracle_eval_inline_ignored_in_dispatch_branch(
        self,
        mock_logger: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Dispatch branch: ``oracle_eval_inline=true`` is logged-and-ignored, not raised.

        SkyPilot hands the run to a worker pod; oracle eval runs out-of-band
        via its own workflow. Asserts no eval subprocess fires and the INFO
        log mentions the override was ignored.

        :param mock_logger: Patched ``generate_dataset.logger`` — the
            established loguru capture pattern in this file.
        :param monkeypatch: Patches argv + dispatch + the oracle-eval seam.
        :param tmp_path: Holds the minimal compute template the dispatch
            branch reads from disk.
        """
        import synth_setter.cli.generate_dataset as gd
        import synth_setter.pipeline.skypilot_launch as sl

        template = tmp_path / "template.yaml"
        template.write_text("resources:\n  cloud: runpod\nenvs:\n  X: ''\n")
        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
            f"skypilot_launch.compute_template={template}",
            "oracle_eval_inline=true",
        ]
        monkeypatch.setattr("sys.argv", argv)
        monkeypatch.setattr(sl, "dispatch_via_skypilot", lambda *_a, **_k: None)
        monkeypatch.setattr(
            gd,
            "generate",
            lambda *_a, **_k: pytest.fail("generate must not fire on dispatch branch"),
        )
        oracle_mock = MagicMock()
        monkeypatch.setattr(gd, "_run_oracle_eval_subprocess", oracle_mock)

        gd.main()

        oracle_mock.assert_not_called()
        info_messages = [str(c.args[0]) for c in mock_logger.info.call_args_list]
        ignored_lines = [
            m for m in info_messages if "oracle_eval_inline=True" in m and "ignored" in m
        ]
        assert len(ignored_lines) == 1, (
            f"expected exactly one INFO log mentioning 'oracle_eval_inline=True' + 'ignored'; "
            f"got messages: {info_messages!r}"
        )


def _write_vds_split_with_shard(data_dir: Path) -> np.ndarray:
    """Write a real shard + a VDS ``test.h5`` + ``stats.npz`` into ``data_dir``.

    Follows the basename-VDS contract of
    :func:`synth_setter.pipeline.data.reshard._write_split`: the split's
    ``audio`` / ``param_array`` are HDF5 virtual datasets whose source is the
    shard referenced **by basename**, so they resolve only when the shard sits
    beside the split — the on-disk shape the inline oracle-eval reads (#1396).

    :param data_dir: Destination dir; receives ``shard-000000.h5``,
        ``test.h5``, and ``stats.npz``.
    :returns: The shard's ``audio`` array (``float16``) so callers can assert
        the read resolves to the real bytes.
    """
    rows, channels, samples, num_params = 2, 2, 8, 4
    audio_dtype = DATASET_FIELD_DTYPES["audio"]
    param_dtype = DATASET_FIELD_DTYPES["param_array"]
    # Start at 1 so every value is distinct and nonzero (a fill-value zeros read
    # is then unmistakable) while staying in the field dtype — no scalar promotion.
    audio = np.arange(1, rows * channels * samples + 1, dtype=audio_dtype).reshape(
        rows, channels, samples
    )
    param = np.arange(1, rows * num_params + 1, dtype=param_dtype).reshape(rows, num_params)

    shard = data_dir / "shard-000000.h5"
    with h5py.File(shard, "w") as f:
        f.create_dataset("audio", data=audio)
        f.create_dataset("param_array", data=param)

    fields = (
        ("audio", audio_dtype, (channels, samples)),
        ("param_array", param_dtype, (num_params,)),
    )
    with h5py.File(data_dir / "test.h5", "w", libver="latest") as f:
        for key, dtype, tail in fields:
            layout = h5py.VirtualLayout(shape=(rows, *tail), dtype=dtype)
            layout[0:rows] = h5py.VirtualSource(shard.name, key, shape=(rows, *tail), dtype=dtype)
            f.create_virtual_dataset(key, layout)

    np.savez(
        data_dir / STATS_NPZ_FILENAME,
        mean=np.zeros(1, dtype=np.float32),
        std=np.ones(1, dtype=np.float32),
    )
    return audio


class TestInlineOracleEvalVdsInPlaceRead:
    """Behavioral proof for #1396: the inline eval reads the VDS splits in place.

    The finalized ``{split}.h5`` files are HDF5 virtual datasets referencing
    their source shards by basename, so ``mode=predict`` reads the audio dataset
    as fill-value zeros unless the shards sit beside the split. These tests drive
    the real predict read path (:class:`SurgeXTDataset` with ``read_audio=True``)
    to pin that the read resolves to real bytes only when co-located.
    """

    def test_predict_audio_read_vds_split_beside_shards_returns_shard_data(
        self,
        tmp_path: Path,
    ) -> None:
        """With the shard co-located, the predict-row audio read resolves to real bytes.

        Reads row ``1`` — the slice that crashed in production after row 0's
        fill-value zeros — through the same datamodule the eval uses.

        :param tmp_path: Holds the co-located shard, VDS split, and stats.
        """
        from synth_setter.data.surge_datamodule import SurgeXTDataset

        audio = _write_vds_split_with_shard(tmp_path)
        dataset = SurgeXTDataset(
            tmp_path / "test.h5",
            batch_size=1,
            ot=False,
            read_audio=True,
            read_mel=False,
            use_saved_mean_and_variance=True,
        )

        assert dataset.dataset_file is not None
        audio_ds = dataset.dataset_file["audio"]
        assert isinstance(audio_ds, h5py.Dataset) and audio_ds.is_virtual, (
            "split must be a virtual dataset"
        )
        audio_item = dataset[1]["audio"]
        assert audio_item is not None
        np.testing.assert_array_equal(audio_item.numpy(), audio[1:2].astype(np.float32))

    def test_predict_audio_read_vds_split_without_shards_returns_dangling_zeros(
        self,
        tmp_path: Path,
    ) -> None:
        """Without the shard beside it, the same read dangles to fill-value zeros.

        Reproduces the #1396 failure mode: copying only the split + stats away from the shards (as
        the old download did) makes the audio unreadable, so the read returns zeros instead of the
        shard bytes.

        :param tmp_path: Roots the populated source dir and the split-only copy.
        """
        from synth_setter.data.surge_datamodule import SurgeXTDataset

        source_dir = tmp_path / "source"
        source_dir.mkdir()
        audio = _write_vds_split_with_shard(source_dir)

        split_only = tmp_path / "split_only"
        split_only.mkdir()
        shutil.copy(source_dir / "test.h5", split_only / "test.h5")
        shutil.copy(source_dir / STATS_NPZ_FILENAME, split_only / STATS_NPZ_FILENAME)

        dataset = SurgeXTDataset(
            split_only / "test.h5",
            batch_size=1,
            ot=False,
            read_audio=True,
            read_mel=False,
            use_saved_mean_and_variance=True,
        )

        dangling_item = dataset[1]["audio"]
        assert dangling_item is not None
        dangling = dangling_item.numpy()
        assert np.array_equal(dangling, np.zeros_like(dangling)), (
            "dangling VDS read should fall back to fill-value zeros"
        )
        assert not np.array_equal(dangling, audio[1:2].astype(np.float32)), (
            "without co-located shards the read must not reach the real audio"
        )


class TestMainSpecPersistence:
    """``main()`` writes the local spec, loads R2 env, uploads the canonical spec on every path.

    The R2 upload is launcher-side and happens once per ``main()`` invocation:
    after the local write, before the local-run / dispatch branch is taken.
    Workers in the dispatch path no longer re-upload the spec (the worker's
    ``generate(spec, work_dir, loggers)`` writes shards only); the canonical R2 object
    exists before any worker boots.
    """

    @pytest.fixture(autouse=True)
    def _set_default_skypilot_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Pin single-worker rank/world + isolate Hydra's per-run dir to ``tmp_path``.

        ``@hydra.main`` resolves ``${paths.log_dir}`` from ``${oc.env:PROJECT_ROOT}``;
        redirecting PROJECT_ROOT keeps the per-run dir under the test tree.

        :param monkeypatch: Pytest fixture used to set env vars.
        :param tmp_path: Per-test tmp dir hosting PROJECT_ROOT.
        """
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")
        monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))

    @pytest.fixture(autouse=True)
    def _stub_run_and_spec_io(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub ``generate()``, the spec_io helpers, and ``r2_io.ensure_r2_env_loaded``.

        Tests assert via the module-level mocks ``gd.write_spec_locally``,
        ``gd.upload_spec``, and ``gd.r2_io.ensure_r2_env_loaded`` to keep test
        signatures stable for pydoclint.

        :param monkeypatch: Pytest fixture used to patch module-level callables.
        """
        import synth_setter.cli.generate_dataset as gd

        monkeypatch.setattr(gd, "generate", lambda _spec, _work_dir, _loggers: None)
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
        """``main()`` calls ``write_spec_locally`` with ``Path(cfg.paths.output_dir)``.

        Pinned by cross-reference: ``main()`` passes the same value to both
        ``write_spec_locally`` and ``generate()`` (the local-run shard
        scratch dir), so equality with the captured ``generate`` arg
        anchors the source without hard-coding the timestamped Hydra dir.

        :param monkeypatch: Pytest fixture used to patch ``sys.argv``.
        """
        import synth_setter.cli.generate_dataset as gd

        generate_mock = MagicMock(return_value=None)
        monkeypatch.setattr(gd, "generate", generate_mock)

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
        generate_mock.assert_called_once()
        _, generate_work_dir, _ = generate_mock.call_args[0]
        assert called_out == generate_work_dir

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

    def test_main_uploads_spec_with_r2_creds_present_in_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``upload_spec`` sees R2 creds in ``os.environ`` (set by ``ensure_r2_env_loaded``).

        Asserts the observable invariant — credentials are present in process
        env when the upload fires — rather than the internal call ORDER. A
        benign re-ordering that still loads creds before uploading passes; a
        regression that uploads before ``ensure_r2_env_loaded`` populates the
        env fails because the stub records an absent key.

        :param monkeypatch: Pytest fixture used to patch ``sys.argv``.
        """
        import synth_setter.cli.generate_dataset as gd
        from synth_setter.pipeline.r2_io import _SECRET_R2_ENV_KEYS

        probe_key = _SECRET_R2_ENV_KEYS[0]
        monkeypatch.delenv(probe_key, raising=False)

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
        ]
        monkeypatch.setattr("sys.argv", argv)

        def _load_creds(*_a: object, **_k: object) -> None:
            # setenv (not raw os.environ) so monkeypatch restores it on teardown.
            monkeypatch.setenv(probe_key, "stub-access-key-id")

        monkeypatch.setattr(gd.r2_io, "ensure_r2_env_loaded", _load_creds)

        creds_present_at_upload: dict[str, bool] = {}

        def _record_env(*_a: object, **_k: object) -> str:
            creds_present_at_upload["present"] = probe_key in os.environ
            return "r2://stub-bucket/stub-key/input_spec.json"

        monkeypatch.setattr("synth_setter.cli.generate_dataset.upload_spec", _record_env)

        gd.main()

        assert creds_present_at_upload.get("present") is True

    def test_dispatch_branch_passes_canonical_spec_uri_via_extra_envs(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``main()`` forwards ``spec.r2.input_spec_uri()`` via ``sky_cfg.extra_envs``.

        The canonical spec URI (with run prefix) lands in
        ``sky_cfg.extra_envs[WORKER_SPEC_URI_ENV]`` so each rank reads the same
        R2 object ``main()`` just uploaded.

        :param monkeypatch: Pytest fixture used to patch ``sys.argv`` + dispatch.
        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        import synth_setter.cli.generate_dataset as gd
        import synth_setter.pipeline.skypilot_launch as sl
        from synth_setter.pipeline.constants import WORKER_SPEC_URI_ENV

        template = self._write_minimal_template(tmp_path)
        monkeypatch.setattr("sys.argv", self._dispatch_argv(template))

        recorded: dict[str, object] = {}

        def _fake_dispatch(sky_cfg: object) -> None:
            recorded["sky_cfg"] = sky_cfg

        monkeypatch.setattr(sl, "dispatch_via_skypilot", _fake_dispatch)

        gd.main()

        sky_cfg = recorded["sky_cfg"]
        spec = gd.write_spec_locally.call_args[0][0]  # type: ignore[attr-defined]
        assert isinstance(spec, DatasetSpec)
        assert sky_cfg.extra_envs[WORKER_SPEC_URI_ENV] == spec.r2.input_spec_uri()  # type: ignore[attr-defined]

    def test_main_does_not_emit_spec_uri_sentinel(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``main()`` must not print the ``::synth-setter-spec-uri::`` marker on stdout.

        CI derives the URI via ``synth-setter-spec-uri`` (Hydra-compose) — see #1154.

        :param monkeypatch: Pytest fixture used to patch ``sys.argv`` + dispatch.
        :param tmp_path: Pytest fixture providing a fresh test directory.
        :param capsys: Pytest fixture capturing stdout/stderr.
        """
        import synth_setter.cli.generate_dataset as gd
        import synth_setter.pipeline.skypilot_launch as sl

        template = self._write_minimal_template(tmp_path)
        monkeypatch.setattr("sys.argv", self._dispatch_argv(template))
        monkeypatch.setattr(sl, "dispatch_via_skypilot", lambda *_a, **_k: None)

        gd.main()

        assert "::synth-setter-spec-uri::" not in capsys.readouterr().out

    def test_generate_dataset_pins_smoke_job_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """``main()`` pins the dataset-specific job-name stem before dispatching.

        The launcher is domain-neutral; the dataset-specific
        ``synth-setter-smoke-<task[:8]>`` stem lives on the caller side so the
        worker job name still encodes the task.

        :param monkeypatch: Pytest fixture used to patch ``sys.argv`` + dispatch.
        :param tmp_path: Pytest fixture providing a fresh test directory.
        """
        import synth_setter.cli.generate_dataset as gd
        import synth_setter.pipeline.skypilot_launch as sl

        template = self._write_minimal_template(tmp_path)
        monkeypatch.setattr("sys.argv", self._dispatch_argv(template))

        recorded: dict[str, object] = {}

        def _fake_dispatch(sky_cfg: object) -> None:
            recorded["sky_cfg"] = sky_cfg

        monkeypatch.setattr(sl, "dispatch_via_skypilot", _fake_dispatch)

        gd.main()

        sky_cfg = recorded["sky_cfg"]
        spec_call = gd.write_spec_locally.call_args[0][0]  # type: ignore[attr-defined]
        assert sky_cfg.job_name == gd._smoke_job_name(spec_call)  # type: ignore[attr-defined]


class TestMainHydraOutputDir:
    """``cfg.paths.output_dir`` resolves to Hydra's per-run dir inside ``main()``.

    Pins the @hydra.main decoration contract for the launcher entrypoint.
    """

    @pytest.fixture(autouse=True)
    def _isolate_hydra_output_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Redirect PROJECT_ROOT to tmp so Hydra writes the per-run dir under the test tree.

        :param monkeypatch: Pytest fixture used to override env vars.
        :param tmp_path: Per-test tmp dir hosting the synthetic checkout root.
        """
        monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")

    @pytest.fixture(autouse=True)
    def _stub_run_and_spec_io(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub the launcher's R2 + dispatch surface so main() runs without I/O.

        :param monkeypatch: Pytest fixture used to patch module-level callables.
        """
        import synth_setter.cli.generate_dataset as gd

        monkeypatch.setattr(gd, "generate", lambda _spec, _work_dir, _loggers: None)
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

    def test_main_resolves_output_dir_under_hydra_main(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Inside main(), cfg.paths.output_dir equals HydraConfig.get().runtime.output_dir.

        Pins the @hydra.main decoration contract: the per-run dir is supplied by
        Hydra runtime rather than pinned by the launcher to a hand-picked anchor.

        :param monkeypatch: Pytest fixture used to patch ``sys.argv`` + capture cfg.
        """
        from hydra.core.hydra_config import HydraConfig

        import synth_setter.cli.generate_dataset as gd

        argv = [
            "synth-setter-generate-dataset",
            "experiment=generate_dataset/smoke-shard",
            f"render.plugin_path={TEST_PLUGIN_VST3}",
        ]
        monkeypatch.setattr("sys.argv", argv)

        observed: dict[str, str] = {}
        real_spec_from_cfg = gd.spec_from_cfg

        def _capture_then_build(cfg: object) -> DatasetSpec:
            observed["output_dir"] = cfg.paths.output_dir  # type: ignore[attr-defined]
            observed["runtime_output_dir"] = HydraConfig.get().runtime.output_dir
            return real_spec_from_cfg(cfg)  # type: ignore[arg-type]

        monkeypatch.setattr(gd, "spec_from_cfg", _capture_then_build)

        gd.main()

        assert observed["output_dir"] == observed["runtime_output_dir"]


def test_smoke_job_name_rejects_unsafe_task_name() -> None:
    """``_smoke_job_name`` raises with a task-name-aware diagnostic on malformed task_name.

    Pins the dataset-aware error message that the launcher's
    ``_JOB_NAME_RE`` validator would otherwise surface without spec context.
    """
    from synth_setter.cli.generate_dataset import _smoke_job_name

    bad_spec = SimpleNamespace(task_name="bad.task.name")
    with pytest.raises(ValueError, match=r"fix spec.task_name or pin"):
        _smoke_job_name(bad_spec)  # type: ignore[arg-type]


class TestValidateCopySource:
    """``_validate_copy_source`` — the imperative shell around the preflight."""

    def test_no_copy_dataset_root_is_a_noop(self, spec: DatasetSpec) -> None:
        """A spec with no copy source skips the preflight entirely (no disk read).

        :param spec: Single-shard spec fixture with ``copy_dataset_root`` unset.
        """
        from synth_setter.cli.generate_dataset import _validate_copy_source

        _validate_copy_source(spec)  # no raise, no source dir needed

    def test_matching_source_spec_passes(self, tmp_path: Path) -> None:
        """A copy root holding a matching ``input_spec.json`` validates clean.

        :param tmp_path: Pytest tmp dir used as the synced copy source root.
        """
        from synth_setter.cli.generate_dataset import _validate_copy_source

        copy_root = tmp_path / "source"
        copy_root.mkdir()
        source = DatasetSpec(**_base_spec_kwargs(tmp_path))  # type: ignore[arg-type]
        (copy_root / INPUT_SPEC_FILENAME).write_text(source.model_dump_json())
        target = DatasetSpec(
            **_base_spec_kwargs(tmp_path, copy_dataset_root=str(copy_root))  # type: ignore[arg-type]
        )

        _validate_copy_source(target)  # no raise

    def test_missing_source_spec_raises_with_path(self, tmp_path: Path) -> None:
        """A copy root without an ``input_spec.json`` fails loudly, naming the file.

        :param tmp_path: Pytest tmp dir; the copy root is left without a spec.
        """
        from synth_setter.cli.generate_dataset import _validate_copy_source

        copy_root = tmp_path / "source"
        copy_root.mkdir()
        target = DatasetSpec(
            **_base_spec_kwargs(tmp_path, copy_dataset_root=str(copy_root))  # type: ignore[arg-type]
        )

        with pytest.raises(ValueError, match=INPUT_SPEC_FILENAME):
            _validate_copy_source(target)

    def test_mismatched_source_spec_raises(self, tmp_path: Path) -> None:
        """A source spec with a different ``param_spec_name`` is rejected at preflight.

        :param tmp_path: Pytest tmp dir used as the synced copy source root.
        """
        from synth_setter.cli.generate_dataset import _validate_copy_source

        copy_root = tmp_path / "source"
        copy_root.mkdir()
        source = DatasetSpec(
            **_base_spec_kwargs(
                tmp_path,
                render={
                    **_base_spec_kwargs(tmp_path)["render"],  # type: ignore[dict-item]
                    "param_spec_name": "surge_xt",
                },
            )  # type: ignore[arg-type]
        )
        (copy_root / INPUT_SPEC_FILENAME).write_text(source.model_dump_json())
        target = DatasetSpec(
            **_base_spec_kwargs(tmp_path, copy_dataset_root=str(copy_root))  # type: ignore[arg-type]
        )

        with pytest.raises(ValueError, match="param_spec_name"):
            _validate_copy_source(target)
