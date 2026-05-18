"""Tests for synth_setter.cli.generate_dataset — spec-driven run.

The entrypoint's public surface is a single ``run(spec)`` function that, for
each shard in ``spec.shards``, shells out to ``generate_vst_dataset.py`` to
render it, uploads to R2 at ``r2:{bucket}/{prefix}/``, and unlinks the local
file. The spec itself is uploaded by the launcher (see
``synth_setter.pipeline.skypilot_launch.upload_spec_to_r2``); the worker does
not re-upload it.

Tests monkeypatch ``_rclone_copy`` and ``subprocess.check_call`` and assert on
recorded call args + ordering.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from synth_setter.cli.generate_dataset import (
    VST_HEADLESS_WRAPPER,
    build_generate_args,
    run,
)
from synth_setter.pipeline.constants import INPUT_SPEC_FILENAME
from synth_setter.pipeline.schemas.spec import DatasetSpec, RenderConfig

# Reusable VST3 bundle with a real Contents/moduleinfo.json so
# extract_renderer_version (called by run) returns a deterministic version
# without loading any .so via pedalboard. Version inside is "1.0.0-test" — the
# specs built in this file pin renderer_version to the same string so the
# constraint check passes.
TEST_PLUGIN_VST3 = Path(__file__).resolve().parent.parent / "fixtures" / "TestPlugin.vst3"
TEST_PLUGIN_VERSION = "1.0.0-test"


def _find_script_index(args: list[str]) -> int:
    """Locate generate_vst_dataset.py in subprocess args, with a clear failure on miss.

    The args layout depends on platform — `[wrapper, python, script, output, ...]` on Linux
    versus `[python, script, output, ...]` elsewhere — so callers locate the script by name
    rather than fixed index.
    """
    for i, a in enumerate(args):
        if a.endswith("generate_vst_dataset.py"):
            return i
    raise AssertionError(f"generate_vst_dataset.py not found in subprocess args: {args}")


def _materialize_shard(args: list[str]) -> int:
    """subprocess.check_call side effect that writes the expected shard file.

    Mirrors the production contract: generate_vst_dataset.py exits 0 only after writing the
    HDF5 to its output path. Tests that don't supply this side effect would trip the
    `shard_path.is_file()` check in `_render_and_upload_shard`.
    """
    script_idx = _find_script_index(args)
    output_file = Path(args[script_idx + 1])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(b"")
    return 0


def _base_spec_kwargs(tmp_path: Path, **overrides: object) -> dict[str, object]:
    """Return valid DatasetSpec kwargs for direct construction."""
    kwargs: dict[str, object] = {
        "task_name": "test-dataset",
        "run_id": "test-dataset-20260328T120000000Z",
        "r2_prefix": "data/test-dataset/test-dataset-20260328T120000000Z/",
        "created_at": datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc),
        "git_sha": "a" * 40,
        "is_repo_dirty": False,
        "output_format": "hdf5",
        "train_val_test_sizes": [10000, 0, 0],
        "base_seed": 42,
        "r2_bucket": "intermediate-data",
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
# run — full flow orchestration
# ---------------------------------------------------------------------------


class TestRun:
    """Run() orchestrates per-shard generate → upload (no spec re-upload by the worker)."""

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

    @patch("synth_setter.cli.generate_dataset.subprocess.check_call")
    @patch("synth_setter.cli.generate_dataset._rclone_copy")
    def test_worker_does_not_re_upload_spec(  # noqa: DOC101,DOC103
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        spec: DatasetSpec,
    ) -> None:
        """Worker uploads shards only; the launcher owns the spec upload.

        Pinned to ensure the worker never re-uploads ``input_spec.json`` to
        ``{spec.r2_prefix}input_spec.json``. The launcher's
        ``upload_spec_to_r2`` writes the spec to
        ``skypilot-launcher-specs/<job_name>.json``; the worker-side copy
        served no consumer.
        """
        mock_check_call.side_effect = _materialize_shard
        run(spec)

        rclone_calls = mock_rclone.call_args_list
        # Exactly one upload: the shard. No spec upload.
        assert len(rclone_calls) == 1
        for src, _dest in (c[0] for c in rclone_calls):
            assert not src.endswith(INPUT_SPEC_FILENAME), (
                f"worker must not re-upload spec; got rclone src={src!r}"
            )

    @patch("synth_setter.cli.generate_dataset.subprocess.check_call")
    @patch("synth_setter.cli.generate_dataset._rclone_copy")
    def test_invokes_generate_vst_dataset_with_spec_derived_args(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        spec: DatasetSpec,
    ) -> None:
        """subprocess.check_call invokes generate_vst_dataset.py with spec-derived args."""
        mock_check_call.side_effect = _materialize_shard
        run(spec)

        mock_check_call.assert_called_once()
        args = mock_check_call.call_args[0][0]
        # args = [VST_HEADLESS_WRAPPER (linux only), python, generate_vst_dataset.py, ...]
        assert any("generate_vst_dataset.py" in a for a in args)
        assert str(spec.render.samples_per_shard) in args

    @patch("synth_setter.cli.generate_dataset.subprocess.check_call")
    @patch("synth_setter.cli.generate_dataset._rclone_copy")
    def test_shard_generation_runs_under_headless_vst_wrapper(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        spec: DatasetSpec,
    ) -> None:
        """Prefix the VST subprocess with ``run-linux-vst-headless.sh`` on Linux.

        X11 bootstrap lives at the audio-rendering boundary (this subprocess) so the
        docker_entrypoint click CLI can stay X11-agnostic — idle and passthrough modes don't pay
        the Xvfb startup cost. The wrapper is Linux-only (Xvfb is a Linux X11 server); on macOS and
        other platforms the generator is invoked directly without a wrapper prefix.
        """
        mock_check_call.side_effect = _materialize_shard
        run(spec)

        args = mock_check_call.call_args[0][0]
        if sys.platform == "linux":
            assert args[0] == VST_HEADLESS_WRAPPER
            assert args[2] == "src/synth_setter/data/vst/generate_vst_dataset.py"
        else:
            assert VST_HEADLESS_WRAPPER not in args
            assert args[1] == "src/synth_setter/data/vst/generate_vst_dataset.py"

    @patch("synth_setter.cli.generate_dataset.subprocess.check_call")
    @patch("synth_setter.cli.generate_dataset._rclone_copy")
    def test_uploads_shard_to_r2_after_generation(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        spec: DatasetSpec,
    ) -> None:
        """The single rclone call uploads the shard to r2:{bucket}/{prefix}/."""
        mock_check_call.side_effect = _materialize_shard
        run(spec)

        rclone_calls = mock_rclone.call_args_list
        assert len(rclone_calls) == 1
        shard_src, shard_dest = rclone_calls[0][0]
        assert "shard-000000.h5" in shard_src
        assert shard_dest == f"r2:{spec.r2_bucket}/{spec.r2_prefix}"

    @patch("synth_setter.cli.generate_dataset.subprocess.check_call")
    @patch("synth_setter.cli.generate_dataset._rclone_copy")
    def test_subprocess_failure_propagates(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        spec: DatasetSpec,
    ) -> None:
        """CalledProcessError from generate_vst_dataset propagates to caller."""
        mock_check_call.side_effect = subprocess.CalledProcessError(1, "generate_vst_dataset.py")

        with pytest.raises(subprocess.CalledProcessError):
            run(spec)

    @patch("synth_setter.cli.generate_dataset.subprocess.check_call")
    @patch("synth_setter.cli.generate_dataset._rclone_copy")
    def test_rclone_failure_propagates(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        spec: DatasetSpec,
    ) -> None:
        """CalledProcessError from rclone propagates to caller."""
        # The renderer must succeed so we reach the shard-upload rclone call.
        mock_check_call.side_effect = _materialize_shard
        mock_rclone.side_effect = subprocess.CalledProcessError(1, "rclone")

        with pytest.raises(subprocess.CalledProcessError):
            run(spec)

    @patch("synth_setter.cli.generate_dataset.subprocess.check_call")
    @patch("synth_setter.cli.generate_dataset._rclone_copy")
    def test_run_with_three_shards_renders_each_shard(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Multi-shard run invokes generate_vst_dataset.py once per shard, in order."""
        spec = _multi_shard_spec(tmp_path, n=3)
        mock_check_call.side_effect = _materialize_shard

        run(spec)

        assert mock_check_call.call_count == 3
        rendered_filenames = []
        for call in mock_check_call.call_args_list:
            args = call[0][0]
            output_file = args[_find_script_index(args) + 1]
            rendered_filenames.append(Path(output_file).name)
        assert rendered_filenames == [s.filename for s in spec.shards]

    @patch("synth_setter.cli.generate_dataset.subprocess.check_call")
    @patch("synth_setter.cli.generate_dataset._rclone_copy")
    def test_multi_shard_run_uploads_once_per_shard_and_no_spec(  # noqa: DOC101,DOC103
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Every rclone call is a shard upload; the worker never re-uploads the spec."""
        spec = _multi_shard_spec(tmp_path, n=3)
        mock_check_call.side_effect = _materialize_shard

        run(spec)

        assert mock_rclone.call_count == 3
        spec_uploads = [
            call for call in mock_rclone.call_args_list if call[0][0].endswith(INPUT_SPEC_FILENAME)
        ]
        assert spec_uploads == []

    @patch("synth_setter.cli.generate_dataset.subprocess.check_call")
    @patch("synth_setter.cli.generate_dataset._rclone_copy")
    def test_each_shard_uploaded_after_its_render(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Render and upload are interleaved per shard: render0, upload0, render1, upload1, ..."""
        spec = _multi_shard_spec(tmp_path, n=3)
        mock_check_call.side_effect = _materialize_shard
        manager = MagicMock()
        manager.attach_mock(mock_rclone, "rclone")
        manager.attach_mock(mock_check_call, "check_call")

        run(spec)

        call_names = [c[0] for c in manager.mock_calls]
        # check_call/rclone alternate per shard (no leading spec upload).
        assert call_names == [
            "check_call",  # render shard 0
            "rclone",  # upload shard 0
            "check_call",  # render shard 1
            "rclone",  # upload shard 1
            "check_call",  # render shard 2
            "rclone",  # upload shard 2
        ]

    @patch("synth_setter.cli.generate_dataset.subprocess.check_call")
    @patch("synth_setter.cli.generate_dataset._rclone_copy")
    def test_local_shard_file_removed_after_upload(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Each shard's local HDF5 is unlinked after upload to bound disk to one shard."""
        spec = _multi_shard_spec(tmp_path, n=3)
        mock_check_call.side_effect = _materialize_shard
        # Track which paths existed at upload time, and which still exist after run().
        uploaded_paths: list[Path] = []

        def _record_upload(src: str, dest: str) -> None:
            uploaded_paths.append(Path(src))

        mock_rclone.side_effect = _record_upload

        run(spec)

        # All uploads are shards now (the worker no longer re-uploads the spec).
        assert len(uploaded_paths) == 3
        for shard_path in uploaded_paths:
            assert shard_path.name != INPUT_SPEC_FILENAME
            assert not shard_path.exists(), f"shard file still on disk after run: {shard_path}"

    @patch("synth_setter.cli.generate_dataset.subprocess.check_call")
    @patch("synth_setter.cli.generate_dataset._rclone_copy")
    def test_subprocess_failure_in_second_shard_propagates_immediately(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Mid-loop subprocess failure raises immediately; later shards are not attempted."""
        spec = _multi_shard_spec(tmp_path, n=3)
        # First call materializes shard 0's file (so the existence check passes and the loop
        # advances to shard 1); second call raises mid-loop; third call must never run.
        call_count = 0

        def _side_effect(args: list[str]) -> int:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise subprocess.CalledProcessError(1, "generate_vst_dataset.py")
            return _materialize_shard(args)

        mock_check_call.side_effect = _side_effect

        with pytest.raises(subprocess.CalledProcessError):
            run(spec)

        assert mock_check_call.call_count == 2

    @patch("synth_setter.cli.generate_dataset.subprocess.check_call")
    @patch("synth_setter.cli.generate_dataset._rclone_copy")
    def test_subprocess_exits_zero_without_writing_shard_raises(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        spec: DatasetSpec,
    ) -> None:
        """If the renderer exits 0 but never wrote the expected shard file, fail loudly.

        Catches a generator bug at the rendering boundary instead of letting it surface as a less-
        direct rclone "source not found" further down the pipeline.
        """
        # subprocess returns successfully but produces no file.
        mock_check_call.return_value = 0

        with pytest.raises(RuntimeError, match="did not write expected shard file"):
            run(spec)

        # No rclone calls: the worker no longer re-uploads the spec, and the
        # missing-shard check fires before the shard upload would have run.
        assert mock_rclone.call_count == 0

    @patch("synth_setter.cli.generate_dataset.subprocess.check_call")
    @patch("synth_setter.cli.generate_dataset._rclone_copy")
    def test_renderer_version_mismatch_raises_before_uploads(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Fail before any rclone/subprocess work when plugin version disagrees with spec.

        This prevents emitting a shard tagged with the wrong renderer_version.
        """
        kwargs = _base_spec_kwargs(tmp_path)
        kwargs["render"] = {**kwargs["render"], "renderer_version": "999.999.999"}  # type: ignore[dict-item]
        spec = DatasetSpec(**kwargs)  # type: ignore[arg-type]

        with pytest.raises(RuntimeError, match="Renderer version mismatch"):
            run(spec)
        mock_rclone.assert_not_called()
        mock_check_call.assert_not_called()

    @patch("synth_setter.cli.generate_dataset.subprocess.check_call")
    @patch("synth_setter.cli.generate_dataset._rclone_copy")
    def test_run_raises_when_skypilot_env_missing(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Missing partition env → ValueError before any rclone or subprocess work.

        Removes the silent-default smell where a worker invoked without partition env would
        otherwise duplicate every shard across every node.
        """
        monkeypatch.delenv("SYNTH_SETTER_WORKER_RANK", raising=False)
        monkeypatch.delenv("SYNTH_SETTER_NUM_WORKERS", raising=False)
        spec = _multi_shard_spec(tmp_path, n=3)

        with pytest.raises(ValueError) as excinfo:
            run(spec)
        message = str(excinfo.value)
        assert "SYNTH_SETTER_WORKER_RANK" in message
        assert "SYNTH_SETTER_NUM_WORKERS" in message
        mock_rclone.assert_not_called()
        mock_check_call.assert_not_called()

    @patch("synth_setter.cli.generate_dataset.subprocess.check_call")
    @patch("synth_setter.cli.generate_dataset._rclone_copy")
    def test_rank_0_of_2_renders_only_first_half_of_shards(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Worker 0 of a 2-node partition with 3 shards renders shards 0 and 1 only."""
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "2")
        spec = _multi_shard_spec(tmp_path, n=3)
        mock_check_call.side_effect = _materialize_shard

        run(spec)

        rendered_filenames = []
        for call in mock_check_call.call_args_list:
            args = call[0][0]
            output_file = args[_find_script_index(args) + 1]
            rendered_filenames.append(Path(output_file).name)
        assert rendered_filenames == [spec.shards[0].filename, spec.shards[1].filename]

    @patch("synth_setter.cli.generate_dataset.subprocess.check_call")
    @patch("synth_setter.cli.generate_dataset._rclone_copy")
    def test_rank_1_of_2_renders_only_remaining_shard(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Worker 1 of a 2-node partition with 3 shards renders shard 2 only."""
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "1")
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "2")
        spec = _multi_shard_spec(tmp_path, n=3)
        mock_check_call.side_effect = _materialize_shard

        run(spec)

        rendered_filenames = []
        for call in mock_check_call.call_args_list:
            args = call[0][0]
            output_file = args[_find_script_index(args) + 1]
            rendered_filenames.append(Path(output_file).name)
        assert rendered_filenames == [spec.shards[2].filename]

    @patch("synth_setter.cli.generate_dataset.subprocess.check_call")
    @patch("synth_setter.cli.generate_dataset._rclone_copy")
    def test_no_spec_upload_regardless_of_partition(  # noqa: DOC101,DOC103
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No worker re-uploads the spec, regardless of rank/world."""
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "1")
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "2")
        spec = _multi_shard_spec(tmp_path, n=3)
        mock_check_call.side_effect = _materialize_shard

        run(spec)

        spec_uploads = [
            call for call in mock_rclone.call_args_list if call[0][0].endswith(INPUT_SPEC_FILENAME)
        ]
        assert spec_uploads == []

    @patch("synth_setter.cli.generate_dataset.subprocess.check_call")
    @patch("synth_setter.cli.generate_dataset._rclone_copy")
    def test_excess_worker_exits_cleanly_without_rendering_or_uploading(  # noqa: DOC101,DOC103
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When world > num_shards, the excess workers exit cleanly with zero uploads.

        A 4-node partition over 3 shards leaves worker 3 with an empty range — it renders nothing
        and (since the worker spec re-upload is gone) makes zero rclone calls.
        """
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "3")
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "4")
        spec = _multi_shard_spec(tmp_path, n=3)

        run(spec)

        mock_check_call.assert_not_called()
        mock_rclone.assert_not_called()

    # Skip-existing-shards — see #750.

    @patch("synth_setter.cli.generate_dataset.subprocess.check_call")
    @patch("synth_setter.cli.generate_dataset._rclone_copy")
    def test_run_skips_render_when_shard_already_in_r2(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        spec: DatasetSpec,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Object present (size > 0) → renderer is not invoked, shard upload is not attempted."""
        monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda *_a, **_k: 12345)

        run(spec)

        mock_check_call.assert_not_called()
        # No rclone calls: the shard is already in R2 and the worker no
        # longer re-uploads the spec.
        mock_rclone.assert_not_called()

    @patch("synth_setter.cli.generate_dataset.subprocess.check_call")
    @patch("synth_setter.cli.generate_dataset._rclone_copy")
    def test_run_renders_when_object_absent(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        spec: DatasetSpec,
    ) -> None:
        """Object absent (None) → render proceeds as before.

        Relies on the autouse ``_default_shard_absent_in_r2`` fixture's default of None.
        """
        mock_check_call.side_effect = _materialize_shard

        run(spec)

        mock_check_call.assert_called_once()

    @patch("synth_setter.cli.generate_dataset.subprocess.check_call")
    @patch("synth_setter.cli.generate_dataset._rclone_copy")
    def test_run_renders_when_object_zero_size(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        spec: DatasetSpec,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Zero-byte object is treated as absent — defensive against half-uploaded objects."""
        monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", lambda *_a, **_k: 0)
        mock_check_call.side_effect = _materialize_shard

        run(spec)

        mock_check_call.assert_called_once()

    @patch("synth_setter.cli.generate_dataset.subprocess.check_call")
    @patch("synth_setter.cli.generate_dataset._rclone_copy")
    def test_run_skip_path_probes_full_object_uri_per_shard(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Probe is called once per assigned shard with the full object URI under r2_prefix."""
        spec = _multi_shard_spec(tmp_path, n=3)
        probed_uris: list[str] = []

        def _probe(uri: str) -> None:
            probed_uris.append(uri)
            return None

        monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", _probe)
        mock_check_call.side_effect = _materialize_shard

        run(spec)

        assert probed_uris == [
            f"r2://{spec.r2_bucket}/{spec.r2_prefix}{shard.filename}" for shard in spec.shards
        ]

    @patch("synth_setter.cli.generate_dataset.subprocess.check_call")
    @patch("synth_setter.cli.generate_dataset._rclone_copy")
    def test_run_renders_only_absent_shards_in_mixed_run(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Mid-run resumption: shard 0 already in R2, shards 1 and 2 absent → render only 1 and 2."""
        spec = _multi_shard_spec(tmp_path, n=3)

        def _present_only_for_shard_0(uri: str) -> int | None:
            return 9999 if uri.endswith("shard-000000.h5") else None

        monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", _present_only_for_shard_0)
        mock_check_call.side_effect = _materialize_shard

        run(spec)

        rendered_filenames = []
        for call in mock_check_call.call_args_list:
            args = call[0][0]
            output_file = args[_find_script_index(args) + 1]
            rendered_filenames.append(Path(output_file).name)
        assert rendered_filenames == ["shard-000001.h5", "shard-000002.h5"]

    @patch("synth_setter.cli.generate_dataset.logger")
    @patch("synth_setter.cli.generate_dataset.subprocess.check_call")
    @patch("synth_setter.cli.generate_dataset._rclone_copy")
    def test_run_logs_summary_with_rendered_and_skipped_counts(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        mock_logger: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """End-of-run summary reports rendered/skipped counts over the assigned range."""
        spec = _multi_shard_spec(tmp_path, n=3)

        def _present_only_for_shard_0(uri: str) -> int | None:
            return 9999 if uri.endswith("shard-000000.h5") else None

        monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", _present_only_for_shard_0)
        mock_check_call.side_effect = _materialize_shard

        run(spec)

        info_messages = [str(c.args[0]) for c in mock_logger.info.call_args_list]
        summary_lines = [m for m in info_messages if "rendered=" in m and "skipped=" in m]
        assert len(summary_lines) == 1, f"expected exactly one summary line, got: {info_messages}"
        assert "rendered=2" in summary_lines[0]
        assert "skipped=1" in summary_lines[0]

    @patch("synth_setter.cli.generate_dataset.subprocess.check_call")
    @patch("synth_setter.cli.generate_dataset._rclone_copy")
    def test_run_probe_failure_propagates(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        spec: DatasetSpec,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-zero rclone exit during the probe propagates as CalledProcessError."""

        def _raise(*_a: object, **_k: object) -> None:
            raise subprocess.CalledProcessError(1, ["rclone", "lsf"])

        monkeypatch.setattr("synth_setter.pipeline.r2_io.object_size", _raise)

        with pytest.raises(subprocess.CalledProcessError):
            run(spec)

        mock_check_call.assert_not_called()


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
        """``data``, ``r2``, ``paths``, ``hydra`` are dropped so strict validation passes.

        DatasetSpec is configured with ``extra="forbid"``; if any of these groups leaked through,
        construction would raise on the unknown field. The assertion is implicit in the absence
        of a ValidationError.
        """
        from omegaconf import OmegaConf

        from synth_setter.cli.generate_dataset import spec_from_cfg

        cfg_dict: dict[str, object] = dict(valid_dataset_spec_kwargs)
        cfg_dict["data"] = {"sample_rate": 16000}
        cfg_dict["r2"] = {"bucket": "intermediate-data", "prefix_root": "data/"}
        cfg_dict["paths"] = {"root_dir": "/fake-root"}
        cfg_dict["hydra"] = {"runtime": {"output_dir": "/fake-out"}}

        spec = spec_from_cfg(OmegaConf.create(cfg_dict))

        assert spec.task_name == valid_dataset_spec_kwargs["task_name"]

    def test_resolves_interpolations_before_dropping_groups(
        self, valid_dataset_spec_kwargs: dict[str, object]
    ) -> None:
        """``${r2.bucket}`` interpolation is resolved before the ``r2`` group is dropped.

        Mirrors the production composition: ``configs/dataset.yaml`` has
        ``r2_bucket: ${r2.bucket}`` and the ``r2`` group is only present for that interpolation.
        Dropping ``r2`` before resolving would lose the bucket value.
        """
        from omegaconf import OmegaConf

        from synth_setter.cli.generate_dataset import spec_from_cfg

        kwargs = dict(valid_dataset_spec_kwargs)
        kwargs["r2_bucket"] = "${r2.bucket}"
        kwargs["r2"] = {"bucket": "interpolated-bucket"}

        spec = spec_from_cfg(OmegaConf.create(kwargs))

        assert spec.r2_bucket == "interpolated-bucket"


# PROJECT_ROOT-bootstrap behavior is exercised end-to-end by tests/pipeline/test_configs/
# test_experiment_yamls.py — those tests fail with an InterpolationResolutionError if the
# module's import-time `rootutils.setup_root(...)` ever stops setting PROJECT_ROOT.


# ---------------------------------------------------------------------------
# _build_worker_cmd — shell-quoted cmd injection for sky.Task.run
# ---------------------------------------------------------------------------


class TestBuildWorkerCmd:
    """The worker cmd reconstructs the operator's Hydra invocation under bash."""

    @pytest.fixture()
    def spec(self, tmp_path: Path) -> DatasetSpec:  # noqa: DOC101,DOC103,DOC201,DOC203
        """Reusable DatasetSpec for worker-cmd construction (no I/O — pure kwargs)."""
        return DatasetSpec(**_base_spec_kwargs(tmp_path))  # type: ignore[arg-type]

    def test_cmd_uses_from_hydra_console_script(self, spec: DatasetSpec) -> None:  # noqa: DOC101,DOC103
        """The worker reproduces the composition by re-entering the from_hydra entry point."""
        from synth_setter.cli.generate_dataset import _build_worker_cmd

        cmd = _build_worker_cmd(["experiment=foo"], spec)
        assert "synth-setter-generate-dataset-from-hydra" in cmd
        assert "experiment=foo" in cmd

    def test_cmd_cds_to_worker_repo_root_not_launcher_repo(  # noqa: DOC101,DOC103
        self, spec: DatasetSpec
    ) -> None:
        """Cd target is the worker checkout, not the launcher's path."""
        from synth_setter.cli.generate_dataset import _WORKER_REPO_ROOT, _build_worker_cmd

        cmd = _build_worker_cmd([], spec)
        assert cmd.startswith(f"cd {_WORKER_REPO_ROOT}")
        assert _WORKER_REPO_ROOT == "/home/build/synth-setter"

    def test_cmd_runs_sync_worker_checkout_before_exec(  # noqa: DOC101,DOC103
        self, spec: DatasetSpec
    ) -> None:
        """sync_worker_checkout.sh bypasses dev-snapshot bake-lag when WORKER_GIT_REF is set."""
        from synth_setter.cli.generate_dataset import _build_worker_cmd

        cmd = _build_worker_cmd([], spec)
        sync_idx = cmd.find("bash scripts/sync_worker_checkout.sh")
        exec_idx = cmd.find("exec synth-setter-generate-dataset-from-hydra")
        assert sync_idx != -1, f"sync step missing from cmd: {cmd!r}"
        assert exec_idx != -1, f"exec step missing from cmd: {cmd!r}"
        assert sync_idx < exec_idx, "sync_worker_checkout must run before exec"

    def test_cmd_pins_spec_created_at_via_hydra_override(  # noqa: DOC101,DOC103
        self, spec: DatasetSpec
    ) -> None:
        """Worker compose must inherit launcher's created_at to land on the same r2_prefix."""
        from synth_setter.cli.generate_dataset import _build_worker_cmd

        cmd = _build_worker_cmd([], spec)
        # `+key=value` is Hydra's add-key syntax; spec.created_at.isoformat() goes in verbatim
        # (no surrounding quotes added by shlex when the value has no shell metachars).
        assert f"+created_at={spec.created_at.isoformat()}" in cmd

    def test_cmd_shell_quotes_overrides_with_spaces(  # noqa: DOC101,DOC103
        self, spec: DatasetSpec
    ) -> None:
        """Spaces and special chars in an override survive bash interpretation in run:."""
        from synth_setter.cli.generate_dataset import _build_worker_cmd

        cmd = _build_worker_cmd(["task_name=value with space"], spec)
        # shlex.quote wraps the whole assignment in single quotes; the bare-word form
        # would be split into two argv items by bash.
        assert "'task_name=value with space'" in cmd

    def test_cmd_handles_empty_operator_overrides(  # noqa: DOC101,DOC103
        self, spec: DatasetSpec
    ) -> None:
        """No operator overrides → cmd is just cd + sync + exec + pinned-runtime override."""
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
    def _set_default_skypilot_env(self, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: DOC101,DOC103
        """Set single-worker rank/world env so the local branch's run() succeeds."""
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")

    def test_compute_template_null_calls_run_locally(  # noqa: DOC101,DOC103
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """compute_template=null routes to run(spec) with a DatasetSpec; dispatch stays unused."""
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

    def test_compute_template_set_calls_dispatch_via_skypilot(  # noqa: DOC101,DOC103
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """compute_template=<path> routes through dispatch_via_skypilot with cmd populated."""
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

        def _fake_dispatch(spec: object, sky_cfg: object) -> None:
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

    def test_operator_supplied_cmd_is_rejected(  # noqa: DOC101,DOC103
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A `+skypilot_launch.cmd=…` override is rejected before any dispatch fires.

        Uses Hydra's `+key=value` add-syntax because the key isn't in
        configs/skypilot_launch/default.yaml (struct-mode would otherwise reject it before our
        guard runs).
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
