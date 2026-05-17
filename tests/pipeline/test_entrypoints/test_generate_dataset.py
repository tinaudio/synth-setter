"""Tests for synth_setter.cli.generate_dataset — spec-driven run.

The entrypoint's public surface is a single ``run(spec)`` function that:
  1. Serializes the spec to a tempfile.
  2. Uploads the spec to R2 at ``r2:{bucket}/{prefix}/input_spec.json``.
  3. For each shard in ``spec.shards``, shells out to ``generate_vst_dataset.py``
     to render it, uploads to R2 at ``r2:{bucket}/{prefix}/``, and unlinks the
     local file.

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
    """Run() orchestrates: serialize → upload spec → generate → upload shard."""

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
    def test_uploads_spec_to_r2_at_expected_path(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        spec: DatasetSpec,
    ) -> None:
        """Spec upload copies spec_path into the prefix directory (rclone preserves basename)."""
        mock_check_call.side_effect = _materialize_shard
        run(spec)

        rclone_calls = mock_rclone.call_args_list
        assert len(rclone_calls) == 2
        spec_src, spec_dest = rclone_calls[0][0]
        # Local spec file is already named INPUT_SPEC_FILENAME; `rclone copy`
        # into the prefix directory preserves the basename → final object
        # key is `{prefix}{INPUT_SPEC_FILENAME}`.
        assert spec_src.endswith(INPUT_SPEC_FILENAME)
        assert spec_dest == f"r2:{spec.r2_bucket}/{spec.r2_prefix}"

    @patch("synth_setter.cli.generate_dataset.subprocess.check_call")
    @patch("synth_setter.cli.generate_dataset._rclone_copy")
    def test_spec_upload_precedes_shard_generation(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        spec: DatasetSpec,
    ) -> None:
        """Spec is uploaded to R2 before the shard-generating subprocess runs."""
        mock_check_call.side_effect = _materialize_shard
        manager = MagicMock()
        manager.attach_mock(mock_rclone, "rclone")
        manager.attach_mock(mock_check_call, "check_call")

        run(spec)

        call_names = [c[0] for c in manager.mock_calls]
        assert call_names.index("rclone") < call_names.index("check_call")

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
        """Second rclone call uploads the shard to r2:{bucket}/{prefix}/."""
        mock_check_call.side_effect = _materialize_shard
        run(spec)

        rclone_calls = mock_rclone.call_args_list
        assert len(rclone_calls) == 2
        shard_src, shard_dest = rclone_calls[1][0]
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
    def test_spec_uploaded_exactly_once_for_multi_shard_run(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Spec is uploaded once per run; remaining rclone calls are per-shard uploads."""
        spec = _multi_shard_spec(tmp_path, n=3)
        mock_check_call.side_effect = _materialize_shard

        run(spec)

        assert mock_rclone.call_count == 1 + 3
        spec_uploads = [
            call for call in mock_rclone.call_args_list if call[0][0].endswith(INPUT_SPEC_FILENAME)
        ]
        assert len(spec_uploads) == 1

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
        # First call is the spec upload; thereafter check_call/rclone alternate per shard.
        assert call_names == [
            "rclone",  # spec upload
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

        shard_uploads = [p for p in uploaded_paths if p.name != INPUT_SPEC_FILENAME]
        assert len(shard_uploads) == 3
        for shard_path in shard_uploads:
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

        # Spec was uploaded (1 rclone call), but no shard upload was attempted.
        assert mock_rclone.call_count == 1

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
    def test_spec_uploaded_exactly_once_independent_of_partition(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Spec upload is partition-independent: every worker uploads it exactly once."""
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "1")
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "2")
        spec = _multi_shard_spec(tmp_path, n=3)
        mock_check_call.side_effect = _materialize_shard

        run(spec)

        spec_uploads = [
            call for call in mock_rclone.call_args_list if call[0][0].endswith(INPUT_SPEC_FILENAME)
        ]
        assert len(spec_uploads) == 1

    @patch("synth_setter.cli.generate_dataset.subprocess.check_call")
    @patch("synth_setter.cli.generate_dataset._rclone_copy")
    def test_excess_worker_renders_no_shards_but_still_uploads_spec(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When world > num_shards, the excess workers exit cleanly without rendering.

        A 4-node partition over 3 shards leaves worker 3 with an empty range — it must still upload
        the spec (idempotent) but render zero shards.
        """
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "3")
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "4")
        spec = _multi_shard_spec(tmp_path, n=3)

        run(spec)

        mock_check_call.assert_not_called()
        spec_uploads = [
            call for call in mock_rclone.call_args_list if call[0][0].endswith(INPUT_SPEC_FILENAME)
        ]
        assert len(spec_uploads) == 1

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
        # Spec is still uploaded; no per-shard upload happens because the shard is already there.
        assert mock_rclone.call_count == 1
        assert mock_rclone.call_args_list[0][0][0].endswith(INPUT_SPEC_FILENAME)

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
# main — compute_template dispatch
# ---------------------------------------------------------------------------


class TestMainComputeTemplateDispatch:
    """``main(cfg)`` dispatches via SkyPilot when ``cfg.compute_template`` is set; else local."""

    def _make_cfg(  # noqa: DOC101,DOC103,DOC201,DOC203
        self, valid_dataset_spec_kwargs: dict[str, object], compute_template: object
    ) -> object:
        """Build a DictConfig that mirrors what `configs/dataset.yaml` composes."""
        from omegaconf import OmegaConf

        cfg_dict = dict(valid_dataset_spec_kwargs)
        cfg_dict["compute_template"] = compute_template
        return OmegaConf.create(cfg_dict)

    def test_null_compute_template_runs_locally(  # noqa: DOC101,DOC103
        self, valid_dataset_spec_kwargs: dict[str, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`compute_template: null` runs the spec via the in-process `run()` path."""
        from synth_setter.cli import generate_dataset

        run_mock = MagicMock()
        dispatch_mock = MagicMock()
        monkeypatch.setattr(generate_dataset, "run", run_mock)
        # `dispatch_via_skypilot` is imported lazily inside main(); patch on its source module.
        monkeypatch.setattr(
            "synth_setter.pipeline.skypilot_launch.dispatch_via_skypilot", dispatch_mock
        )

        cfg = self._make_cfg(valid_dataset_spec_kwargs, compute_template=None)
        generate_dataset.main.__wrapped__(cfg)

        run_mock.assert_called_once()
        dispatch_mock.assert_not_called()

    def test_compute_template_dispatches_via_skypilot(  # noqa: DOC101,DOC103
        self, valid_dataset_spec_kwargs: dict[str, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-null `compute_template` loads the YAML and calls `dispatch_via_skypilot`."""
        from synth_setter.cli import generate_dataset
        from synth_setter.pipeline.schemas.compute import ComputeConfig

        run_mock = MagicMock()
        dispatch_mock = MagicMock()
        monkeypatch.setattr(generate_dataset, "run", run_mock)
        monkeypatch.setattr(
            "synth_setter.pipeline.skypilot_launch.dispatch_via_skypilot", dispatch_mock
        )

        cfg = self._make_cfg(valid_dataset_spec_kwargs, compute_template="runpod-template")
        generate_dataset.main.__wrapped__(cfg)

        run_mock.assert_not_called()
        dispatch_mock.assert_called_once()
        (passed_spec, passed_compute), _ = dispatch_mock.call_args
        assert passed_spec.task_name == valid_dataset_spec_kwargs["task_name"]
        assert isinstance(passed_compute, ComputeConfig)
        assert passed_compute.resources["cloud"] == "runpod"

    def test_spec_from_cfg_drops_compute_template(  # noqa: DOC101,DOC103
        self, valid_dataset_spec_kwargs: dict[str, object]
    ) -> None:
        """``compute_template`` is a dispatch knob, not a spec field — drop it before validation.

        DatasetSpec uses ``extra="forbid"``; if `compute_template` leaked through, construction
        would raise on the unknown field. Implicit assertion: spec_from_cfg succeeds.
        """
        from omegaconf import OmegaConf

        from synth_setter.cli.generate_dataset import spec_from_cfg

        cfg_dict = dict(valid_dataset_spec_kwargs)
        cfg_dict["compute_template"] = "runpod-template"

        spec = spec_from_cfg(OmegaConf.create(cfg_dict))

        assert spec.task_name == valid_dataset_spec_kwargs["task_name"]
