"""Tests for src/generate_dataset.py — spec-driven run.

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

from src.generate_dataset import (
    VST_HEADLESS_WRAPPER,
    build_generate_args,
    run,
)
from src.pipeline.constants import INPUT_SPEC_FILENAME
from src.pipeline.schemas.spec import DatasetSpec

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
        "train_val_test_seeds": [42, 43, 44],
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
            "sample_batch_size": 32,
            "batch_per_shard": 10000,
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

    @patch("src.generate_dataset.subprocess.check_call")
    @patch("src.generate_dataset._rclone_copy")
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

    @patch("src.generate_dataset.subprocess.check_call")
    @patch("src.generate_dataset._rclone_copy")
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

    @patch("src.generate_dataset.subprocess.check_call")
    @patch("src.generate_dataset._rclone_copy")
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
        # args = [VST_HEADLESS_WRAPPER (linux only), python, generate_vst_dataset.py,
        #         <output-path>, --render-cfg-json, <json>]
        assert any("generate_vst_dataset.py" in a for a in args)
        render_cfg_idx = args.index("--render-cfg-json")
        rendered_json = args[render_cfg_idx + 1]
        assert str(spec.render.batch_per_shard) in rendered_json

    @patch("src.generate_dataset.subprocess.check_call")
    @patch("src.generate_dataset._rclone_copy")
    def test_shard_generation_runs_under_headless_vst_wrapper(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        spec: DatasetSpec,
    ) -> None:
        """The VST subprocess is prefixed with scripts/run-linux-vst-headless.sh on Linux.

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
            assert args[2] == "src/data/vst/generate_vst_dataset.py"
        else:
            assert VST_HEADLESS_WRAPPER not in args
            assert args[1] == "src/data/vst/generate_vst_dataset.py"

    @patch("src.generate_dataset.subprocess.check_call")
    @patch("src.generate_dataset._rclone_copy")
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

    @patch("src.generate_dataset.subprocess.check_call")
    @patch("src.generate_dataset._rclone_copy")
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

    @patch("src.generate_dataset.subprocess.check_call")
    @patch("src.generate_dataset._rclone_copy")
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

    @patch("src.generate_dataset.subprocess.check_call")
    @patch("src.generate_dataset._rclone_copy")
    def test_wds_output_format_uploads_tar_shard(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """output_format 'wds' renders a .tar shard and uploads only the tar (not the temp h5)."""
        monkeypatch.setenv("SYNTH_SETTER_WORKER_RANK", "0")
        monkeypatch.setenv("SYNTH_SETTER_NUM_WORKERS", "1")
        kwargs = _base_spec_kwargs(tmp_path, output_format="wds")
        spec = DatasetSpec(**kwargs)  # type: ignore[arg-type]

        def _materialize_tar(args: list[str]) -> int:
            script_idx = _find_script_index(args)
            tar_file = Path(args[script_idx + 1])
            tar_file.parent.mkdir(parents=True, exist_ok=True)
            tar_file.write_bytes(b"")
            return 0

        mock_check_call.side_effect = _materialize_tar

        run(spec)

        uploaded = [call[0][0] for call in mock_rclone.call_args_list]
        assert any(p.endswith("shard-000000.tar") for p in uploaded)
        assert not any(p.endswith("shard-000000.h5") for p in uploaded)

    @patch("src.generate_dataset.subprocess.check_call")
    @patch("src.generate_dataset._rclone_copy")
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

    @patch("src.generate_dataset.subprocess.check_call")
    @patch("src.generate_dataset._rclone_copy")
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

    @patch("src.generate_dataset.subprocess.check_call")
    @patch("src.generate_dataset._rclone_copy")
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

    @patch("src.generate_dataset.subprocess.check_call")
    @patch("src.generate_dataset._rclone_copy")
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

    @patch("src.generate_dataset.subprocess.check_call")
    @patch("src.generate_dataset._rclone_copy")
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

    @patch("src.generate_dataset.subprocess.check_call")
    @patch("src.generate_dataset._rclone_copy")
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

    @patch("src.generate_dataset.subprocess.check_call")
    @patch("src.generate_dataset._rclone_copy")
    def test_renderer_version_mismatch_raises_before_uploads(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When the plugin's actual version disagrees with spec.renderer_version, run() fails
        before any rclone/subprocess work happens (prevents emitting a shard tagged with the wrong
        renderer_version)."""
        kwargs = _base_spec_kwargs(tmp_path)
        kwargs["render"] = {**kwargs["render"], "renderer_version": "999.999.999"}  # type: ignore[dict-item]
        spec = DatasetSpec(**kwargs)  # type: ignore[arg-type]

        with pytest.raises(RuntimeError, match="Renderer version mismatch"):
            run(spec)
        mock_rclone.assert_not_called()
        mock_check_call.assert_not_called()

    @patch("src.generate_dataset.subprocess.check_call")
    @patch("src.generate_dataset._rclone_copy")
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

    @patch("src.generate_dataset.subprocess.check_call")
    @patch("src.generate_dataset._rclone_copy")
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

    @patch("src.generate_dataset.subprocess.check_call")
    @patch("src.generate_dataset._rclone_copy")
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

    @patch("src.generate_dataset.subprocess.check_call")
    @patch("src.generate_dataset._rclone_copy")
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

    @patch("src.generate_dataset.subprocess.check_call")
    @patch("src.generate_dataset._rclone_copy")
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

    def test_render_cfg_json_arg_carries_full_render_config(self, spec: DatasetSpec) -> None:
        """The single ``--render-cfg-json`` arg holds a serialized RenderConfig — every render
        field round-trips."""
        shard = spec.shards[0]

        args = build_generate_args(spec, shard, Path("out"))

        render_cfg_idx = args.index("--render-cfg-json")
        rendered_json = args[render_cfg_idx + 1]
        # Deserialization must produce a RenderConfig equal to spec.render.
        from src.pipeline.schemas.spec import RenderConfig

        restored = RenderConfig.model_validate_json(rendered_json)
        assert restored == spec.render

    def test_args_start_with_python_and_script(self, spec: DatasetSpec) -> None:
        """First arg is the Python executable, second is the generation script."""
        shard = spec.shards[0]

        args = build_generate_args(spec, shard, Path("out"))

        assert args[1] == "src/data/vst/generate_vst_dataset.py"

    def test_wds_output_format_passes_tar_positional_and_no_format_flag(
        self, tmp_path: Path
    ) -> None:
        """For wds specs, positional is the tar path; format is implicit in the extension."""
        kwargs = _base_spec_kwargs(tmp_path, output_format="wds")
        spec = DatasetSpec(**kwargs)  # type: ignore[arg-type]
        shard = spec.shards[0]

        args = build_generate_args(spec, shard, tmp_path)

        assert args[2] == str(tmp_path / "shard-000000.tar")
        assert "--format" not in args
        assert "--wds-out" not in args


# ---------------------------------------------------------------------------
# _dataset_spec_from_cfg — Hydra → Pydantic adapter
# ---------------------------------------------------------------------------


class TestDatasetSpecFromCfg:
    """``_dataset_spec_from_cfg`` rejects non-mapping Hydra configs with a clear error."""

    def test_dataset_spec_from_cfg_non_mapping_raises_type_error(self) -> None:
        """A composed config that resolves to a list (not a dict) raises TypeError loudly."""
        from omegaconf import OmegaConf

        from src.generate_dataset import _dataset_spec_from_cfg

        list_cfg = OmegaConf.create([1, 2, 3])

        with pytest.raises(TypeError, match="composed Hydra config is not a mapping"):
            _dataset_spec_from_cfg(list_cfg)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# compose_dataset_spec — Hydra composition entrypoint
# ---------------------------------------------------------------------------


class TestComposeDatasetSpec:
    """compose_dataset_spec materializes a DatasetSpec from a Hydra experiment."""

    def test_composes_known_experiment(self) -> None:
        """``datagen/ci-materialize-test`` composes to a 3-shard hdf5 spec."""
        from src.generate_dataset import compose_dataset_spec

        spec = compose_dataset_spec("datagen/ci-materialize-test")
        assert spec.task_name == "ci-materialize-test"
        assert spec.output_format == "hdf5"
        assert spec.num_shards == 3

    def test_overrides_apply(self) -> None:
        """Hydra-style overrides modify nested fields on the composed spec."""
        from src.generate_dataset import compose_dataset_spec

        spec = compose_dataset_spec(
            "datagen/ci-materialize-test", overrides=["render.sample_rate=22050"]
        )
        assert spec.render.sample_rate == 22050

    def test_unknown_experiment_raises_missing_config(self) -> None:
        """A typo'd experiment name surfaces as Hydra's MissingConfigException, not a KeyError."""
        from hydra.errors import MissingConfigException

        from src.generate_dataset import compose_dataset_spec

        with pytest.raises(MissingConfigException):
            compose_dataset_spec("datagen/does-not-exist")

    def test_invalid_override_field_raises_composition_error(self) -> None:
        """Overrides targeting a non-existent field raise ConfigCompositionException."""
        from hydra.errors import ConfigCompositionException

        from src.generate_dataset import compose_dataset_spec

        with pytest.raises(ConfigCompositionException):
            compose_dataset_spec(
                "datagen/ci-materialize-test",
                overrides=["render.does_not_exist=42"],
            )

    def test_global_hydra_left_uninitialized_after_failure(self) -> None:
        """A failed compose still leaves GlobalHydra in the uninitialized state.

        ``initialize_config_dir``'s context manager promises to clear GlobalHydra on
        exit; ``compose_dataset_spec`` adds a try/finally outside the context manager
        to defend against an exception thrown before the manager runs.
        """
        from hydra.core.global_hydra import GlobalHydra
        from hydra.errors import MissingConfigException

        from src.generate_dataset import compose_dataset_spec

        with pytest.raises(MissingConfigException):
            compose_dataset_spec("datagen/does-not-exist")

        assert not GlobalHydra.instance().is_initialized()

    def test_consecutive_composes_are_idempotent(self) -> None:
        """Calling compose twice produces equivalent specs — GlobalHydra reset works."""
        from src.generate_dataset import compose_dataset_spec

        spec_a = compose_dataset_spec("datagen/ci-materialize-test")
        spec_b = compose_dataset_spec("datagen/ci-materialize-test")

        assert spec_a.task_name == spec_b.task_name
        assert spec_a.num_shards == spec_b.num_shards
        assert spec_a.render == spec_b.render


# ---------------------------------------------------------------------------
# main — @hydra.main entrypoint
# ---------------------------------------------------------------------------


class TestMain:
    """``main(cfg)`` materializes the spec and delegates to ``run``."""

    def test_main_passes_composed_spec_to_run(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Main() builds a DatasetSpec from cfg and forwards it to run() unchanged."""
        # Compose a known cfg via the same path main()'s decorator would use.
        # Calling compose_dataset_spec gives us the spec; we then need an
        # equivalent DictConfig to feed into main.__wrapped__. Since main's body
        # only calls _dataset_spec_from_cfg(cfg) → run(spec), we can drive it via
        # any DictConfig that round-trips to the same DatasetSpec.
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        from src import generate_dataset
        from src.generate_dataset import compose_dataset_spec

        captured: list[generate_dataset.DatasetSpec] = []

        def _capture_run(spec: generate_dataset.DatasetSpec) -> None:
            captured.append(spec)

        monkeypatch.setattr(generate_dataset, "run", _capture_run)

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        try:
            with initialize_config_dir(
                version_base="1.3", config_dir=str(generate_dataset._CONFIGS_DIR)
            ):
                cfg = compose(
                    config_name="dataset",
                    overrides=["experiment=datagen/ci-materialize-test"],
                )
        finally:
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()

        # Bypass the @hydra.main decorator and call the underlying function directly.
        generate_dataset.main.__wrapped__(cfg)

        assert len(captured) == 1
        expected = compose_dataset_spec("datagen/ci-materialize-test")
        # task_name, num_shards, and render config are deterministic for this experiment.
        assert captured[0].task_name == expected.task_name
        assert captured[0].num_shards == expected.num_shards
        assert captured[0].render == expected.render
