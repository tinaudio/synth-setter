"""Tests for pipeline/entrypoints/generate_dataset.py — spec-driven run.

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
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pipeline.constants import INPUT_SPEC_FILENAME
from pipeline.entrypoints.generate_dataset import (
    VST_HEADLESS_WRAPPER,
    build_generate_args,
    run,
)
from pipeline.schemas.spec import DatasetPipelineSpec, ShardSpec

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
    # Args layout: [wrapper, python, generate_vst_dataset.py, output_file, ...].
    output_file = Path(args[3])
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_bytes(b"")
    return 0


def _base_spec_kwargs(tmp_path: Path, **overrides: object) -> dict[str, object]:
    """Return valid DatasetPipelineSpec kwargs for direct construction."""
    kwargs: dict[str, object] = {
        "run_id": "test-dataset-20260328T120000Z",
        "r2_prefix": "data/test-dataset/test-dataset-20260328T120000Z/",
        "created_at": datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc),
        "code_version": "a" * 40,
        "is_repo_dirty": False,
        "param_spec": "surge_simple",
        "renderer_version": TEST_PLUGIN_VERSION,
        "output_format": "hdf5",
        "sample_rate": 16000,
        "shard_size": 10000,
        "base_seed": 42,
        "num_params": 92,
        "r2_bucket": "intermediate-data",
        "splits": {"train": 1, "val": 0, "test": 0},
        "plugin_path": str(TEST_PLUGIN_VST3),
        "preset_path": "presets/surge-base.vstpreset",
        "channels": 2,
        "velocity": 100,
        "signal_duration_seconds": 4.0,
        "min_loudness": -55.0,
        "sample_batch_size": 32,
        "shards": (ShardSpec(shard_id=0, filename="shard-000000.h5", seed=42),),
    }
    kwargs.update(overrides)
    return kwargs


@pytest.fixture()
def spec(tmp_path: Path) -> DatasetPipelineSpec:
    """Return a valid single-shard DatasetPipelineSpec."""
    return DatasetPipelineSpec(**_base_spec_kwargs(tmp_path))  # type: ignore[arg-type]


def _multi_shard_spec(tmp_path: Path, n: int = 3) -> DatasetPipelineSpec:
    """Return a DatasetPipelineSpec with ``n`` shards (deterministic filenames/seeds)."""
    shards = tuple(
        ShardSpec(shard_id=i, filename=f"shard-{i:06d}.h5", seed=42 + i) for i in range(n)
    )
    kwargs = _base_spec_kwargs(
        tmp_path,
        splits={"train": n, "val": 0, "test": 0},
        shards=shards,
    )
    return DatasetPipelineSpec(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# run — full flow orchestration
# ---------------------------------------------------------------------------


class TestRun:
    """Run() orchestrates: serialize → upload spec → generate → upload shard."""

    @patch("pipeline.entrypoints.generate_dataset.subprocess.check_call")
    @patch("pipeline.entrypoints.generate_dataset._rclone_copy")
    def test_uploads_spec_to_r2_at_expected_path(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        spec: DatasetPipelineSpec,
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

    @patch("pipeline.entrypoints.generate_dataset.subprocess.check_call")
    @patch("pipeline.entrypoints.generate_dataset._rclone_copy")
    def test_spec_upload_precedes_shard_generation(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        spec: DatasetPipelineSpec,
    ) -> None:
        """Spec is uploaded to R2 before the shard-generating subprocess runs."""
        mock_check_call.side_effect = _materialize_shard
        manager = MagicMock()
        manager.attach_mock(mock_rclone, "rclone")
        manager.attach_mock(mock_check_call, "check_call")

        run(spec)

        call_names = [c[0] for c in manager.mock_calls]
        assert call_names.index("rclone") < call_names.index("check_call")

    @patch("pipeline.entrypoints.generate_dataset.subprocess.check_call")
    @patch("pipeline.entrypoints.generate_dataset._rclone_copy")
    def test_invokes_generate_vst_dataset_with_spec_derived_args(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        spec: DatasetPipelineSpec,
    ) -> None:
        """subprocess.check_call invokes generate_vst_dataset.py with spec-derived args."""
        mock_check_call.side_effect = _materialize_shard
        run(spec)

        mock_check_call.assert_called_once()
        args = mock_check_call.call_args[0][0]
        # args = [VST_HEADLESS_WRAPPER, python, generate_vst_dataset.py, ...]
        assert any("generate_vst_dataset.py" in a for a in args)
        assert str(spec.shard_size) in args

    @patch("pipeline.entrypoints.generate_dataset.subprocess.check_call")
    @patch("pipeline.entrypoints.generate_dataset._rclone_copy")
    def test_shard_generation_runs_under_headless_vst_wrapper(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        spec: DatasetPipelineSpec,
    ) -> None:
        """The VST subprocess is prefixed with scripts/run-linux-vst-headless.sh.

        X11 bootstrap lives at the audio-rendering boundary (this subprocess) so the
        docker_entrypoint click CLI can stay X11-agnostic — idle and passthrough modes don't pay
        the Xvfb startup cost.
        """
        mock_check_call.side_effect = _materialize_shard
        run(spec)

        args = mock_check_call.call_args[0][0]
        assert args[0] == VST_HEADLESS_WRAPPER
        # Wrapper prefixes the original generate_vst_dataset.py invocation,
        # so the python interpreter + script must appear immediately after.
        assert args[2] == "src/data/vst/generate_vst_dataset.py"

    @patch("pipeline.entrypoints.generate_dataset.subprocess.check_call")
    @patch("pipeline.entrypoints.generate_dataset._rclone_copy")
    def test_uploads_shard_to_r2_after_generation(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        spec: DatasetPipelineSpec,
    ) -> None:
        """Second rclone call uploads the shard to r2:{bucket}/{prefix}/."""
        mock_check_call.side_effect = _materialize_shard
        run(spec)

        rclone_calls = mock_rclone.call_args_list
        assert len(rclone_calls) == 2
        shard_src, shard_dest = rclone_calls[1][0]
        assert "shard-000000.h5" in shard_src
        assert shard_dest == f"r2:{spec.r2_bucket}/{spec.r2_prefix}"

    @patch("pipeline.entrypoints.generate_dataset.subprocess.check_call")
    @patch("pipeline.entrypoints.generate_dataset._rclone_copy")
    def test_subprocess_failure_propagates(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        spec: DatasetPipelineSpec,
    ) -> None:
        """CalledProcessError from generate_vst_dataset propagates to caller."""
        mock_check_call.side_effect = subprocess.CalledProcessError(1, "generate_vst_dataset.py")

        with pytest.raises(subprocess.CalledProcessError):
            run(spec)

    @patch("pipeline.entrypoints.generate_dataset.subprocess.check_call")
    @patch("pipeline.entrypoints.generate_dataset._rclone_copy")
    def test_rclone_failure_propagates(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        spec: DatasetPipelineSpec,
    ) -> None:
        """CalledProcessError from rclone propagates to caller."""
        mock_rclone.side_effect = subprocess.CalledProcessError(1, "rclone")

        with pytest.raises(subprocess.CalledProcessError):
            run(spec)

    def test_wds_output_format_raises(self, tmp_path: Path) -> None:
        """output_format 'wds' raises ValueError — generate_vst_dataset only supports hdf5."""
        kwargs = _base_spec_kwargs(tmp_path, output_format="wds")
        spec = DatasetPipelineSpec(**kwargs)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="only supports hdf5"):
            run(spec)

    @patch("pipeline.entrypoints.generate_dataset.subprocess.check_call")
    @patch("pipeline.entrypoints.generate_dataset._rclone_copy")
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
            # Args layout: [wrapper, python, generate_vst_dataset.py, output_file, ...].
            output_file = args[3]
            rendered_filenames.append(Path(output_file).name)
        assert rendered_filenames == [s.filename for s in spec.shards]

    @patch("pipeline.entrypoints.generate_dataset.subprocess.check_call")
    @patch("pipeline.entrypoints.generate_dataset._rclone_copy")
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

    @patch("pipeline.entrypoints.generate_dataset.subprocess.check_call")
    @patch("pipeline.entrypoints.generate_dataset._rclone_copy")
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

    @patch("pipeline.entrypoints.generate_dataset.subprocess.check_call")
    @patch("pipeline.entrypoints.generate_dataset._rclone_copy")
    def test_previous_shard_unlinked_before_next_render(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Each shard's local HDF5 is unlinked before the next render starts.

        Asserted as an in-loop invariant (not post-run) — the run wraps everything in
        ``tempfile.TemporaryDirectory()`` so a post-run existence check would pass
        regardless of whether unlink ran. This bounds local disk to one shard's worth
        at a time.
        """
        spec = _multi_shard_spec(tmp_path, n=3)
        rendered_paths: list[Path] = []

        def _render_side_effect(args: list[str]) -> int:
            # Args layout: [wrapper, python, generate_vst_dataset.py, output_file, ...].
            output_file = Path(args[3])
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_bytes(b"")
            # Before rendering shard i (i >= 1), shard i-1 must already be gone.
            if rendered_paths:
                previous = rendered_paths[-1]
                assert not previous.exists(), (
                    f"previous shard {previous.name} still on disk when "
                    f"rendering {output_file.name}"
                )
            rendered_paths.append(output_file)
            return 0

        mock_check_call.side_effect = _render_side_effect

        run(spec)

        # Sanity: the side effect must have fired once per shard.
        assert len(rendered_paths) == 3

    @patch("pipeline.entrypoints.generate_dataset.subprocess.check_call")
    @patch("pipeline.entrypoints.generate_dataset._rclone_copy")
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

    @patch("pipeline.entrypoints.generate_dataset.subprocess.check_call")
    @patch("pipeline.entrypoints.generate_dataset._rclone_copy")
    def test_subprocess_exits_zero_without_writing_shard_raises(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        spec: DatasetPipelineSpec,
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

    @patch("pipeline.entrypoints.generate_dataset.subprocess.check_call")
    @patch("pipeline.entrypoints.generate_dataset._rclone_copy")
    def test_renderer_version_mismatch_raises_before_uploads(
        self,
        mock_rclone: MagicMock,
        mock_check_call: MagicMock,
        tmp_path: Path,
    ) -> None:
        """When the plugin's actual version disagrees with spec.renderer_version, run() fails
        before any rclone/subprocess work happens (prevents emitting a shard tagged with the wrong
        renderer_version)."""
        kwargs = _base_spec_kwargs(tmp_path, renderer_version="999.999.999")
        spec = DatasetPipelineSpec(**kwargs)  # type: ignore[arg-type]

        with pytest.raises(RuntimeError, match="Renderer version mismatch"):
            run(spec)
        mock_rclone.assert_not_called()
        mock_check_call.assert_not_called()


# ---------------------------------------------------------------------------
# build_generate_args — arg construction from spec + shard
# ---------------------------------------------------------------------------


class TestBuildGenerateArgs:
    """build_generate_args() produces correct CLI arg lists from spec + shard."""

    def test_output_file_uses_shard_filename(
        self, spec: DatasetPipelineSpec, tmp_path: Path
    ) -> None:
        """Output file path is {output_dir}/{shard.filename}."""
        shard = spec.shards[0]

        args = build_generate_args(spec, shard, tmp_path)

        assert args[2] == str(tmp_path / "shard-000000.h5")

    def test_num_samples_is_shard_size(self, spec: DatasetPipelineSpec) -> None:
        """num_samples arg comes from spec.shard_size."""
        shard = spec.shards[0]

        args = build_generate_args(spec, shard, Path("out"))

        assert args[3] == str(spec.shard_size)

    def test_all_spec_fields_passed_as_options(self, spec: DatasetPipelineSpec) -> None:
        """All generation parameters from spec are passed as --key value options."""
        shard = spec.shards[0]

        args = build_generate_args(spec, shard, Path("out"))

        option_keys: set[str] = set()
        i = 4
        while i < len(args):
            if args[i].startswith("--"):
                option_keys.add(args[i].lstrip("-"))
                i += 2
            else:
                i += 1

        expected_keys = {
            "plugin_path",
            "preset_path",
            "sample_rate",
            "channels",
            "velocity",
            "signal_duration_seconds",
            "min_loudness",
            "param_spec",
            "sample_batch_size",
        }
        assert expected_keys <= option_keys

    def test_args_start_with_python_and_script(self, spec: DatasetPipelineSpec) -> None:
        """First arg is the Python executable, second is the generation script."""
        shard = spec.shards[0]

        args = build_generate_args(spec, shard, Path("out"))

        assert args[1] == "src/data/vst/generate_vst_dataset.py"


# ---------------------------------------------------------------------------
# __main__ — fail loud
# ---------------------------------------------------------------------------


class TestMainFailLoud:
    """The module is no longer executable as ``python -m``."""

    def test_running_module_as_main_raises_system_exit(self) -> None:
        """Executing the module's __main__ block raises SystemExit with a pointer to the CLI."""
        import runpy

        with pytest.raises(SystemExit) as exc_info:
            runpy.run_module("pipeline.entrypoints.generate_dataset", run_name="__main__")

        # SystemExit.code carries the message (string), not an int.
        assert "docker_entrypoint" in str(exc_info.value.code)
