"""Tests for pipeline/entrypoints/generate_dataset.py — spec-driven run.

The entrypoint's public surface is a single ``run(spec)`` function that:
  1. Serializes the spec to a tempfile.
  2. Uploads the spec to R2 at ``r2:{bucket}/{prefix}/input_spec.json``.
  3. Shells out to ``generate_vst_dataset.py`` to produce the shard.
  4. Uploads the shard to R2 at ``r2:{bucket}/{prefix}/``.

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

    def test_num_shards_greater_than_one_raises(self, tmp_path: Path) -> None:
        """num_shards > 1 raises NotImplementedError (multi-shard is orchestrator territory)."""
        multi_shards = tuple(
            ShardSpec(shard_id=i, filename=f"shard-{i:06d}.h5", seed=42 + i) for i in range(3)
        )
        kwargs = _base_spec_kwargs(
            tmp_path,
            splits={"train": 1, "val": 1, "test": 1},
            shards=multi_shards,
        )
        spec = DatasetPipelineSpec(**kwargs)  # type: ignore[arg-type]

        with pytest.raises(NotImplementedError, match="num_shards > 1"):
            run(spec)

    def test_wds_output_format_raises(self, tmp_path: Path) -> None:
        """output_format 'wds' raises ValueError — generate_vst_dataset only supports hdf5."""
        kwargs = _base_spec_kwargs(tmp_path, output_format="wds")
        spec = DatasetPipelineSpec(**kwargs)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="only supports hdf5"):
            run(spec)

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
