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
from hydra.core.global_hydra import GlobalHydra
from hydra.errors import ConfigCompositionException, MissingConfigException
from omegaconf import OmegaConf
from pydantic import ValidationError

from src import generate_dataset
from src.generate_dataset import (
    VST_HEADLESS_WRAPPER,
    _dataset_spec_from_cfg,
    build_generate_args,
    compose_dataset_spec,
    run,
)
from src.pipeline.constants import INPUT_SPEC_FILENAME
from src.pipeline.schemas.spec import DatasetSpec, RenderConfig

# Reusable VST3 bundle with a real Contents/moduleinfo.json so
# extract_renderer_version (called by run) returns a deterministic version
# without loading any .so via pedalboard. Version inside is "1.0.0-test" — the
# specs built in this file pin renderer_version to the same string so the
# constraint check passes.
TEST_PLUGIN_VST3 = Path(__file__).resolve().parent.parent / "fixtures" / "TestPlugin.vst3"
TEST_PLUGIN_VERSION = "1.0.0-test"


# Per-field flag → RenderConfig field name and the type to coerce CLI args back to.
# RenderConfig is strict-mode (no str → int / str → float coercion), so the test must
# cast each flag value before model_validate. Mirrors generate_vst_dataset.main's flags.
_RENDER_FLAG_FIELDS: dict[str, tuple[str, type]] = {
    "--plugin-path": ("plugin_path", str),
    "--preset-path": ("preset_path", str),
    "--param-spec-name": ("param_spec_name", str),
    "--renderer-version": ("renderer_version", str),
    "--sample-rate": ("sample_rate", int),
    "--channels": ("channels", int),
    "--velocity": ("velocity", int),
    "--signal-duration-seconds": ("signal_duration_seconds", float),
    "--min-loudness": ("min_loudness", float),
    "--sample-batch-size": ("sample_batch_size", int),
    "--batch-per-shard": ("batch_per_shard", int),
}


def _render_config_from_cli_args(args: list[str]) -> RenderConfig:
    """Reconstruct a ``RenderConfig`` from the per-field CLI flags emitted by
    build_generate_args."""
    kwargs: dict[str, object] = {}
    for flag, (field_name, caster) in _RENDER_FLAG_FIELDS.items():
        kwargs[field_name] = caster(args[args.index(flag) + 1])
    return RenderConfig.model_validate(kwargs)


def test_render_flag_fields_covers_every_render_config_field() -> None:
    """Adding a new RenderConfig field without updating _RENDER_FLAG_FIELDS must fail loudly."""
    flag_field_names = {field_name for field_name, _ in _RENDER_FLAG_FIELDS.values()}
    assert flag_field_names == set(RenderConfig.model_fields), (
        "drift between _RENDER_FLAG_FIELDS and RenderConfig.model_fields"
    )


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
        """Per-field flags round-trip back to a RenderConfig equal to spec.render."""
        mock_check_call.side_effect = _materialize_shard
        run(spec)

        mock_check_call.assert_called_once()
        args = mock_check_call.call_args[0][0]
        assert any("generate_vst_dataset.py" in a for a in args)
        restored = _render_config_from_cli_args(args)
        assert restored == spec.render

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

    def test_per_field_flags_carry_full_render_config(self, spec: DatasetSpec) -> None:
        """The per-field flags hold every RenderConfig field — full round-trip."""
        shard = spec.shards[0]

        args = build_generate_args(spec, shard, Path("out"))

        restored = _render_config_from_cli_args(args)
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
        list_cfg = OmegaConf.create([1, 2, 3])

        with pytest.raises(TypeError, match="composed Hydra config is not a mapping"):
            _dataset_spec_from_cfg(list_cfg)  # type: ignore[arg-type]

    def test_dataset_spec_from_cfg_missing_required_fields_raises_validation_error(
        self,
    ) -> None:
        """A config dict missing a required DatasetSpec field surfaces as ValidationError.

        ``_dataset_spec_from_cfg`` is the trust boundary between Hydra composition and the
        Pydantic spec; a YAML that omits required fields must surface a clean Pydantic
        error rather than crashing inside the spec constructor. Provides enough fields
        for the runtime default_factories (run_id, r2_prefix) to fire so the failure
        attributes to the missing field, not to a default_factory that crashed first.
        """
        cfg = OmegaConf.create(
            {
                "task_name": "missing-render-and-sizes",
                "output_format": "hdf5",
                "base_seed": 42,
                "r2_bucket": "intermediate-data",
                # train_val_test_sizes and render intentionally omitted
            }
        )

        with pytest.raises(ValidationError):
            _dataset_spec_from_cfg(cfg)  # type: ignore[arg-type]

    def test_dataset_spec_from_cfg_unknown_key_raises_validation_error(self) -> None:
        """Unknown top-level keys reach DatasetSpec's ``extra="forbid"`` boundary.

        The adapter strips only known Hydra group sub-trees (``data``, ``hydra``); any
        other top-level key — typically a typo in YAML — must surface as a Pydantic
        ``ValidationError`` rather than being silently dropped. This pins a fail-closed
        contract at the Hydra → Pydantic trust boundary.
        """
        cfg = OmegaConf.create(
            {
                "task_name": "with-typo",
                "output_format": "hdf5",
                "base_seed": 42,
                "r2_bucket": "intermediate-data",
                # Deliberate typo: not a DatasetSpec field, not a known Hydra group key.
                "trian_val_test_sizes": [80, 10, 10],
            }
        )

        with pytest.raises(ValidationError):
            _dataset_spec_from_cfg(cfg)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# compose_dataset_spec — Hydra composition entrypoint
# ---------------------------------------------------------------------------


class TestComposeDatasetSpec:
    """compose_dataset_spec materializes a DatasetSpec from a Hydra experiment."""

    def test_composes_known_experiment(self) -> None:
        """``datagen/ci-materialize-test`` composes to a 3-shard hdf5 spec."""
        spec = compose_dataset_spec("datagen/ci-materialize-test")
        assert spec.task_name == "ci-materialize-test"
        assert spec.output_format == "hdf5"
        assert spec.num_shards == 3

    def test_overrides_apply(self) -> None:
        """Hydra-style overrides modify nested fields on the composed spec."""
        spec = compose_dataset_spec(
            "datagen/ci-materialize-test", overrides=["render.sample_rate=22050"]
        )
        assert spec.render.sample_rate == 22050

    def test_unknown_experiment_raises_missing_config(self) -> None:
        """A typo'd experiment name surfaces as Hydra's MissingConfigException, not a KeyError."""
        with pytest.raises(MissingConfigException):
            compose_dataset_spec("datagen/does-not-exist")

    def test_invalid_override_field_raises_composition_error(self) -> None:
        """Overrides targeting a non-existent field raise ConfigCompositionException."""
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
        with pytest.raises(MissingConfigException):
            compose_dataset_spec("datagen/does-not-exist")

        assert not GlobalHydra.instance().is_initialized()

    def test_consecutive_composes_are_idempotent(self) -> None:
        """Calling compose twice produces equivalent specs — GlobalHydra reset works."""
        spec_a = compose_dataset_spec("datagen/ci-materialize-test")
        spec_b = compose_dataset_spec("datagen/ci-materialize-test")

        assert spec_a.task_name == spec_b.task_name
        assert spec_a.num_shards == spec_b.num_shards
        assert spec_a.render == spec_b.render


# ---------------------------------------------------------------------------
# load_spec_from_uri — local path / r2:// dispatch
# ---------------------------------------------------------------------------


class TestLoadSpecFromUri:
    """load_spec_from_uri() reads local paths directly and r2:// via downloaded_to_tempfile."""

    def test_local_path_reads_directly(self, tmp_path: Path) -> None:
        """Local paths bypass rclone and parse the file in place."""
        spec_path = tmp_path / "spec.json"
        spec = DatasetSpec(**_base_spec_kwargs(tmp_path))  # type: ignore[arg-type]
        spec_path.write_text(spec.model_dump_json())

        loaded = generate_dataset.load_spec_from_uri(str(spec_path))

        assert loaded.task_name == spec.task_name
        assert loaded.num_shards == spec.num_shards

    def test_local_malformed_json_raises_validation_error(self, tmp_path: Path) -> None:
        """A local spec file containing malformed JSON surfaces a Pydantic ValidationError."""
        spec_path = tmp_path / "spec.json"
        spec_path.write_text('{"task_name": "missing-required-fields"}')

        with pytest.raises(ValidationError):
            generate_dataset.load_spec_from_uri(str(spec_path))

    def test_local_non_json_payload_raises_validation_error(self, tmp_path: Path) -> None:
        """Non-JSON text from the trust boundary also surfaces a clean ValidationError."""
        spec_path = tmp_path / "spec.json"
        spec_path.write_text("this is not json")

        with pytest.raises(ValidationError):
            generate_dataset.load_spec_from_uri(str(spec_path))

    def test_r2_uri_dispatches_through_downloaded_to_tempfile(self, tmp_path: Path) -> None:
        """R2:// URIs delegate to ``src.pipeline.r2_io.downloaded_to_tempfile``.

        Critical regression: an earlier implementation called ``rclone copy`` against a tempdir
        and then read ``Path(tmpdir) / Path(uri).name``, which broke for r2:// URIs whose key
        contained subdirectory components (rclone copy mirrors directory structure under the
        destination). ``downloaded_to_tempfile`` uses ``rclone copyto`` with an explicit
        destination path, sidestepping the subdirectory hazard.
        """
        spec = DatasetSpec(**_base_spec_kwargs(tmp_path))  # type: ignore[arg-type]
        spec_json = spec.model_dump_json()

        captured_args: list[list[str]] = []

        def fake_check_call(args: list[str]) -> None:
            captured_args.append(args)
            # downloaded_to_tempfile uses rclone copyto, so the last arg is the target file path.
            Path(args[-1]).write_text(spec_json)

        with patch("src.pipeline.r2_io.subprocess.check_call", side_effect=fake_check_call):
            loaded = generate_dataset.load_spec_from_uri(
                "r2://bucket/skypilot-launcher-specs/cluster-abc.json"
            )

        assert loaded.task_name == spec.task_name
        assert len(captured_args) == 1
        assert captured_args[0][1] == "copyto"
        assert captured_args[0][-2] == "r2:bucket/skypilot-launcher-specs/cluster-abc.json"
