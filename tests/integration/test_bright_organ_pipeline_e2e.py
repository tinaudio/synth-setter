"""Real R2 brightOrgan generation, finalization, training, and evaluation E2E."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import lance
import numpy as np
import pandas as pd
import pytest
import torch
from pedalboard.io import AudioFile
from PIL import Image

from synth_setter.data.vst.shapes import AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.lance_staging import shard_has_complete_attempt
from synth_setter.pipeline.schemas.spec import DatasetSpec, Split
from synth_setter.pipeline.spec_io import load_spec_from_root

pytestmark = [pytest.mark.integration_r2, pytest.mark.r2, pytest.mark.slow]

_TASK_NAME = "bright-organ-e2e"
_BASE_SEED = 1808
_GENERATE_TIMEOUT_SECONDS = 300
_FINALIZE_TIMEOUT_SECONDS = 180
_TRAIN_TIMEOUT_SECONDS = 300
_EVAL_TIMEOUT_SECONDS = 300


def _run_id() -> str:
    """Return an R2-isolated run identifier.

    :returns: Run identifier unique across local and CI attempts.
    """
    github_run_id = os.environ.get("GITHUB_RUN_ID", "local")
    github_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "0")
    return f"bright-organ-{github_run_id}-{github_attempt}-{uuid.uuid4().hex[:8]}"


def _cli(name: str) -> str:
    """Resolve one installed command-line executable.

    :param name: Executable basename.
    :returns: Absolute executable path.
    :raises RuntimeError: If the active environment does not expose ``name``.
    """
    executable = Path(sys.executable).with_name(name)
    if executable.is_file():
        return str(executable)
    resolved = shutil.which(name)
    if resolved is None:
        raise RuntimeError(f"required public CLI is unavailable: {name}")
    return resolved


def _run_cli(argv: list[str], *, timeout: int, stage: str) -> None:
    """Run one production CLI and expose both output streams on failure.

    :param argv: Complete public-CLI argv.
    :param timeout: Wall-clock ceiling in seconds.
    :param stage: Human-readable pipeline stage for failure output.
    :raises AssertionError: If the command fails or exceeds ``timeout``.
    """
    try:
        result = subprocess.run(  # noqa: S603
            argv,
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        raise AssertionError(
            f"{stage} timed out after {timeout}s\n"
            f"--- STDOUT (tail) ---\n{stdout[-2000:]}\n"
            f"--- STDERR (tail) ---\n{stderr[-2000:]}"
        ) from error
    assert result.returncode == 0, (
        f"{stage} exited {result.returncode}\n"
        f"--- STDOUT (tail) ---\n{result.stdout[-2000:]}\n"
        f"--- STDERR (tail) ---\n{result.stderr[-2000:]}"
    )


def _purge_r2_prefix(spec: DatasetSpec) -> None:
    """Best-effort exact-prefix cleanup after the production-path run.

    :param spec: Generated dataset identity whose prefix can be deleted safely.
    """
    result = subprocess.run(  # noqa: S603
        [
            _cli("rclone"),
            "purge",
            f"r2:{spec.r2.bucket}/{spec.r2.prefix}",
            "--checksum",
            "--contimeout=10s",
            "--timeout=60s",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        sys.stderr.write(
            f"WARN: exact-prefix cleanup exited {result.returncode} for "
            f"r2:{spec.r2.bucket}/{spec.r2.prefix}\n{result.stderr[-1000:]}\n"
        )


@contextmanager
def _generated_spec(tmp_path: Path) -> Iterator[DatasetSpec]:
    """Generate real brightOrgan rows and yield their R2-backed spec.

    :yields DatasetSpec: Generated spec, cleaned up after the test.
    :param tmp_path: Per-test local work area.
    """
    probe = subprocess.run(  # noqa: S603
        [_cli("rclone"), "lsd", "r2:", "--checksum"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert probe.returncode == 0, (
        "real R2 prerequisite probe failed\n"
        f"--- STDOUT ---\n{probe.stdout[-2000:]}\n"
        f"--- STDERR ---\n{probe.stderr[-2000:]}"
    )

    run_id = _run_id()
    generate_dir = tmp_path / "generate"
    generate_dir.mkdir()
    _run_cli(
        [
            _cli("synth-setter-generate-dataset"),
            "experiment=generate_dataset/smoke-shard-lance",
            f"task_name={_TASK_NAME}",
            f"run_id={run_id}",
            "synth=faust_bright_organ",
            "render=faust_bright_organ",
            "train_val_test_sizes=[2,2,2]",
            "render.samples_per_shard=2",
            "render.samples_per_render_batch=1",
            "mask_degenerate_bins=true",
            "~logger",
            f"base_seed={_BASE_SEED}",
            f"train_val_test_seeds=[{_BASE_SEED},{_BASE_SEED + 1},{_BASE_SEED + 2}]",
            f"paths.output_dir={generate_dir}",
            "hydra.job.chdir=false",
        ],
        timeout=_GENERATE_TIMEOUT_SECONDS,
        stage="brightOrgan generation",
    )
    root_uri = f"r2://intermediate-data/data/{_TASK_NAME}/{run_id}/"
    spec = load_spec_from_root(root_uri)
    try:
        assert spec.render.param_spec_name == "faust_bright_organ"
        assert spec.train_val_test_sizes == (2, 2, 2)
        assert spec.render.samples_per_shard == 2
        assert spec.render.samples_per_render_batch == 1
        for shard in spec.shards:
            assert shard_has_complete_attempt(spec, shard.shard_id), (
                f"generation did not complete staged shard {shard.shard_id}"
            )
        yield spec
    finally:
        _purge_r2_prefix(spec)


def _assert_finalized_lance_split(spec: DatasetSpec, split: Split) -> None:
    """Read one finalized R2 Lance split and assert its production contract.

    :param spec: Finalized R2 dataset identity.
    :param split: Train, validation, or test split name.
    """
    target, storage_options = r2_io.lance_target(spec.r2.split_lance_uri(split))
    table = lance.dataset(target, storage_options=storage_options).to_table(
        columns=[AUDIO_FIELD, MEL_SPEC_FIELD, PARAM_ARRAY_FIELD]
    )
    audio = table.column(AUDIO_FIELD).combine_chunks().to_numpy_ndarray()
    mel = table.column(MEL_SPEC_FIELD).combine_chunks().to_numpy_ndarray()
    params = table.column(PARAM_ARRAY_FIELD).combine_chunks().to_numpy_ndarray()

    assert table.num_rows == 2
    assert audio.shape == (2, 2, 176_400)
    assert audio.dtype == np.float16
    assert mel.shape == (2, 2, 128, 401)
    assert mel.dtype == np.float32
    assert params.shape == (2, 13)
    assert params.dtype == np.float32
    assert np.isfinite(audio).all()
    assert np.isfinite(mel).all()
    assert np.isfinite(params).all()
    assert np.all((params >= 0.0) & (params <= 1.0))
    assert float(np.max(np.abs(audio))) > 1e-4


def test_bright_organ_generate_finalize_train_eval_production_path(tmp_path: Path) -> None:
    """Run real Faust generation through R2 Lance training and rendered prediction.

    :param tmp_path: Per-test directory for real CLI output and hydrated Lance data.
    """
    with _generated_spec(tmp_path) as spec:
        root_uri = spec.r2.input_spec_uri().removesuffix("input_spec.json")
        finalize_dir = tmp_path / "finalize"
        finalize_dir.mkdir()
        _run_cli(
            [
                _cli("synth-setter-finalize-dataset"),
                f"dataset_root_uri={root_uri}",
                "~logger",
                f"paths.output_dir={finalize_dir}",
                "hydra.job.chdir=false",
            ],
            timeout=_FINALIZE_TIMEOUT_SECONDS,
            stage="brightOrgan finalization",
        )

        assert r2_io.object_size(spec.r2.dataset_complete_marker_uri()) is not None
        assert r2_io.object_size(spec.r2.stats_uri()) is not None
        with r2_io.downloaded_to_tempfile(spec.r2.stats_uri()) as stats_path:
            with np.load(stats_path) as stats:
                assert stats["mean"].shape == (2, 128, 401)
                assert stats["std"].shape == (2, 128, 401)
                assert np.isfinite(stats["mean"]).all()
                assert np.isfinite(stats["std"]).all()
        for split in ("train", "val", "test"):
            _assert_finalized_lance_split(spec, split)

        train_dir = tmp_path / "train"
        train_dir.mkdir()
        _run_cli(
            [
                _cli("synth-setter-train"),
                "experiment=surge/ffn_simple",
                "trainer=cpu",
                "callbacks=model_checkpoint",
                "logger=[]",
                f"seed={_BASE_SEED}",
                "test=false",
                "training.val_audio_probe=false",
                f"datamodule.download_dataset_root_uri={root_uri}",
                f"datamodule.dataset_root={train_dir / 'data'}",
                "synth=faust_bright_organ",
                "datamodule.batch_size=1",
                "datamodule.num_workers=0",
                "datamodule.persistent_workers=false",
                "trainer.max_steps=1",
                "trainer.min_steps=1",
                "+trainer.limit_train_batches=1",
                "trainer.limit_val_batches=1",
                f"callbacks.model_checkpoint.dirpath={train_dir / 'checkpoints'}",
                "callbacks.model_checkpoint.save_last=true",
                "callbacks.model_checkpoint.every_n_train_steps=1",
                "model.net.d_model=16",
                "model.net.n_heads=1",
                "model.net.n_layers=1",
                "model.compile=false",
                f"paths.output_dir={train_dir}",
                "hydra.job.chdir=false",
            ],
            timeout=_TRAIN_TIMEOUT_SECONDS,
            stage="brightOrgan CPU training",
        )
        checkpoint_path = train_dir / "checkpoints" / "last.ckpt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert checkpoint["state_dict"]
        assert checkpoint["global_step"] >= 1

        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        _run_cli(
            [
                _cli("synth-setter-eval"),
                "experiment=surge/ffn_simple",
                "trainer=cpu",
                "callbacks=eval_vst",
                "logger=[]",
                "mode=predict",
                f"seed={_BASE_SEED}",
                f"ckpt_path={checkpoint_path}",
                "render=faust_bright_organ",
                "evaluation.render_vst=true",
                "evaluation.rerender_target=true",
                "evaluation.compute_metrics=true",
                "evaluation.num_workers=1",
                f"datamodule.download_dataset_root_uri={root_uri}",
                f"datamodule.dataset_root={eval_dir / 'data'}",
                "synth=faust_bright_organ",
                "datamodule.batch_size=1",
                "datamodule.num_workers=0",
                "datamodule.persistent_workers=false",
                "+trainer.limit_predict_batches=1",
                "model.net.d_model=16",
                "model.net.n_heads=1",
                "model.net.n_layers=1",
                "model.compile=false",
                f"paths.output_dir={eval_dir}",
                "hydra.job.chdir=false",
            ],
            timeout=_EVAL_TIMEOUT_SECONDS,
            stage="brightOrgan prediction and postprocessing",
        )

        prediction = torch.load(
            eval_dir / "predictions" / "pred-0.pt", map_location="cpu", weights_only=True
        )
        assert prediction.shape == (1, 13)
        assert torch.isfinite(prediction).all()
        assert torch.isfinite(
            torch.load(
                eval_dir / "predictions" / "target-audio-0.pt",
                map_location="cpu",
                weights_only=True,
            )
        ).all()
        params = pd.read_csv(eval_dir / "audio" / "sample_0" / "params.csv", index_col=0)
        assert {"pred", "target"} <= set(params.columns)
        numeric_params = params.drop(index="note_start_and_end")
        numeric_values = numeric_params[["pred", "target"]].apply(pd.to_numeric)
        assert np.isfinite(numeric_values.to_numpy()).all()
        with Image.open(eval_dir / "audio" / "sample_0" / "spec.png") as image:
            image.verify()
        with AudioFile(str(eval_dir / "audio" / "sample_0" / "target.wav")) as target_file:
            target_audio = target_file.read(target_file.frames)
        with AudioFile(str(eval_dir / "audio" / "sample_0" / "pred.wav")) as prediction_file:
            predicted_audio = prediction_file.read(prediction_file.frames)
        assert np.isfinite(target_audio).all()
        assert np.isfinite(predicted_audio).all()
        assert float(np.max(np.abs(target_audio))) > 1e-4
        metrics = pd.read_csv(eval_dir / "metrics" / "metrics.csv")
        assert not metrics.empty
        assert np.isfinite(metrics.select_dtypes(include=[np.number]).to_numpy()).all()
