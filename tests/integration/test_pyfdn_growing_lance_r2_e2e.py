"""Real public-CLI pyFDN, R2, growing refresh, and checkpoint-resume E2E."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import lance
import numpy as np
import pytest
import torch

from synth_setter.param_spec_name import ParamSpecName
from synth_setter.pipeline import r2_io
from synth_setter.pipeline.data.growing_lance import ActiveGrowingSnapshot
from synth_setter.pipeline.schemas.spec import DatasetSpec, RenderConfig
from synth_setter.pipeline.spec_io import upload_spec
from synth_setter.synth_spec import SynthName, SynthSpec

pytestmark = [pytest.mark.integration_r2, pytest.mark.r2, pytest.mark.slow]


def _run(argv: list[str], *, timeout: int = 900) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(  # noqa: S603 — argv contains installed CLIs and test-owned values.
        argv, capture_output=True, text=True, check=False, timeout=timeout
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return result


def _wait_for_version(
    active_path: Path, prior_version: int, process: subprocess.Popen[str]
) -> int:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        assert process.poll() is None, "training exited before growing activation"
        try:
            active = ActiveGrowingSnapshot.model_validate_json(active_path.read_bytes())
        except (OSError, ValueError):
            time.sleep(0.1)
            continue
        if active.remote_version != prior_version:
            return active.remote_version
        time.sleep(0.1)
    raise AssertionError("background materializer did not activate the ready version")


def _train_command(
    dataset_root: Path,
    active_path: Path,
    output_dir: Path,
    *,
    max_epochs: int,
    checkpoint: Path | None = None,
) -> list[str]:
    command = [
        "synth-setter-train",
        "experiment=pyfdn/flow",
        "trainer=cpu",
        f"datamodule.dataset_root={dataset_root}",
        f"training.growing_active_record={active_path}",
        "training.growing_refresh_epoch_interval=1",
        "training.val_audio_probe=false",
        "datamodule.batch_size=1",
        "datamodule.num_workers=0",
        "datamodule.persistent_workers=false",
        f"+trainer.max_epochs={max_epochs}",
        f"+trainer.max_steps={max_epochs}",
        "+trainer.limit_train_batches=1",
        "trainer.limit_val_batches=0",
        "callbacks.model_checkpoint.save_last=true",
        f"callbacks.model_checkpoint.dirpath={output_dir / 'checkpoints'}",
        f"paths.output_dir={output_dir}",
        f"hydra.run.dir={output_dir}",
        "logger=csv",
        "test=false",
    ]
    if checkpoint is not None:
        command.append(f"ckpt_path={checkpoint}")
    return command


def test_pyfdn_r2_public_clis_refresh_at_epoch_boundary_and_resume_checkpoint(
    tmp_path: Path,
) -> None:
    """Public processes publish, poll, adopt, checkpoint, and resume exact data identity.

    :param tmp_path: Local operator, materialization, and training workspace.
    """
    assert r2_io.is_r2_reachable(), "real R2 credentials are required"
    for command in (
        "synth-setter-finalize-dataset",
        "synth-setter-generate-dataset-from-spec-uri",
        "synth-setter-growing-lance",
        "synth-setter-train",
    ):
        assert shutil.which(command), f"installed public CLI is missing: {command}"
    prefix = f"ci-growing-pyfdn/{os.environ.get('GITHUB_RUN_ID', 'local')}/{uuid.uuid4().hex}/"
    spec = DatasetSpec.model_validate(
        {
            "task_name": "pyfdn-growing-r2-e2e",
            "output_format": "lance",
            "train_val_test_sizes": [2, 1, 1],
            "base_seed": 3090,
            "mask_degenerate_bins": True,
            "r2": {"bucket": "intermediate-data", "prefix": prefix},
            "render": RenderConfig(
                synth=SynthSpec(
                    name=SynthName("pyfdn_n8_mono_householder"),
                    param_spec_name=ParamSpecName("pyfdn_n8_mono_householder"),
                    plugin_path="pyfdn",
                    plugin_state_path="",
                    synth_version="0.4.2",
                ),
                renderer_backend="pyfdn",
                pyfdn_excitation="impulse",
                sample_rate=44_100,
                channels=1,
                velocity=0,
                signal_duration_seconds=4.0,
                min_loudness=-100.0,
                audio_dtype="float32",
                mel_spec_dtype="float32",
                samples_per_render_batch=1,
                samples_per_shard=1,
                base_seed=3090,
                attempts_per_sample=100,
                param_sample_cadence="sample",
                plugin_reload_cadence="render",
                gui_toggle_cadence="never",
            ).model_dump(mode="json"),
        }
    )
    r2_io.ensure_r2_env_loaded()
    upload_spec(spec)
    processes: list[subprocess.Popen[str]] = []
    try:
        _run(["synth-setter-generate-dataset-from-spec-uri", spec.r2.input_spec_uri()])
        _run(
            [
                "synth-setter-finalize-dataset",
                f"dataset_root_uri=r2://{spec.r2.bucket}/{spec.r2.prefix}",
                "logger=[]",
            ]
        )
        operator = tmp_path / "operator"
        growing = ["synth-setter-growing-lance"]
        _run(
            growing
            + [
                "init",
                spec.r2.input_spec_uri(),
                "--branch",
                "growing-e2e",
                "--max-train-shards",
                "3",
                "--num-extra-shards",
                "1",
                "--work-dir",
                str(operator),
            ]
        )
        local_root = tmp_path / "growing-local"
        materialize = growing + [
            "materialize",
            spec.r2.input_spec_uri(),
            "--branch",
            "growing-e2e",
            "--local-root",
            str(local_root),
            "--work-dir",
            str(operator / "materializer"),
        ]
        _run(materialize)
        active_path = local_root / "active.json"
        initial = ActiveGrowingSnapshot.model_validate_json(active_path.read_bytes())

        baseline_root = tmp_path / "baseline"
        _run(
            [
                "rclone",
                "copy",
                f"r2:{spec.r2.bucket}/{prefix}",
                str(baseline_root),
                "--checksum",
            ]
        )
        materializer = subprocess.Popen(  # noqa: S603 — fixed public CLI with test-owned paths.
            materialize + ["--poll-seconds", "0.1"], text=True
        )
        processes.append(materializer)
        train_root = tmp_path / "train"
        trainer = subprocess.Popen(  # noqa: S603 — fixed public CLI with test-owned overrides.
            _train_command(baseline_root, active_path, train_root, max_epochs=300),
            text=True,
        )
        processes.append(trainer)
        # One grow driver + N generators, all polling daemons: the plug-and-play
        # producer topology the runbook documents. Every producer exits on its
        # own once the branch reaches capacity.
        grower = subprocess.Popen(  # noqa: S603 — fixed public CLI with test-owned values.
            growing
            + [
                "grow",
                spec.r2.input_spec_uri(),
                "--branch",
                "growing-e2e",
                "--work-dir",
                str(operator / "grow"),
                "--poll-seconds",
                "0.2",
            ],
            text=True,
        )
        processes.append(grower)
        generate_command = growing + [
            "generate",
            spec.r2.input_spec_uri(),
            "--branch",
            "growing-e2e",
            "--work-dir",
            str(operator / "generate"),
            "--poll-seconds",
            "0.2",
        ]
        generators = [
            subprocess.Popen(  # noqa: S603 — fixed public CLI with test-owned values.
                generate_command, text=True
            )
            for _ in range(2)
        ]
        processes.extend(generators)
        for generator in generators:
            assert generator.wait(timeout=900) == 0
        assert grower.wait(timeout=900) == 0
        adopted_version = _wait_for_version(active_path, initial.remote_version, trainer)
        adopted = ActiveGrowingSnapshot.model_validate_json(active_path.read_bytes())
        assert initial.row_count == 2
        assert adopted.row_count == 3
        assert adopted.dataset_path == initial.dataset_path
        train_target, storage_options = r2_io.lance_target(spec.r2.split_lance_uri("train"))
        remote_train = lance.dataset(train_target, storage_options=storage_options)
        assert remote_train.version == 1
        assert remote_train.count_rows() == 2
        baseline_branch = remote_train.checkout_version(("growing-e2e", initial.remote_version))
        refreshed = remote_train.checkout_version(("growing-e2e", adopted_version))
        baseline_files = [
            fragment.metadata.files[0].path for fragment in baseline_branch.get_fragments()
        ]
        refreshed_files = [
            fragment.metadata.files[0].path for fragment in refreshed.get_fragments()
        ]
        assert baseline_branch.count_rows() == 2
        assert refreshed.count_rows() == 3
        assert refreshed_files[: len(baseline_files)] == baseline_files
        audio = np.asarray(refreshed.to_table(columns=["audio"])["audio"].to_pylist())
        assert np.isfinite(audio).all()
        assert np.any(audio != 0)
        for split in ("val", "test"):
            target, options = r2_io.lance_target(spec.r2.split_lance_uri(split))
            pinned = lance.dataset(target, storage_options=options)
            assert pinned.version == 1
            assert pinned.count_rows() == 1
        assert trainer.wait(timeout=900) == 0
        materializer.terminate()
        materializer.wait(timeout=30)

        checkpoint = train_root / "checkpoints" / "last.ckpt"
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = saved["LanceVSTDataModule"]
        assert tuple(state["growing_history"]) == (
            initial.remote_version,
            adopted_version,
        )
        assert state["growing_active_snapshot"]["remote_version"] == adopted_version

        resume_root = tmp_path / "resume"
        _run(
            _train_command(
                baseline_root,
                active_path,
                resume_root,
                max_epochs=301,
                checkpoint=checkpoint,
            )
        )
        resumed = torch.load(
            resume_root / "checkpoints" / "last.ckpt",
            map_location="cpu",
            weights_only=False,
        )["LanceVSTDataModule"]
        assert resumed["growing_active_snapshot"]["remote_version"] == adopted_version
        assert tuple(resumed["growing_history"]) == (
            initial.remote_version,
            adopted_version,
        )
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
        subprocess.run(  # noqa: S603 — teardown URI is unique and test-owned.
            [  # noqa: S607 — rclone is a required integration-test executable.
                "rclone",
                "delete",
                f"r2:{spec.r2.bucket}/{prefix}",
                "--checksum",
            ],
            capture_output=True,
            check=False,
            text=True,
        )
