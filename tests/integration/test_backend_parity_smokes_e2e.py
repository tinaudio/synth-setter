"""Real-R2 production-path coverage for backend-parity dataset presets."""

from __future__ import annotations

import json
import math
import os
import platform
import re
import subprocess
import sys
import uuid
from pathlib import Path

import lance
import numpy as np
import pytest
from pedalboard.io import AudioFile

from synth_setter.pipeline import r2_io
from synth_setter.pipeline.partitioning import available_cpus
from synth_setter.pipeline.schemas.spec import DatasetSpec
from synth_setter.pipeline.spec_io import find_input_specs

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SURGE_PLUGIN = (_REPO_ROOT / "plugins/Surge XT.vst3").resolve()
_KR106_PLUGIN = (_REPO_ROOT / "plugins/Ultramaster KR-106.vst3").resolve()

pytestmark = [
    pytest.mark.integration_r2,
    pytest.mark.r2,
    pytest.mark.requires_vst,
    pytest.mark.slow,
]


@pytest.mark.parametrize(
    ("experiment", "source_plugin", "candidate_plugin"),
    [
        pytest.param(
            "surge-simple-pedalboard-to-dawdreamer-smoke",
            _SURGE_PLUGIN,
            _SURGE_PLUGIN,
            id="surge-dawdreamer",
        ),
        pytest.param(
            "surge-simple-pedalboard-to-surgepy-smoke",
            _SURGE_PLUGIN,
            None,
            marks=pytest.mark.requires_surgepy,
            id="surge-surgepy",
        ),
        pytest.param(
            "ultramaster-kr106-pedalboard-to-dawdreamer-smoke",
            _KR106_PLUGIN,
            _KR106_PLUGIN,
            id="kr106-dawdreamer",
        ),
    ],
)
def test_backend_parity_smoke_real_cli_persists_paired_probe(
    tmp_path: Path,
    experiment: str,
    source_plugin: Path,
    candidate_plugin: Path | None,
) -> None:
    """A named preset persists source/candidate renders of every finalized row.

    :param tmp_path: Isolated local output and download root.
    :param experiment: Hydra experiment preset name.
    :param source_plugin: Source VST3 bundle path.
    :param candidate_plugin: Candidate VST3 bundle path, or ``None`` for SurgePy.
    """
    _require_runtime(experiment, source_plugin)
    run_id = f"parity-{uuid.uuid4().hex}"
    prefix_root = f"test-runs/backend-parity/{run_id}"
    run_dir = tmp_path / "run"
    overrides = [
        f"experiment=generate_dataset/{experiment}",
        f"r2.prefix_root={prefix_root}",
        f"run_id={run_id}",
        f"hydra.run.dir={run_dir}",
        f"synth.plugin_path={source_plugin}",
    ]
    if candidate_plugin is not None:
        overrides.append(f"oracle_eval.candidate.synth.plugin_path={candidate_plugin}")

    env = {
        **os.environ,
        "PYTHONPATH": f"{_REPO_ROOT / 'src'}:{os.environ.get('PYTHONPATH', '')}",
        "WANDB_MODE": "offline",
    }
    spec: DatasetSpec | None = None
    dataset_cleanup: tuple[str, str] | None = None
    probe_cleanup: tuple[str, str] | None = None
    try:
        result = subprocess.run(  # noqa: S603 — fixed module and test-owned overrides
            [sys.executable, "-m", "synth_setter.cli.generate_dataset", *overrides],
            cwd=_REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=1_200,
        )
        (tmp_path / "cli.stdout.log").write_text(result.stdout)
        (tmp_path / "cli.stderr.log").write_text(result.stderr)
        spec_paths = find_input_specs(run_dir / "data")
        if spec_paths:
            assert len(spec_paths) == 1
            spec = DatasetSpec.model_validate_json(spec_paths[0].read_text())
            dataset_cleanup = (spec.r2.bucket, spec.r2.prefix)
            probe_prefix = f"probes/dataset-oracle/{spec.task_name}/{spec.run_id}/"
            probe_cleanup = (spec.r2.bucket, probe_prefix)

        assert result.returncode == 0, (
            f"backend parity CLI exited {result.returncode}\n"
            f"stdout tail:\n{result.stdout[-2_000:]}\n"
            f"stderr tail:\n{result.stderr[-2_000:]}"
        )
        dispatch = re.search(r"parallel dispatch: workers=(\d+) shards=4", result.stderr)
        assert dispatch is not None and int(dispatch.group(1)) >= 2
        assert spec is not None
        assert dataset_cleanup is not None and probe_cleanup is not None

        _assert_downloaded_split_rows(spec, tmp_path / "downloaded")
        _assert_local_probe_audio_and_metrics(run_dir, spec)

        probe_root_uri = f"r2://{probe_cleanup[0]}/{probe_cleanup[1]}"
        downloaded_probe = tmp_path / "probe"
        r2_io.download_dir_no_overwrite(probe_root_uri, downloaded_probe)
        _assert_uploaded_provenance(downloaded_probe, spec)
    finally:
        spec_paths = find_input_specs(run_dir / "data")
        if dataset_cleanup is None and spec_paths:
            cleanup_spec = DatasetSpec.model_validate_json(spec_paths[0].read_text())
            dataset_cleanup = (cleanup_spec.r2.bucket, cleanup_spec.r2.prefix)
            probe_cleanup = (
                cleanup_spec.r2.bucket,
                f"probes/dataset-oracle/{cleanup_spec.task_name}/{cleanup_spec.run_id}/",
            )
        if dataset_cleanup is not None:
            r2_io.purge_prefix(*dataset_cleanup)
        if probe_cleanup is not None:
            r2_io.purge_prefix(*probe_cleanup)


def _require_runtime(experiment: str, source_plugin: Path) -> None:
    """Skip before external writes when a real backend prerequisite is absent.

    :param experiment: Experiment whose platform requirements are checked.
    :param source_plugin: Required source VST3 bundle path.
    """
    if sys.platform != "linux" or available_cpus() < 4:
        pytest.skip("parallel backend parity smoke requires Linux and at least four CPUs")
    if not source_plugin.is_dir():
        pytest.skip(f"required VST3 bundle is absent: {source_plugin}")
    if experiment.startswith("ultramaster") and platform.machine() != "x86_64":
        pytest.skip("Ultramaster KR-106 is available only on x86_64")
    if not r2_io.is_r2_reachable():
        pytest.skip("R2 is unreachable through rclone")


def _assert_downloaded_split_rows(spec: DatasetSpec, destination: Path) -> None:
    """Download finalized Lance splits and verify their configured row counts.

    :param spec: Dataset spec identifying the finalized R2 splits.
    :param destination: Local download root.
    """
    for split, expected_rows in zip(("train", "val", "test"), (8, 4, 4), strict=True):
        split_dir = destination / f"{split}.lance"
        r2_io.download_dir_no_overwrite(spec.r2.split_lance_uri(split), split_dir)
        assert lance.dataset(str(split_dir)).count_rows() == expected_rows


def _assert_local_probe_audio_and_metrics(run_dir: Path, spec: DatasetSpec) -> None:
    """Verify both roles rendered every split and retained exact oracle parameters.

    :param run_dir: Generate-dataset Hydra run directory.
    :param spec: Dataset spec identifying the evaluation run.
    """
    for role in ("source", "candidate"):
        for split, expected_rows in zip(("train", "val", "test"), (8, 4, 4), strict=True):
            eval_dir = run_dir / "oracle_eval" / role / split / spec.run_id
            _assert_split_probe(eval_dir, expected_rows, role)


def _assert_split_probe(eval_dir: Path, expected_rows: int, role: str) -> None:
    """Verify one role/split artifact set is complete and namespaced.

    :param eval_dir: Local role/split evaluation directory.
    :param expected_rows: Required number of rendered sample directories.
    :param role: Metric namespace expected in the artifact set.
    """
    samples = sorted((eval_dir / "audio").glob("sample_*"))
    assert len(samples) == expected_rows
    for sample in samples:
        _assert_sample_audio(sample)

    _assert_metrics(eval_dir / "metrics/metrics.json", role=role)


def _assert_metrics(metrics_path: Path, *, role: str | None = None) -> None:
    """Verify persisted metrics are finite, namespaced, and exactly oracle-correct.

    :param metrics_path: Persisted JSON metrics path.
    :param role: Required metric prefix, or ``None`` to accept any namespace.
    """
    metrics = json.loads(metrics_path.read_text())
    if role is not None:
        assert all(key.startswith(f"{role}/") for key in metrics)
    assert metrics
    assert all(isinstance(value, float) and math.isfinite(value) for value in metrics.values())
    param_mse = [value for key, value in metrics.items() if key.endswith("param_mse")]
    assert param_mse == [0.0]


def _assert_sample_audio(sample: Path) -> None:
    """Verify one paired target/prediction waveform is finite and audible.

    :param sample: Sample directory containing target and prediction WAVs.
    """
    for filename in ("target.wav", "pred.wav"):
        _assert_audio_file(sample / filename)


def _assert_audio_file(audio_path: Path) -> None:
    """Verify one persisted waveform is finite and audible.

    :param audio_path: WAV artifact to decode and validate.
    """
    audio = _read_audio_file(audio_path)
    assert np.isfinite(audio).all()
    assert np.any(audio != 0.0)


def _read_audio_file(audio_path: Path) -> np.ndarray:
    """Read a persisted waveform through the production audio decoder.

    :param audio_path: WAV artifact to decode.
    :return: Decoded channel-first audio samples.
    """
    with AudioFile(str(audio_path)) as audio_file:
        return audio_file.read(audio_file.frames)


def _assert_uploaded_provenance(probe_root: Path, spec: DatasetSpec) -> None:
    """Verify uploaded source and candidate records reference the same finalized rows.

    :param probe_root: Downloaded probe namespace root.
    :param spec: Dataset spec defining expected provenance.
    """
    provenance_files = sorted(probe_root.rglob("provenance.json"))
    metrics_files = sorted(probe_root.rglob("metrics.json"))
    audio_files = sorted(probe_root.rglob("*.wav"))
    assert len(provenance_files) == 6
    assert len(metrics_files) == 6
    assert len(audio_files) == 64
    assert not any(path.name == "predictions" for path in probe_root.rglob("*"))
    for metrics_file in metrics_files:
        _assert_metrics(metrics_file)
    for audio_file in audio_files:
        _assert_audio_file(audio_file)

    probes = [(path.parent, json.loads(path.read_text())) for path in provenance_files]
    records = [record for _, record in probes]
    source_render = spec.render.model_dump(mode="json")
    assert {record["source_dataset_task"] for record in records} == {spec.task_name}
    assert {record["source_run_id"] for record in records} == {spec.run_id}
    assert all(record["source_render"] == source_render for record in records)
    candidate_backends = {record["candidate_render"]["renderer_backend"] for record in records}
    assert "pedalboard" in candidate_backends
    assert len(candidate_backends) == 2
    for split, expected_rows in zip(("train", "val", "test"), (8, 4, 4), strict=True):
        split_probes = [probe for probe in probes if probe[1]["source_split"] == split]
        assert len(split_probes) == 2
        assert {record["source_dataset_uri"] for _, record in split_probes} == {
            spec.r2.split_lance_uri(split)
        }
        source_probe = next(
            root
            for root, record in split_probes
            if record["candidate_render"]["renderer_backend"] == "pedalboard"
        )
        candidate_probe = next(root for root, _ in split_probes if root != source_probe)
        _assert_target_audio_matches(source_probe, candidate_probe, expected_rows)


def _assert_target_audio_matches(
    source_probe: Path, candidate_probe: Path, expected_rows: int
) -> None:
    """Verify both roles consumed the same stored target waveform for every row.

    :param source_probe: Downloaded source-role split probe.
    :param candidate_probe: Downloaded candidate-role split probe.
    :param expected_rows: Required number of paired target WAVs.
    """
    source_targets = {
        path.parent.name: path for path in source_probe.glob("audio/sample_*/target.wav")
    }
    candidate_targets = {
        path.parent.name: path for path in candidate_probe.glob("audio/sample_*/target.wav")
    }
    assert len(source_targets) == expected_rows
    assert source_targets.keys() == candidate_targets.keys()
    for sample_name, source_target in source_targets.items():
        np.testing.assert_array_equal(
            _read_audio_file(source_target),
            _read_audio_file(candidate_targets[sample_name]),
        )
