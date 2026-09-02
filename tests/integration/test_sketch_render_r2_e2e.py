"""Production-path Surge sketch CLI checkpoint/VST/R2 round trip."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from pedalboard.io import AudioFile

from synth_setter.cli._cfg_strength import CfgStrengths
from synth_setter.cli.sketch_render import (
    _CACHE_NAMESPACE,
    _CHECKPOINT_SHA256,
    _STATS_SHA256,
    DEFAULT_CHECKPOINT_URI,
    DEFAULT_STATS_URI,
    _load_model,
    _predict_patch,
    _validate_stats,
    prepare_audio_inputs,
)
from synth_setter.model_cache import cache_r2_file
from synth_setter.pipeline import r2_io

pytestmark = [
    pytest.mark.integration_r2,
    pytest.mark.r2,
    pytest.mark.requires_vst,
    pytest.mark.slow,
]

_CHECKOUT_ROOT = Path(__file__).resolve().parents[2]
_EXPECTED_SAMPLES = 176400


def _write_inputs(
    root: Path, guide_frequency: float = 220.0, reference_frequency: float = 330.0
) -> tuple[Path, Path]:
    sample_rate = 48000
    time = np.arange(4 * sample_rate, dtype=np.float32) / sample_rate
    guide = (0.5 * np.sin(2 * np.pi * guide_frequency * time))[None]
    reference = np.stack(
        [
            0.4 * np.sin(2 * np.pi * reference_frequency * time),
            0.2 * np.sin(2 * np.pi * reference_frequency * time),
        ]
    ).astype(np.float32)
    guide_path = root / "guide.wav"
    ref_path = root / "reference.wav"
    with AudioFile(str(guide_path), "w", sample_rate, 1) as audio_file:
        audio_file.write(guide)
    with AudioFile(str(ref_path), "w", sample_rate, 2) as audio_file:
        audio_file.write(reference)
    return guide_path, ref_path


def _read_wav(path: Path) -> np.ndarray:
    with AudioFile(str(path), "r") as audio_file:
        assert audio_file.samplerate == 44100
        return audio_file.read(audio_file.frames)


def test_real_checkpoint_guide_and_reference_changes_alter_prediction(tmp_path: Path) -> None:
    """The pinned model prediction depends on both conditioning audio inputs.

    :param tmp_path: Holds guide/reference input pairs with isolated changes.
    """
    checkpoint_path = cache_r2_file(DEFAULT_CHECKPOINT_URI, _CACHE_NAMESPACE, _CHECKPOINT_SHA256)
    stats_path = cache_r2_file(DEFAULT_STATS_URI, _CACHE_NAMESPACE, _STATS_SHA256)
    _validate_stats(stats_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(checkpoint_path, device)

    base_root = tmp_path / "base"
    guide_root = tmp_path / "guide-change"
    reference_root = tmp_path / "reference-change"
    base_root.mkdir()
    guide_root.mkdir()
    reference_root.mkdir()
    base = prepare_audio_inputs(*_write_inputs(base_root), stats_path)
    guide_changed = prepare_audio_inputs(
        *_write_inputs(guide_root, guide_frequency=440.0), stats_path
    )
    reference_changed = prepare_audio_inputs(
        *_write_inputs(reference_root, reference_frequency=660.0), stats_path
    )

    requested_strengths = CfgStrengths(content=2.0, sketch=3.0)
    base_params, _, _ = _predict_patch(base, model, requested_strengths)
    guide_params, _, _ = _predict_patch(guide_changed, model, requested_strengths)
    reference_params, _, _ = _predict_patch(reference_changed, model, requested_strengths)

    assert guide_params != base_params
    assert reference_params != base_params


def test_installed_cli_real_checkpoint_surge_and_r2_produce_consumable_audio(
    tmp_path: Path,
) -> None:
    """The public audio command produces finite stereo audio after R2 download.

    :param tmp_path: Holds real input WAVs and downloaded output artifacts.
    """
    if r2_io.object_size(DEFAULT_CHECKPOINT_URI) is None:
        pytest.skip(f"required immutable checkpoint is absent: {DEFAULT_CHECKPOINT_URI}")
    guide_path, ref_path = _write_inputs(tmp_path)
    script = Path(sys.executable).parent / "synth-setter-sketch-render"

    result = subprocess.run(  # noqa: S603 — fixed installed public entrypoint
        [
            str(script),
            "--guide-audio",
            str(guide_path),
            "--reference-audio",
            str(ref_path),
            "--content-cfg-strength",
            "2",
            "--sketch-cfg-strength",
            "3",
        ],
        cwd=_CHECKOUT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=1200,
    )
    assert result.returncode == 0, (
        f"stdout tail:\n{result.stdout[-2000:]}\nstderr tail:\n{result.stderr[-4000:]}"
    )
    stdout_lines = [line for line in result.stdout.splitlines() if line.strip()]
    output_uri = stdout_lines[-1]
    assert output_uri.startswith("r2://intermediate-data/eval/synth-setter-sketch-render/")
    local_lines = [
        line for line in result.stderr.splitlines() if line.startswith("Local output: ")
    ]
    assert len(local_lines) == 1
    local_output = Path(local_lines[0].removeprefix("Local output: "))
    assert local_output.is_dir()

    downloaded = tmp_path / "downloaded"
    try:
        r2_io.download_dir_no_overwrite(output_uri, downloaded)
        assert sorted(path.name for path in downloaded.iterdir()) == [
            "guide.wav",
            "manifest.json",
            "params.csv",
            "pred.wav",
            "ref.wav",
        ]
        pred = _read_wav(downloaded / "pred.wav")
        normalized_guide = _read_wav(downloaded / "guide.wav")
        normalized_ref = _read_wav(downloaded / "ref.wav")
        for audio in (pred, normalized_guide, normalized_ref):
            assert audio.shape == (2, _EXPECTED_SAMPLES)
            assert np.isfinite(audio).all()
        assert np.abs(pred).max() > 1e-5
        assert np.abs(normalized_guide).max() > 1e-5
        assert np.abs(normalized_ref).max() > 1e-5
        with (downloaded / "params.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert rows
        assert {"pred", "pred_effective", "target"} <= set(rows[0])
        manifest = json.loads((downloaded / "manifest.json").read_text())
        assert manifest["content_cfg_strength"] == 2.0
        assert manifest["sketch_cfg_strength"] == 3.0
    finally:
        prefix = output_uri.removeprefix("r2://intermediate-data/") + "/"
        r2_io.purge_prefix("intermediate-data", prefix)
        shutil.rmtree(local_output, ignore_errors=True)
