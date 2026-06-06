"""Unit tests for eval's metric-IO helpers: ``_dump_metric_dict`` and ``_load_audio_metrics``."""

import json
from pathlib import Path

import pytest
import torch

from synth_setter.cli.eval import _dump_metric_dict, _load_audio_metrics

# Shared with ``test_eval_postprocessing.py``; duplicated rather than imported so
# each test module stays self-contained (a one-line literal isn't worth a shared import).
_AGGREGATED_METRICS_CSV = ",mean,std\nmss,0.5,0.1\nwmfcc,0.3,0.05\nsot,0.2,0.02\nrms,0.9,0.01\n"

_EXPECTED_AUDIO_METRICS = {
    "audio/mss_mean": pytest.approx(0.5),
    "audio/mss_std": pytest.approx(0.1),
    "audio/wmfcc_mean": pytest.approx(0.3),
    "audio/wmfcc_std": pytest.approx(0.05),
    "audio/sot_mean": pytest.approx(0.2),
    "audio/sot_std": pytest.approx(0.02),
    "audio/rms_mean": pytest.approx(0.9),
    "audio/rms_std": pytest.approx(0.01),
}


def test_dump_metric_dict_writes_json_with_coerced_scalars(tmp_path: Path) -> None:
    """Lightning tensors and numpy arrays are coerced to native floats / lists in ``metrics.json``.

    Pins the artifact downstream gates (workflow asserter, CSV joiners) read from —
    a torch / numpy dependency in those gates would force imports just to deserialize.

    :param tmp_path: Hydra-style output dir; the ``metrics/`` subdir lands under it.
    """

    import numpy as np

    metric_dict = {
        "test/param_mse": torch.tensor(0.0),
        "test/per_param_mse": torch.tensor([0.0, 0.0, 0.0, 0.0]),
        "audio/mss_mean": np.float32(0.5),
        "raw/string": "v1",
    }
    out_path = _dump_metric_dict(metric_dict, tmp_path)

    assert out_path == tmp_path / "metrics" / "metrics.json"
    assert out_path.is_file()
    payload = json.loads(out_path.read_text())
    assert payload["test/param_mse"] == 0.0
    assert payload["test/per_param_mse"] == [0.0, 0.0, 0.0, 0.0]
    assert payload["audio/mss_mean"] == pytest.approx(0.5)
    assert payload["raw/string"] == "v1"


def test_load_audio_metrics_flattens_mean_and_std(tmp_path: Path) -> None:
    """``aggregated_metrics.csv`` becomes a flat ``audio/<name>_<stat>`` float dict.

    :param tmp_path: Scratch metrics dir seeded with the fixture CSV.
    """
    (tmp_path / "aggregated_metrics.csv").write_text(_AGGREGATED_METRICS_CSV)

    metrics = _load_audio_metrics(tmp_path)

    assert metrics == _EXPECTED_AUDIO_METRICS


def test_load_audio_metrics_returns_python_floats(tmp_path: Path) -> None:
    """Values are plain ``float`` — protects downstream wandb / Lightning logs from numpy scalars.

    :param tmp_path: Scratch metrics dir seeded with the fixture CSV.
    """
    (tmp_path / "aggregated_metrics.csv").write_text(_AGGREGATED_METRICS_CSV)

    metrics = _load_audio_metrics(tmp_path)

    assert all(type(value) is float for value in metrics.values())


def test_load_audio_metrics_missing_csv_raises(tmp_path: Path) -> None:
    """Missing aggregated CSV surfaces a directed FileNotFoundError naming the subprocess.

    :param tmp_path: Used as a metrics dir intentionally left empty to trigger the guard.
    """
    with pytest.raises(
        FileNotFoundError,
        match=r"aggregated_metrics\.csv.*compute_audio_metrics.*did not write",
    ):
        _load_audio_metrics(tmp_path)


def test_load_audio_metrics_includes_shuffled_prefix_when_shuffled_csv_present(
    tmp_path: Path,
) -> None:
    """``aggregated_metrics_shuffled.csv`` present → keys prefixed ``shuffled_audio/`` merged in.

    :param tmp_path: Scratch metrics dir seeded with both CSVs.
    """
    (tmp_path / "aggregated_metrics.csv").write_text(_AGGREGATED_METRICS_CSV)
    (tmp_path / "aggregated_metrics_shuffled.csv").write_text(
        ",mean,std\nmss,1.0,0.2\nwmfcc,0.6,0.1\nsot,0.4,0.04\nrms,0.8,0.02\n"
    )

    metrics = _load_audio_metrics(tmp_path)

    assert "shuffled_audio/mss_mean" in metrics
    assert metrics["shuffled_audio/mss_mean"] == pytest.approx(1.0)
    assert "shuffled_audio/mss_std" in metrics
    assert metrics["shuffled_audio/mss_std"] == pytest.approx(0.2)
    assert "audio/mss_mean" in metrics


def test_load_audio_metrics_skips_shuffled_prefix_when_no_shuffled_csv(tmp_path: Path) -> None:
    """No ``aggregated_metrics_shuffled.csv`` → no ``shuffled_audio/`` keys in result.

    :param tmp_path: Scratch metrics dir with only the normal CSV.
    """
    (tmp_path / "aggregated_metrics.csv").write_text(_AGGREGATED_METRICS_CSV)

    metrics = _load_audio_metrics(tmp_path)

    assert not any(k.startswith("shuffled_audio/") for k in metrics)


def test_load_audio_metrics_shuffled_values_are_python_floats(tmp_path: Path) -> None:
    """Shuffled metric values are plain ``float`` — consistent with the normal path.

    :param tmp_path: Scratch metrics dir seeded with both CSVs.
    """
    (tmp_path / "aggregated_metrics.csv").write_text(_AGGREGATED_METRICS_CSV)
    (tmp_path / "aggregated_metrics_shuffled.csv").write_text(
        ",mean,std\nmss,1.0,0.2\nwmfcc,0.6,0.1\nsot,0.4,0.04\nrms,0.8,0.02\n"
    )

    metrics = _load_audio_metrics(tmp_path)

    shuffled = {k: v for k, v in metrics.items() if k.startswith("shuffled_audio/")}
    assert all(type(v) is float for v in shuffled.values())


def test_load_audio_metrics_missing_stat_column_raises(tmp_path: Path) -> None:
    """A CSV lacking a required stat column surfaces a directed ValueError naming the gap.

    :param tmp_path: Scratch metrics dir seeded with a mean-only CSV.
    """
    (tmp_path / "aggregated_metrics.csv").write_text(",mean\nmss,0.5\n")

    with pytest.raises(ValueError, match=r"missing required stat columns \['std'\]"):
        _load_audio_metrics(tmp_path)
