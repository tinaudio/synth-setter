"""CLI plumbing smoke tests for src/data/vst/generate_vst_dataset.py.

Each test invokes the click ``main`` entry with one flag per ``RenderConfig``
field, with ``render_params`` monkeypatched to return a fast silent buffer so
the tests don't need a real VST plugin. Assertions look at the produced on-disk
artifact (h5 datasets / tar members) rather than mocking ``make_hdf5_dataset``
/ ``make_wds_dataset`` directly.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import h5py
import numpy as np
import pytest
from click.testing import CliRunner

from src.data.vst import generate_vst_dataset
from src.data.vst.generate_vst_dataset import main
from src.pipeline.schemas.spec import ShardMetadata

_SAMPLE_RATE = 16000
_CHANNELS = 2
_DURATION = 4.0
_NUM_SAMPLES = 2


def _render_cfg_args(num_samples: int = _NUM_SAMPLES) -> list[str]:
    """Build the per-field CLI flags for ``main`` (one per RenderConfig field)."""
    return [
        "--plugin-path", "plugins/Surge XT.vst3",
        "--preset-path", "presets/surge-base.vstpreset",
        "--param-spec-name", "surge_simple",
        "--renderer-version", "test",
        "--sample-rate", str(_SAMPLE_RATE),
        "--channels", str(_CHANNELS),
        "--velocity", "100",
        "--signal-duration-seconds", str(_DURATION),
        "--min-loudness", "-99.0",
        "--sample-batch-size", "1",
        "--batch-per-shard", str(num_samples),
    ]


@pytest.fixture()
def stub_render_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ``render_params`` with a fast silent stereo render."""
    audio_shape = (_CHANNELS, int(_SAMPLE_RATE * _DURATION))
    monkeypatch.setattr(
        generate_vst_dataset,
        "render_params",
        lambda *_args, **_kwargs: 0.5 * np.ones(audio_shape, dtype=np.float32),
    )


def test_h5_extension_writes_h5_with_expected_datasets(
    tmp_path: Path, stub_render_params: None
) -> None:
    """A ``.h5`` data_file produces an HDF5 file with the three expected datasets."""
    h5_path = tmp_path / "out.h5"
    runner = CliRunner()

    result = runner.invoke(
        main,
        [str(h5_path), *_render_cfg_args()],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert h5_path.exists()
    with h5py.File(h5_path, "r") as f:
        assert set(f.keys()) >= {"audio", "mel_spec", "param_array"}
        audio = f["audio"]
        assert isinstance(audio, h5py.Dataset)
        assert audio.shape[0] == _NUM_SAMPLES
        assert audio.dtype == np.float16


def test_tar_extension_writes_tar_with_expected_members(
    tmp_path: Path, stub_render_params: None
) -> None:
    """A ``.tar`` data_file produces a tar with per-batch members and metadata.json."""
    tar_path = tmp_path / "out.tar"
    runner = CliRunner()

    result = runner.invoke(
        main,
        [str(tar_path), *_render_cfg_args()],
        catch_exceptions=False,
    )

    assert result.exit_code == 0, result.output
    assert tar_path.exists()
    with tarfile.open(tar_path) as tar:
        names = {m.name for m in tar.getmembers()}
    assert names >= {
        "00000000.audio.npy",
        "00000000.mel_spec.npy",
        "00000000.param_array.npy",
        "00000001.audio.npy",
        "00000001.mel_spec.npy",
        "00000001.param_array.npy",
        "metadata.json",
    }


def test_tar_metadata_member_parses_as_shard_metadata(
    tmp_path: Path, stub_render_params: None
) -> None:
    """The tar's metadata.json parses as a ShardMetadata carrying the CLI's values."""
    tar_path = tmp_path / "out.tar"
    runner = CliRunner()

    runner.invoke(
        main,
        [str(tar_path), *_render_cfg_args()],
        catch_exceptions=False,
    )

    with tarfile.open(tar_path) as tar:
        extracted = tar.extractfile("metadata.json")
        assert extracted is not None
        meta = ShardMetadata.model_validate_json(extracted.read())

    assert meta.velocity == 100
    assert meta.sample_rate == _SAMPLE_RATE
    assert meta.channels == _CHANNELS
    assert meta.signal_duration_seconds == _DURATION
    assert meta.min_loudness == -99.0


def test_tar_audio_member_dtype_is_float16(
    tmp_path: Path, stub_render_params: None
) -> None:
    """The tar's audio.npy member is stored as float16 to match h5 storage precision."""
    tar_path = tmp_path / "out.tar"
    runner = CliRunner()

    runner.invoke(
        main,
        [str(tar_path), *_render_cfg_args()],
        catch_exceptions=False,
    )

    with tarfile.open(tar_path) as tar:
        extracted = tar.extractfile("00000000.audio.npy")
        assert extracted is not None
        audio = np.load(io.BytesIO(extracted.read()))

    assert audio.dtype == np.float16


@pytest.mark.parametrize("filename", ["out.parquet", "out"])
def test_unknown_extension_is_rejected_with_supported_suffixes_listed(
    tmp_path: Path, stub_render_params: None, filename: str
) -> None:
    """A data_file with an unsupported (or absent) extension exits non-zero, naming both supported
    suffixes."""
    runner = CliRunner()

    result = runner.invoke(
        main,
        [str(tmp_path / filename), *_render_cfg_args()],
    )

    assert result.exit_code != 0
    assert "data_file must end in" in result.output
    assert ".h5" in result.output
    assert ".tar" in result.output
